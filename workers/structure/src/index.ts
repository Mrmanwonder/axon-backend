import { callModel } from "@mastery/shared/openrouter.js";
import { consumeQueue, failRun } from "@mastery/shared/worker.js";
import { imageRef } from "@mastery/shared/r2.js";
import { takeBox } from "@mastery/shared/contract.js";
import { pageDimensions, UNPLACEABLE_PAGE_REASON } from "@mastery/shared/page.js";
import { attribute, type RawMark } from "@mastery/shared/attribution.js";
import { mustData, mustOk, mustRpc } from "@mastery/shared/db.js";
import { SYSTEM, instruction, SCHEMA, validate } from "@mastery/shared/prompts/structure.v1.js";
import type { Env } from "@mastery/shared/env.js";

interface StructureMessage {
  run_id: string;
  page_id: string;
  _retries?: number;
}

interface AdvanceResult {
  advanced?: boolean;
  /** Page ids for the crop stage (AXON_FIX_BRIEF.md §8), when the crop stage is
      switched on. */
  enqueue_crop?: string[];
  /** Region ids straight to content, when it is not. */
  enqueue_content?: string[];
  enqueue_reconcile?: boolean;
}

/**
 * Fan out whatever `advance_after_structure` says comes next.
 *
 * Both shapes are handled, and that is the point. The crop stage sits behind a
 * flag in the database (`private.feature_flag`, key `crop_stage`), so this
 * function is called with region ids while the flag is off and page ids once it
 * is on — and it has to be correct either way, because the flag can be flipped
 * without redeploying anything.
 *
 * That symmetry is what was missing when WP4's schema change went to the live
 * database ahead of this worker: the old deployed structure worker understood
 * only `enqueue_content`, the new function returned only `enqueue_crop`, and a
 * run advanced to 'cropping' with nothing to consume it. It sat there until the
 * sweep failed it ten minutes later. Nothing was lost — no paper was submitted
 * in the window — but the ordering requirement was real and it should never
 * have existed. Reading both keys is what removes it.
 *
 * Crop first: if the flag is on, the run has been advanced to 'cropping' and
 * content must not be enqueued behind its back.
 */
async function enqueueFromAdvance(env: Env, runId: string, advance: AdvanceResult): Promise<void> {
  const pageIds = advance.enqueue_crop ?? [];
  const regionIds = advance.enqueue_content ?? [];

  if (pageIds.length) {
    if (!env.CROP_QUEUE) {
      // The database says crop, this worker cannot. Loud rather than silent:
      // the run is in 'cropping' and only the sweep will move it now, so the
      // one useful thing left is to say exactly which flag disagrees with which
      // binding (§10 — do not swallow errors).
      throw new Error(
        `run ${runId}: advance_after_structure returned ${pageIds.length} page(s) for the crop stage, ` +
        "but this worker has no CROP_QUEUE binding. Turn private.feature_flag 'crop_stage' off, " +
        "or deploy a structure worker that binds crop-queue."
      );
    }
    await env.CROP_QUEUE.sendBatch(pageIds.map((pageId) => ({ body: { run_id: runId, page_id: pageId } })));
  } else if (regionIds.length && env.CONTENT_QUEUE) {
    await env.CONTENT_QUEUE.sendBatch(regionIds.map((regionId) => ({ body: { run_id: runId, region_id: regionId } })));
  }

  if (advance.enqueue_reconcile && env.RECONCILE_QUEUE) {
    await env.RECONCILE_QUEUE.send({ run_id: runId });
  }
}

const handler = consumeQueue<StructureMessage>(
  async ({ env, sb, msg, attempt, beat }) => {
    const runId = msg.run_id;
    const pageId = msg.page_id;
    const { data: page } = await sb
      .from("paper_page")
      .select("id, paper_id, student_id, page_number, r2_bucket, r2_key, mask_key, structure_status, layer_fallback, teacher_marks, conditioning_meta, quality_signals")
      .eq("id", pageId)
      .single();
    if (!page) return { detail: { skipped: "no such page" } };

    if (page.structure_status === "done") {
      // The "already done" path must still advance the run — otherwise a
      // second pass over an already-structured page dead-ends the run
      // instead of moving it forward. See AXON_FIX_BRIEF.md §3.2.
      const { data: adv0 } = await sb.rpc("advance_after_structure", { p_run_id: runId });
      if (adv0?.advanced) await enqueueFromAdvance(env, runId, adv0);
      return { detail: { skipped: "already done" } };
    }

    const { data: run } = await sb.from("extraction_run").select("status, route_override").eq("id", runId).single();
    if (!run || ["failed", "rejected", "committed"].includes(run.status)) {
      return { detail: { skipped: run?.status ?? "no run" } };
    }
    const override = run.route_override;

    await sb.from("paper_page").update({ structure_status: "running" }).eq("id", pageId);
    await beat();

    const { count: pageCount } = await sb.from("paper_page").select("id", { count: "exact", head: true }).eq("paper_id", page.paper_id);

    const images = [await imageRef(env, (page.r2_bucket as any) ?? "derived", page.r2_key, "high")];
    if (page.mask_key) {
      images.push(await imageRef(env, "derived", page.mask_key, "low"));
    }

    const { parsed } = await callModel({
      env,
      sb,
      stage: "structure",
      system: SYSTEM,
      instruction: instruction(page.page_number, pageCount ?? 1),
      images,
      schema: SCHEMA,
      validate,
      runId,
      paperId: page.paper_id,
      studentId: page.student_id,
      attempt,
      routeOverride: override,
    });

    if (!parsed.is_graded_exam_paper) {
      await sb.from("page_unreadable").insert({
        paper_id: page.paper_id,
        page_number: page.page_number,
        storage_path: page.r2_key,
        reason: parsed.not_a_paper_reason ?? "This page does not look like part of a marked exam paper.",
      });
      await sb.from("paper_page").update({ structure_status: "unreadable" }).eq("id", pageId);
      const { data: advance2 } = await sb.rpc("advance_after_structure", { p_run_id: runId });
      if (advance2?.advanced) await enqueueFromAdvance(env, runId, advance2);
      return { detail: { unreadable: true } };
    }

    // Every box below is scaled against these two numbers. There is no default
    // for them any more: `?? 2400` / `?? 3200` was not a fallback, it was the
    // only branch that ever ran, and it silently mis-scaled every box on every
    // page in the database. A page whose size cannot be established is a page
    // nothing can be placed on, and that is said out loud rather than papered
    // over. See @mastery/shared/page.ts.
    const dims = pageDimensions(page as any);
    if (!dims) {
      await sb.from("page_unreadable").insert({
        paper_id: page.paper_id,
        page_number: page.page_number,
        storage_path: page.r2_key,
        reason: UNPLACEABLE_PAGE_REASON,
      });
      await sb.from("paper_page").update({ structure_status: "unreadable" }).eq("id", pageId);
      const { data: advanceNoDims } = await sb.rpc("advance_after_structure", { p_run_id: runId });
      if (advanceNoDims?.advanced) await enqueueFromAdvance(env, runId, advanceNoDims);
      return { detail: { unreadable: "no page dimensions" } };
    }
    const { width, height } = dims;

    const existing = await mustData(
      sb.from("question_region")
        .select("id, order_index, page_spans")
        .eq("run_id", runId)
        .order("order_index", { ascending: false })
        .limit(1),
      "highest order_index read",
    ) as Array<{ id: string; order_index: number; page_spans: unknown[] }>;
    let nextIndex = existing?.length ? existing[0].order_index + 1 : 0;

    const created: Array<{ id: string; order_index: number; spans: unknown[] }> = [];
    /** Everything on THIS page a teacher mark could belong to: the questions
        created below, plus any prior question stitched onto this page. */
    const candidates: Array<{ id: string; order_index: number; spans: unknown[] }> = [];
    const toInsert: Array<{ order_index: number; span: unknown; row: Record<string, unknown> }> = [];

    for (const [i, region] of parsed.regions.entries()) {
      const box = takeBox(region.box, page.page_number, width, height);
      if (!box) continue;
      const span = { page: page.page_number, box: { x: box.x, y: box.y, w: box.w, h: box.h } };

      if (i === 0 && region.continues_from_previous && existing?.length) {
        const prior = existing[0];
        const spans = [...prior.page_spans, span];
        await mustOk(
          sb.from("question_region").update({ page_spans: spans }).eq("id", prior.id),
          "stitch continuation span",
        );
        // The stitched question is a candidate for THIS page's teacher marks,
        // and it used to `continue` straight past this — never entering
        // `created`, which is the only list mark attribution looks at.
        //
        // Concretely: page 2 carries the tail of Q3 and then Q4. The teacher
        // writes 4 in the margin beside the Q3 tail and 2 beside Q4. With Q3
        // absent from the candidates, the 4 either attaches to nothing or, far
        // worse, to Q4 — a plausible, confidently wrong mark on the wrong
        // question, which is exactly the failure hard rule 1 exists to prevent.
        //
        // Its span here is this page's band only. Attribution is per-page
        // geometry, so handing it the question's earlier bands on other pages
        // would compare a margin mark against a box that is not on this page.
        // It goes first because a continuation is always the top band.
        candidates.push({ id: prior.id, order_index: prior.order_index, spans: [span] });
        continue;
      }

      const numberBox = takeBox(region.number_box, page.page_number, width, height);
      const label = numberBox ? region.candidate_number : null;
      toInsert.push({
        order_index: nextIndex,
        span,
        row: {
          run_id: runId,
          paper_id: page.paper_id,
          student_id: page.student_id,
          order_index: nextIndex,
          page_spans: [span],
          question_label: label,
          question_label_box: label ? numberBox : null,
          confidence_tier: "unsure",
        },
      });
      nextIndex += 1;
    }

    if (toInsert.length) {
      // Checked. An insert that collided on the unique (run_id, order_index) —
      // the race this worker still has, contained for now by max_concurrency=1
      // — used to be dropped on the floor here, and the page was marked done
      // with its questions missing.
      const insertedRows = await mustData(
        sb.from("question_region").insert(toInsert.map((t) => t.row)).select("id, order_index"),
        "question_region insert",
      ) as Array<{ id: string; order_index: number }>;
      const byOrder = new Map((insertedRows ?? []).map((r) => [r.order_index, r]));
      for (const t of toInsert) {
        const row = byOrder.get(t.order_index);
        if (row) {
          created.push({ id: row.id, order_index: row.order_index, spans: [t.span] });
          candidates.push({ id: row.id, order_index: row.order_index, spans: [t.span] });
        }
      }
    }

    const marks: RawMark[] = page.teacher_marks ?? [];
    if (marks.length && candidates.length) {
      const regions = candidates.map((c, i) => ({ order_index: i, label: null, spans: c.spans as any }));
      const attributed = attribute({
        regions,
        marks,
        marginBands: new Map([[page.page_number, null]]),
        pageWidths: new Map([[page.page_number, width]]),
      });
      const rows = attributed.map((m) => ({
        run_id: runId,
        paper_id: page.paper_id,
        student_id: page.student_id,
        region_id: m.region_index === null ? null : candidates[m.region_index]?.id ?? null,
        page_number: m.page_number,
        box: m.box,
        shape: m.shape,
        mark_class: m.mark_class,
        metrics: m.metrics,
        confidence_tier: "unsure",
      }));
      if (rows.length) await mustOk(sb.from("teacher_mark").insert(rows), "teacher_mark insert");
    }

    // After the writes above have all landed, never before: `done` is a claim
    // that this page's questions and marks are in the database.
    await mustOk(sb.from("paper_page").update({ structure_status: "done" }).eq("id", pageId), "structure_status=done");
    const advance = await mustRpc(sb.rpc("advance_after_structure", { p_run_id: runId }), "advance_after_structure") as any;
    if (advance?.advanced) await enqueueFromAdvance(env, runId, advance);

    return { detail: { regions: created.length, marks: marks.length } };
  },
  async ({ env, sb, msg }) => {
    const pageId = msg.page_id;
    const { data: page } = await sb.from("paper_page").select("paper_id, page_number, r2_key").eq("id", pageId).maybeSingle();
    if (page) {
      await sb.from("page_unreadable").insert({
        paper_id: page.paper_id,
        page_number: page.page_number,
        storage_path: page.r2_key,
        reason: "We could not read this page well enough to find the questions on it.",
      });
      await sb.from("paper_page").update({ structure_status: "failed" }).eq("id", pageId);
      const { data: advance } = await sb.rpc("advance_after_structure", { p_run_id: msg.run_id });
      if (advance?.advanced) await enqueueFromAdvance(env, msg.run_id, advance);
    } else {
      let pagesStored = false;
      if (msg.run_id) {
        const { data: run } = await sb.from("extraction_run").select("paper_id").eq("id", msg.run_id).maybeSingle();
        if (run?.paper_id) {
          const { count } = await sb.from("paper_page").select("id", { count: "exact", head: true }).eq("paper_id", run.paper_id).not("r2_key", "is", null);
          pagesStored = (count ?? 0) > 0;
        }
      }
      await failRun(
        sb,
        msg.run_id,
        pagesStored
          ? "We couldn't finish reading this paper's questions just now — your pages are kept, and you can try again."
          : "We could not find the pages for this paper. Try scanning it again."
      );
    }
  }
);

export default { queue: handler } satisfies ExportedHandler<Env, StructureMessage>;
