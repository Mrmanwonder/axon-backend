import { pipelineWrite } from "@mastery/shared/pipeline_write.js";
import { callModel } from "@mastery/shared/model-client.js";
import { failureCodeFor } from "@mastery/shared/failure_codes.js";
import { consumeQueue, failRun } from "@mastery/shared/worker.js";
import { imageRef } from "@mastery/shared/r2.js";
import { planPage, regionsWrittenByPage, structureFailureReason, uniqueLabelKey } from "@mastery/shared/structure_plan.js";
import { pageDimensions, UNPLACEABLE_PAGE_REASON } from "@mastery/shared/page.js";
import { attribute, type RawMark } from "@mastery/shared/attribution.js";
import { mustData, mustOk, mustRpc, mustMaybe } from "@mastery/shared/db.js";
import { SYSTEM, instruction, SCHEMA, validate } from "@mastery/shared/prompts/structure.v1.js";
import { loadStructurePage } from "@mastery/shared/structure-page.js";
import { ConfigurationError } from "@mastery/shared/errors.js";
import type { Env } from "@mastery/shared/env.js";
import { chunkedSendBatch } from "@mastery/shared/chunked_send.js";

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
    await chunkedSendBatch(
      env.CROP_QUEUE,
      pageIds,
      (pageId) => ({ body: { run_id: runId, page_id: pageId } }),
    );
  } else if (regionIds.length) {
    if (!env.CONTENT_QUEUE) throw new Error("Content queue is not configured");
    await chunkedSendBatch(
      env.CONTENT_QUEUE,
      regionIds,
      (regionId) => ({ body: { run_id: runId, region_id: regionId } }),
    );
  }

  if (advance.enqueue_reconcile) {
    if (!env.RECONCILE_QUEUE) throw new Error("Reconcile queue is not configured");
    await env.RECONCILE_QUEUE.send({ run_id: runId });
  }
}

const handler = consumeQueue<StructureMessage>(
  async ({ env, sb, msg, attempt, beat }) => {
    const runId = msg.run_id;
    const pageId = msg.page_id;
    const page = await loadStructurePage(sb, pageId);
    if (!page) return { detail: { skipped: "no such page" } };

    const run = await mustMaybe<{ status: string; route_override: any }>(sb.from("extraction_run").select("status, route_override").eq("id", runId).maybeSingle(), "structure run read");
    if (!run || ["failed", "rejected", "committed", "needs_review", "ready", "explaining"].includes(run.status)) {
      return { detail: { skipped: run?.status ?? "no run" } };
    }

    if (["done", "unreadable", "failed"].includes(page.structure_status)) {
      // The "already done" path must still advance the run — otherwise a
      // second pass over an already-structured page dead-ends the run
      // instead of moving it forward. See AXON_FIX_BRIEF.md §3.2.
      const adv0 = await mustRpc(sb.rpc("advance_after_structure", { p_run_id: runId }), "advance_after_structure");
      await enqueueFromAdvance(env, runId, adv0 ?? {});
      return { detail: { skipped: "already done" } };
    }

    const override = run.route_override;

    if (!await pipelineWrite(sb, runId, "structure", { page_id: pageId, patch: { structure_status: "running" } })) return { detail: { skipped: "stale structure work" } };
    await beat();

    const countResult = await sb.from("paper_page").select("id", { count: "exact", head: true }).eq("paper_id", page.paper_id);
    await mustOk(Promise.resolve(countResult), "structure page count");
    const pageCount = countResult.count;

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
      if (!await pipelineWrite(sb, runId, "structure", { page_id: pageId, patch: { structure_status: "unreadable" },
        unreadable_reason: parsed.not_a_paper_reason ?? "This page does not look like part of a marked exam paper." })) return { detail: { skipped: "stale structure result" } };
      const advance2 = await mustRpc(sb.rpc("advance_after_structure", { p_run_id: runId }), "advance_after_structure");
      await enqueueFromAdvance(env, runId, advance2 ?? {});
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
      if (!await pipelineWrite(sb, runId, "structure", { page_id: pageId, patch: { structure_status: "unreadable" },
        unreadable_reason: UNPLACEABLE_PAGE_REASON })) return { detail: { skipped: "stale structure result" } };
      const advanceNoDims = await mustRpc(sb.rpc("advance_after_structure", { p_run_id: runId }), "advance_after_structure");
      await enqueueFromAdvance(env, runId, advanceNoDims ?? {});
      return { detail: { unreadable: "no page dimensions" } };
    }
    const { width, height } = dims;

    // A redelivered page starts from nothing. Queue delivery is at-least-once,
    // so an attempt can die after its regions landed and before the page was
    // marked done; the retry used to insert the same questions a second time
    // (and collide on the label index). Everything this page wrote for this
    // run is removed first: its marks, then the regions whose first span is
    // this page. Assembly is run-level now, so no other region carries a span
    // written by this page until every page is done.
    await mustOk(
      sb.from("teacher_mark").delete().eq("run_id", runId).eq("page_number", page.page_number),
      "clear this page's earlier teacher marks",
    );
    const runRegions = await mustData(
      sb.from("question_region").select("id, order_index, page_spans, question_label").eq("run_id", runId),
      "run regions read",
    ) as Array<{ id: string; order_index: number; page_spans: unknown; question_label: string | null }>;
    const stale = regionsWrittenByPage(runRegions, page.page_number);
    if (stale.length) {
      await mustOk(sb.from("question_region").delete().in("id", stale), "clear this page's earlier regions");
    }
    const kept = runRegions.filter((r) => !stale.includes(r.id));
    const nextIndex = kept.reduce((m, r) => Math.max(m, r.order_index + 1), 0);
    const takenLabels = new Set(
      kept.map((r) => uniqueLabelKey(r.question_label)).filter((k): k is string => k !== null),
    );

    const plan = planPage({
      regions: parsed.regions,
      page: page.page_number,
      width,
      height,
      runId,
      paperId: page.paper_id,
      studentId: page.student_id,
      nextIndex,
      takenLabels,
    });

    const created: Array<{ id: string; order_index: number; spans: unknown[] }> = [];
    /** Everything on THIS page a teacher mark could belong to. A continuation
        band is one of them: it is written as its own region here and merged
        into the question it continues by `private.assemble_structure`, which
        moves its marks with it. */
    const candidates: Array<{ id: string; order_index: number; spans: unknown[] }> = [];

    if (plan.length) {
      // Checked. The (run_id, order_index) race this worker still has is
      // contained by max_concurrency=1; a collision throws and the retry
      // starts clean (above) rather than marking the page done without its
      // questions.
      const insertedRows = await mustData(
        sb.from("question_region").insert(plan.map((t) => t.row)).select("id, order_index"),
        "question_region insert",
      ) as Array<{ id: string; order_index: number }>;
      const byOrder = new Map((insertedRows ?? []).map((r) => [r.order_index, r]));
      for (const t of plan) {
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
        // No persisted margin band: leave glyph classification uncertain.
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
    if (!await pipelineWrite(sb, runId, "structure", { page_id: pageId, patch: { structure_status: "done" } })) return { detail: { skipped: "stale structure result" } };
    const advance = await mustRpc(sb.rpc("advance_after_structure", { p_run_id: runId }), "advance_after_structure") as any;
    await enqueueFromAdvance(env, runId, advance ?? {});

    return { detail: { regions: created.length, marks: marks.length } };
  },
  async ({ env, sb, msg }, error) => {
    if (error instanceof ConfigurationError) {
      await failRun(sb, msg.run_id, "A processing service could not read this paper. Your pages are kept. Please try again later.", failureCodeFor("structure", error));
      return;
    }
    const pageId = msg.page_id;
    const page = await mustMaybe<{ paper_id: string; student_id: string; page_number: number; r2_key: string }>(sb.from("paper_page").select("paper_id, student_id, page_number, r2_key").eq("id", pageId).maybeSingle(), "structure failure page read");
    if (page) {
      if (!await pipelineWrite(sb, msg.run_id, "structure", { page_id: pageId, patch: { structure_status: "failed" },
        unreadable_reason: structureFailureReason(error) })) return;
      const advance = await mustRpc(sb.rpc("advance_after_structure", { p_run_id: msg.run_id }), "advance_after_structure");
      await enqueueFromAdvance(env, msg.run_id, advance ?? {});
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
          : "We could not find the pages for this paper. Try scanning it again.",
        pagesStored ? failureCodeFor("structure", error) : "structure_pages_missing",
      );
    }
  }
);

export default { queue: handler } satisfies ExportedHandler<Env, StructureMessage>;
