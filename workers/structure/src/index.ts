import { callModel } from "@mastery/shared/openrouter.js";
import { consumeQueue, failRun } from "@mastery/shared/worker.js";
import { imageRef } from "@mastery/shared/r2.js";
import { takeBox } from "@mastery/shared/contract.js";
import { attribute, type RawMark } from "@mastery/shared/attribution.js";
import { SYSTEM, instruction, SCHEMA, validate } from "@mastery/shared/prompts/structure.v1.js";
import type { Env } from "@mastery/shared/env.js";

interface StructureMessage {
  run_id: string;
  page_id: string;
  _retries?: number;
}

interface AdvanceResult {
  advanced?: boolean;
  enqueue_content?: string[];
  enqueue_reconcile?: boolean;
}

async function enqueueFromAdvance(env: Env, runId: string, advance: AdvanceResult): Promise<void> {
  const regionIds = advance.enqueue_content ?? [];
  if (env.CONTENT_QUEUE && regionIds.length) {
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
      .select("id, paper_id, student_id, page_number, r2_bucket, r2_key, mask_key, structure_status, layer_fallback, teacher_marks, conditioning_meta")
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

    const meta = (page.conditioning_meta as any) ?? {};
    const width = meta.width ?? 2400;
    const height = meta.height ?? 3200;

    const { data: existing } = await sb
      .from("question_region")
      .select("id, order_index, page_spans")
      .eq("run_id", runId)
      .order("order_index", { ascending: false })
      .limit(1);
    let nextIndex = existing?.length ? existing[0].order_index + 1 : 0;

    const created: Array<{ id: string; order_index: number; spans: unknown[] }> = [];
    const toInsert: Array<{ order_index: number; span: unknown; row: Record<string, unknown> }> = [];

    for (const [i, region] of parsed.regions.entries()) {
      const box = takeBox(region.box, page.page_number, width, height);
      if (!box) continue;
      const span = { page: page.page_number, box: { x: box.x, y: box.y, w: box.w, h: box.h } };

      if (i === 0 && region.continues_from_previous && existing?.length) {
        const prior = existing[0];
        const spans = [...prior.page_spans, span];
        await sb.from("question_region").update({ page_spans: spans }).eq("id", prior.id);
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
      const { data: insertedRows } = await sb.from("question_region").insert(toInsert.map((t) => t.row)).select("id, order_index");
      const byOrder = new Map((insertedRows ?? []).map((r: any) => [r.order_index, r]));
      for (const t of toInsert) {
        const row = byOrder.get(t.order_index);
        if (row) created.push({ id: row.id, order_index: row.order_index, spans: [t.span] });
      }
    }

    const marks: RawMark[] = page.teacher_marks ?? [];
    if (marks.length && created.length) {
      const regions = created.map((c, i) => ({ order_index: i, label: null, spans: c.spans as any }));
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
        region_id: m.region_index === null ? null : created[m.region_index]?.id ?? null,
        page_number: m.page_number,
        box: m.box,
        shape: m.shape,
        mark_class: m.mark_class,
        metrics: m.metrics,
        confidence_tier: "unsure",
      }));
      if (rows.length) await sb.from("teacher_mark").insert(rows);
    }

    await sb.from("paper_page").update({ structure_status: "done" }).eq("id", pageId);
    const { data: advance } = await sb.rpc("advance_after_structure", { p_run_id: runId });
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
