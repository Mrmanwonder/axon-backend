import { callModel } from "@mastery/shared/openrouter.js";
import { consumeQueue } from "@mastery/shared/worker.js";
import { imageRef } from "@mastery/shared/r2.js";
import { takeBox } from "@mastery/shared/contract.js";
import { pageDimensions } from "@mastery/shared/page.js";
import { SYSTEM, instruction, SCHEMA, validate } from "@mastery/shared/prompts/content.v1.js";
import type { Env } from "@mastery/shared/env.js";

interface ContentMessage {
  run_id: string;
  region_id: string;
  _retries?: number;
}

async function advanceAndEnqueue(env: Env, sb: any, runId: string): Promise<void> {
  const { data: advance } = await sb.rpc("advance_after_content", { p_run_id: runId });
  if (advance?.advanced && advance?.enqueue_reconcile && env.RECONCILE_QUEUE) {
    await env.RECONCILE_QUEUE.send({ run_id: runId });
  }
}

function field(v: { value: unknown; box: unknown; page_index?: number } | null | undefined, spans: Array<{ page: number }>, width: number, height: number) {
  if (!v || v.value === null || v.value === undefined) return { value: null, box: null };
  const page = spans[Math.min(v.page_index ?? 0, spans.length - 1)]?.page ?? spans[0].page;
  const box = takeBox(v.box, page, width, height);
  return box ? { value: v.value, box } : { value: null, box: null };
}

const handler = consumeQueue<ContentMessage>(
  async ({ env, sb, msg, attempt, beat }) => {
    const runId = msg.run_id;
    const regionId = msg.region_id;
    const { data: region } = await sb
      .from("question_region")
      .select("id, paper_id, student_id, order_index, question_label, question_label_box, page_spans, extract_status, crop_key, cropmask_key")
      .eq("id", regionId)
      .single();
    if (!region) return { detail: { skipped: "no such question" } };

    if (region.extract_status === "done") {
      await advanceAndEnqueue(env, sb, runId);
      return { detail: { skipped: "already read" } };
    }

    const { data: run } = await sb.from("extraction_run").select("status, route_override").eq("id", runId).single();
    if (!run || ["failed", "rejected", "committed"].includes(run.status)) {
      return { detail: { skipped: run?.status ?? "no run" } };
    }
    const override = run.route_override;

    await sb.from("question_region").update({ extract_status: "running" }).eq("id", regionId);
    await beat();

    const spans: Array<{ page: number }> = region.page_spans ?? [];
    if (!spans.length) throw new Error("a question with no page span has nothing to read");

    // Prefer a pre-cut crop (§8 of AXON_FIX_BRIEF.md) over sending the whole
    // page. crop_key is null on every region today, so this always falls
    // back to the full-page path — that gap is what §8 closes, not this file.
    const images = region.crop_key
      ? (
          await Promise.all([
            imageRef(env, "derived", region.crop_key, "high"),
            region.cropmask_key ? imageRef(env, "derived", region.cropmask_key, "low") : Promise.resolve(null),
          ])
        ).filter((x): x is NonNullable<typeof x> => !!x)
      : await (async () => {
          const { data: pages } = await sb
            .from("paper_page")
            .select("page_number, r2_bucket, r2_key, mask_key, layer_fallback")
            .eq("paper_id", region.paper_id)
            .in("page_number", spans.map((s) => s.page))
            .order("page_number");
          return Promise.all((pages ?? []).filter((p: any) => p.r2_key).map((p: any) => imageRef(env, p.r2_bucket ?? "derived", p.r2_key, "high")));
        })();
    if (!images.length) throw new Error("nothing to look at for this question");

    const { data: marks } = await sb.from("teacher_mark").select("shape, mark_class, page_number").eq("region_id", regionId);
    const { data: firstPage } = await sb
      .from("paper_page")
      .select("layer_fallback, conditioning_meta, quality_signals")
      .eq("paper_id", region.paper_id)
      .eq("page_number", spans[0].page)
      .maybeSingle();

    const { parsed } = await callModel({
      env,
      sb,
      stage: "content",
      system: SYSTEM,
      instruction: instruction({
        label: region.question_label,
        pageNumbers: spans.map((s) => s.page),
        layerFallback: firstPage?.layer_fallback ?? null,
        teacherMarks: (marks ?? []).map((m: any) => ({ shape: m.shape, where: `on page ${m.page_number}` })),
      }),
      images,
      schema: SCHEMA,
      validate,
      runId,
      paperId: region.paper_id,
      regionId,
      studentId: region.student_id,
      attempt,
      routeOverride: override,
    });

    try {
      // See @mastery/shared/page.ts for why there is no default here. A region
      // on a page whose size cannot be established is flagged for review rather
      // than filled in against a guessed page shape — every box it produced
      // would be wrong by an unknown amount, which is worse than an admitted
      // gap (CLAUDE.md rule 4).
      const dims = pageDimensions((firstPage ?? {}) as any);
      if (!dims) {
        await sb
          .from("question_region")
          .update({
            extract_status: "done",
            confidence_tier: "unreadable",
            needs_review: true,
            confidence_signals: { unreadable_reason: "We could not work out the size of this page, so we cannot say where anything on it sits." },
            updated_at: new Date().toISOString(),
          })
          .eq("id", regionId);
        await advanceAndEnqueue(env, sb, runId);
        return { detail: { unreadable: "no page dimensions" } };
      }
      const { width, height } = dims;

      const label = field(parsed.question_label as any, spans, width, height);
      const question = field(parsed.question_text, spans, width, height);
      const answer = field(parsed.student_answer, spans, width, height);
      const awarded = field(parsed.marks_awarded, spans, width, height);
      const available = field(parsed.marks_available, spans, width, height);
      const remark = field(parsed.teacher_remark, spans, width, height);

      if (parsed.unreadable) {
        await sb
          .from("question_region")
          .update({
            extract_status: "done",
            confidence_tier: "unreadable",
            needs_review: true,
            confidence_signals: { unreadable_reason: parsed.unreadable_reason },
            updated_at: new Date().toISOString(),
          })
          .eq("id", regionId);
        await advanceAndEnqueue(env, sb, runId);
        return { detail: { unreadable: parsed.unreadable_reason } };
      }

      const { error: updErr } = await sb
        .from("question_region")
        .update({
          question_label: label.value ?? region.question_label,
          question_label_box: label.box ?? region.question_label_box,
          question_text: question.value,
          question_text_box: question.box,
          student_answer: answer.value,
          student_answer_box: answer.box,
          marks_awarded: awarded.value,
          marks_awarded_box: awarded.box,
          marks_available: available.value,
          marks_available_box: available.box,
          teacher_remark: remark.value,
          teacher_remark_box: remark.box,
          region_type: parsed.region_type,
          extract_status: "done",
          updated_at: new Date().toISOString(),
        })
        .eq("id", regionId);
      if (updErr) throw new Error("question_region update failed: " + updErr.message + " | " + JSON.stringify(updErr));

      if (awarded.value !== null && awarded.box) {
        const { error: tmErr } = await sb
          .from("teacher_mark")
          .update({ value: awarded.value })
          .eq("region_id", regionId)
          .eq("mark_class", "marginal_number")
          .is("value", null);
        if (tmErr) throw new Error("teacher_mark update failed: " + tmErr.message + " | " + JSON.stringify(tmErr));
      }

      await advanceAndEnqueue(env, sb, runId);
      return { detail: { awarded: awarded.value, available: available.value } };
    } catch (e) {
      console.error("mastery-content post-model error", regionId, String((e as any)?.stack ?? e));
      throw e;
    }
  },
  async ({ env, sb, msg }) => {
    const regionId = msg.region_id;
    const { data: region } = await sb.from("question_region").select("confidence_signals").eq("id", regionId).maybeSingle();
    await sb
      .from("question_region")
      .update({
        extract_status: "failed",
        confidence_tier: "unreadable",
        needs_review: true,
        confidence_signals: {
          ...(region?.confidence_signals ?? {}),
          unreadable_reason: "We could not finish reading this question. It has been flagged for review.",
        },
        updated_at: new Date().toISOString(),
      })
      .eq("id", regionId);
    await advanceAndEnqueue(env, sb, msg.run_id);
  }
);

export default { queue: handler } satisfies ExportedHandler<Env, ContentMessage>;
