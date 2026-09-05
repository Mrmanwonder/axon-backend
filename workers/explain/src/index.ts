import { callModel } from "@mastery/shared/openrouter.js";
import { consumeQueue } from "@mastery/shared/worker.js";
import { clearsTheFloor } from "@mastery/shared/quality_floor.js";
import { SYSTEM, instruction, SCHEMA, validate } from "@mastery/shared/prompts/explain_tier1.v1.js";
import type { Env } from "@mastery/shared/env.js";

interface ExplainMessage {
  run_id: string;
  region_id: string;
  _retries?: number;
}

// This worker only ever explains as Tier 1 (see shared/prompts/explain_tier1.v1.ts) —
// there is no Tier 2 prompt grounded in canonical_question.marking_scheme yet.
// A Tier 2 question with no scheme match logs and falls through to Tier 1
// rather than fabricating a scheme (CLAUDE.md rule 2).
const handler = consumeQueue<ExplainMessage>(
  async ({ env, sb, msg, attempt, beat }) => {
    const runId = msg.run_id;
    const regionId = msg.region_id;
    const { data: region } = await sb
      .from("question_region")
      .select("id, paper_id, student_id, question_label, question_text, student_answer, teacher_remark, marks_awarded, marks_available, explain_status, student_confirmed_at")
      .eq("id", regionId)
      .single();
    if (!region) return { detail: { skipped: "no such question" } };
    if (region.explain_status === "done") return { detail: { skipped: "already explained" } };

    // The gate that makes explanation-before-review always fail: see
    // AXON_FIX_BRIEF.md §4.A1/A2. This worker will not explain a question the
    // student has not confirmed, however early or often it is queued.
    if (!region.student_confirmed_at) {
      await sb.from("question_region").update({ explain_status: "pending" }).eq("id", regionId);
      return { detail: { skipped: "not confirmed" } };
    }

    const awarded = Number(region.marks_awarded);
    const available = Number(region.marks_available);
    if (!Number.isFinite(awarded) || !Number.isFinite(available) || awarded >= available) {
      await sb.from("question_region").update({ explain_status: "skipped" }).eq("id", regionId);
      await sb.rpc("advance_after_explain", { p_run_id: runId });
      return { detail: { skipped: "no marks lost" } };
    }

    await sb.from("question_region").update({ explain_status: "running" }).eq("id", regionId);
    await beat();

    const { data: run } = await sb.from("extraction_run").select("route_override").eq("id", runId).maybeSingle();
    const override = run?.route_override;
    const { data: paper } = await sb.from("paper").select("subject, tier").eq("id", region.paper_id).single();
    const { data: student } = await sb.from("student").select("class_level").eq("id", region.student_id).single();

    if (paper?.tier === "tier_2") {
      const { data: matched } = await sb.from("question_region").select("canonical_question_id").eq("id", regionId).single();
      if (!matched?.canonical_question_id) {
        console.info("tier 2 question with no scheme match; explaining as tier 1", regionId);
      }
    }

    const { data: marks } = await sb.from("teacher_mark").select("mark_class, comment_text").eq("region_id", regionId);

    const { parsed, model, promptVersion } = await callModel({
      env,
      sb,
      stage: "explain",
      system: SYSTEM,
      instruction: instruction({
        label: region.question_label,
        subject: paper?.subject ?? null,
        classLevel: student?.class_level ?? null,
        marksAwarded: awarded,
        marksAvailable: available,
        questionText: region.question_text,
        studentAnswer: region.student_answer,
        teacherRemark: region.teacher_remark,
        markShapes: (marks ?? []).map((m: any) => m.mark_class).filter((c: string) => c !== "unknown"),
      }),
      schema: SCHEMA,
      validate,
      runId,
      paperId: region.paper_id,
      regionId,
      studentId: region.student_id,
      attempt,
      routeOverride: override,
    });

    const marksLost = Math.round((available - awarded) * 100) / 100;
    const doThisNext = clearsTheFloor(parsed.do_this_next) ? parsed.do_this_next : null;

    // A decomposition that accounts for more marks than the teacher took is not
    // a decomposition, it is a second opinion on the mark — hard rule 1. The
    // marks_awarded number is the fact here; anything that would contradict it
    // is dropped and the flat cause carries the question on its own.
    const decomposed = parsed.loss_reasons.reduce((sum, r) => sum + r.marks, 0);
    const lossReasons = decomposed > marksLost ? [] : parsed.loss_reasons;
    if (decomposed > marksLost) {
      console.info("loss_reasons exceeded the marks actually lost; keeping the flat cause", regionId, decomposed, marksLost);
    }

    await sb.from("region_explanation").insert({
      region_id: regionId,
      run_id: runId,
      student_id: region.student_id,
      tier: "tier_1",
      cause: parsed.cause,
      marks_lost: parsed.cause ? marksLost : null,
      body: parsed.body,
      do_this_next: doThisNext,
      concepts: parsed.concepts,
      command_word: parsed.command_word,
      command_word_note: parsed.command_word_note,
      model_answer: parsed.model_answer,
      loss_reasons: lossReasons,
      model_version: model,
      prompt_version: promptVersion,
    });

    await sb.from("question_region").update({ explain_status: "done" }).eq("id", regionId);
    await sb.rpc("advance_after_explain", { p_run_id: runId });

    return { detail: { cause: parsed.cause, floor_cleared: !!doThisNext } };
  },
  async ({ sb, msg }) => {
    await sb.from("question_region").update({ explain_status: "failed" }).eq("id", msg.region_id);
    await sb.rpc("advance_after_explain", { p_run_id: msg.run_id });
  }
);

export default { queue: handler } satisfies ExportedHandler<Env, ExplainMessage>;
