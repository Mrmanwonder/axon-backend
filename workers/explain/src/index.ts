import { callModel } from "@mastery/shared/openrouter.js";
import { consumeQueue } from "@mastery/shared/worker.js";
import { mustOk, mustOne, mustMaybe, mustData, mustAffectRows, mustRpc } from "@mastery/shared/db.js";
import { clearsTheFloor } from "@mastery/shared/quality_floor.js";
import { SYSTEM, instruction, SCHEMA, validate } from "@mastery/shared/prompts/explain_tier1.v1.js";
import { normalisePartKey, resolveDependencies } from "@mastery/shared/question_parts.js";
import { gateModelAnswer } from "@mastery/shared/grounding.js";
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
    // maybeSingle + checked: a failed read used to be indistinguishable from
    // "no such question", and skipping a question because the database blinked
    // retires the message for a region that still needs explaining.
    const region = await mustMaybe(
      sb.from("question_region")
        .select("id, paper_id, student_id, question_label, question_text, student_answer, teacher_remark, marks_awarded, marks_available, explain_status, student_confirmed_at")
        .eq("id", regionId)
        .maybeSingle(),
      "question_region read",
    ) as any;
    if (!region) return { detail: { skipped: "no such question" } };
    if (region.explain_status === "done") return { detail: { skipped: "already explained" } };

    // The gate that makes explanation-before-review always fail: see
    // AXON_FIX_BRIEF.md §4.A1/A2. This worker will not explain a question the
    // student has not confirmed, however early or often it is queued.
    if (!region.student_confirmed_at) {
      await mustOk(sb.from("question_region").update({ explain_status: "pending" }).eq("id", regionId), "explain_status=pending");
      return { detail: { skipped: "not confirmed" } };
    }

    const awarded = Number(region.marks_awarded);
    const available = Number(region.marks_available);
    if (!Number.isFinite(awarded) || !Number.isFinite(available) || awarded >= available) {
      await mustOk(sb.from("question_region").update({ explain_status: "skipped" }).eq("id", regionId), "explain_status=skipped");
      await mustRpc(sb.rpc("advance_after_explain", { p_run_id: runId }), "advance_after_explain");
      return { detail: { skipped: "no marks lost" } };
    }

    // An atomic claim, not a status write. Cloudflare Queues are at-least-once,
    // so two deliveries can both read a region that is not done and both go on
    // to call the model — two paid calls, two explanations, one of them
    // overwriting the other. Narrowing the UPDATE by the status we expect means
    // exactly one delivery matches a row; the loser matches nothing and
    // mustAffectRows turns that into a permanent skip rather than a second call.
    try {
      await mustAffectRows(
        sb.from("question_region")
          .update({ explain_status: "running" })
          .eq("id", regionId)
          .in("explain_status", ["pending", "queued", "failed"])
          .select("id"),
        "claim region for explanation",
      );
    } catch {
      return { detail: { skipped: "already claimed by another delivery" } };
    }
    await beat();

    const run = await mustMaybe(sb.from("extraction_run").select("route_override").eq("id", runId).maybeSingle(), "extraction_run read") as any;
    const override = run?.route_override;
    const paper = await mustOne(sb.from("paper").select("subject, tier").eq("id", region.paper_id).maybeSingle(), "paper read") as any;
    const student = await mustOne(sb.from("student").select("class_level").eq("id", region.student_id).maybeSingle(), "student read") as any;

    if (paper?.tier === "tier_2") {
      const matched = await mustMaybe(sb.from("question_region").select("canonical_question_id").eq("id", regionId).maybeSingle(), "canonical match read") as any;
      if (!matched?.canonical_question_id) {
        console.info("tier 2 question with no scheme match; explaining as tier 1", regionId);
      }
    }

    const marks = await mustData(sb.from("teacher_mark").select("mark_class, comment_text").eq("region_id", regionId), "teacher_mark read") as any[];

    // The parts this question depends on. A Cambridge part routinely refers
    // back — "Justify your answer given in part (d)(i)" — and until this fetch
    // existed the model was handed that sentence with no (d)(i) anywhere in the
    // prompt. It was in the database the whole time: same run, same table, the
    // adjacent order_index. shared/src/question_parts.ts records what that
    // produced on a real paper.
    const siblingRows = await mustData(sb
      .from("question_region")
      .select("id, question_label, question_text, student_answer, marks_awarded, marks_available, order_index")
      .eq("run_id", runId)
      .order("order_index"), "sibling parts read") as any[];

    const ownOrderIndex =
      (siblingRows ?? []).find((s: any) => s.id === regionId)?.order_index ?? 0;

    const siblings = (siblingRows ?? [])
      .filter((s: any) => s.id !== regionId)
      .map((s: any) => ({
        label: s.question_label ?? "",
        key: normalisePartKey(s.question_label),
        questionText: s.question_text,
        studentAnswer: s.student_answer,
        marksAwarded: s.marks_awarded === null ? null : Number(s.marks_awarded),
        marksAvailable: s.marks_available === null ? null : Number(s.marks_available),
        orderIndex: s.order_index as number,
      }));

    const deps = resolveDependencies(region.question_text, ownOrderIndex, siblings as any);
    if (deps.unresolved.length) {
      console.info("question refers to parts not found in this run", regionId, deps.unresolved.join(","));
    }

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
        priorParts: deps.resolved,
        unresolvedParts: deps.unresolved,
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

    // The gate in front of the corrected working. It is the highest-trust thing
    // on the card — the sentence a student copies into their notes — so it gets
    // a bar the rest of the card does not, and a withheld one is stored as
    // withheld rather than dropped. An empty slot is honest; the alternative
    // shipped a paragraph about signal-to-noise ratio under a floating-point
    // question and called it "the corrected working".
    const grounding = gateModelAnswer({
      modelAnswer: parsed.model_answer,
      questionText: region.question_text,
      studentAnswer: region.student_answer,
      contextText: deps.resolved.flatMap((p) => [p.questionText, p.studentAnswer].filter(Boolean) as string[]),
      unresolvedDependencies: deps.unresolved,
    });
    if (grounding.status !== "complete") {
      console.info("model_answer withheld", regionId, grounding.status);
    }

    // The model said it could not explain this from what it was given. Honour
    // that instead of storing whatever prose came with the refusal: the region
    // is marked skipped, the run advances, and the card renders nothing. An
    // empty slot is honest; an explanation the model itself disowned is not.
    if (!parsed.can_explain) {
      console.info("model declined to explain", regionId);
      await mustOk(sb.from("question_region").update({ explain_status: "skipped" }).eq("id", regionId), "explain_status=skipped (can_explain false)");
      await mustRpc(sb.rpc("advance_after_explain", { p_run_id: runId }), "advance_after_explain");
      return { detail: { skipped: "model could not explain from the evidence given" } };
    }

    // Checked, and BEFORE the status moves. This insert used to be unchecked
    // and the status set regardless, so a database failure here produced a
    // region reading explain_status = 'done' with no explanation row behind it
    // — a question the student is told has been explained, showing nothing,
    // with no queue message left to try again.
    //
    // Upserted on region_id so a redelivery that gets past the claim — a lease
    // that expired mid-model-call, say — replaces its own row rather than
    // failing on the unique constraint. region_id alone is the real constraint
    // in the database (checked, not assumed: region_explanation has UNIQUE
    // (region_id), not the (run_id, region_id) pair a rescan might suggest), so
    // one question has exactly one explanation and a rescan replaces it.
    await mustOk(sb.from("region_explanation").upsert({
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
      model_answer: grounding.modelAnswer,
      grounding_status: grounding.status,
      model_answer_source: grounding.source,
      // What the explanation was actually built from, so a card can be traced
      // back to its grounding rather than taken on trust.
      depends_on_parts: deps.resolved.map((p) => p.label),
      unresolved_parts: deps.unresolved,
      loss_reasons: lossReasons,
      model_version: model,
      prompt_version: promptVersion,
    }, { onConflict: "region_id" }), "region_explanation upsert");

    // Only now. `done` is a claim that the explanation exists, and it is only
    // true once the line above has returned without an error.
    await mustOk(sb.from("question_region").update({ explain_status: "done" }).eq("id", regionId), "explain_status=done");
    await mustRpc(sb.rpc("advance_after_explain", { p_run_id: runId }), "advance_after_explain");

    return { detail: { cause: parsed.cause, floor_cleared: !!doThisNext, prior_parts: deps.resolved.length, grounding: grounding.status } };
  },
  // Checked, and therefore able to throw. The harness treats a throw here as
  // "the terminal state was not recorded" and retries rather than acknowledging
  // — which is the whole point: a failure to write `failed` used to be
  // swallowed, and the message acknowledged anyway, stranding the region.
  async ({ sb, msg }) => {
    await mustOk(sb.from("question_region").update({ explain_status: "failed" }).eq("id", msg.region_id), "explain_status=failed");
    await mustRpc(sb.rpc("advance_after_explain", { p_run_id: msg.run_id }), "advance_after_explain");
  }
);

export default { queue: handler } satisfies ExportedHandler<Env, ExplainMessage>;
