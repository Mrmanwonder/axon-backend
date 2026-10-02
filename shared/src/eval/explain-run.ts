/**
 * One candidate model on one eval case (AXO-41). It runs inside a worker that already holds the
 * provider key, so no secret ever leaves Cloudflare, and it writes exactly one thing: its
 * eval_case_result row. It never reads or writes question_region, region_explanation,
 * mark_loss_event or any table a student sees.
 *
 * Calls are tagged for cost attribution without a schema change: model_call.run_id carries the
 * eval_run id and model_call.region_id carries the case id, so SQL can sum cost and latency per
 * candidate and per case afterwards. Web tools are off, so the comparison measures the model and
 * prompt alone (the pipeline's Tier 1 path adds Tavily context; that difference is recorded in the
 * run summary, not hidden).
 */
import type { SupabaseClient } from "@supabase/supabase-js";
import { callModel as realCallModel, type RouteOverride } from "../model-client.js";
import type { Env } from "../env.js";
import { gateModelAnswer } from "../grounding.js";
import { normalisePartKey } from "../question_parts.js";
import {
  SYSTEM as TIER1_SYSTEM,
  instruction as tier1Instruction,
  SCHEMA as TIER1_SCHEMA,
  validate as validateTier1,
} from "../prompts/explain_tier1.v2.js";
import * as judge from "./judge.v1.js";
import { checkExplain } from "./explain-score.js";

export interface EvalCandidate {
  key: string;
  model: string;
  thinking_level?: "minimal" | "low" | "medium" | "high";
  max_tokens?: number;
}

export interface EvalExplainMessage {
  eval: { eval_run_id: string; case_id: string; candidate: EvalCandidate };
  _retries?: number;
}

export const JUDGE_ROUTE: RouteOverride = {
  primary_model: "gemini-3.1-pro-preview",
  thinking_level: "high",
  max_tokens: 8192,
  prompt_version: judge.JUDGE_PROMPT_VERSION,
};

interface SyntheticInput {
  label: string;
  subject: string | null;
  class_level: number | string | null;
  marks_awarded: number;
  marks_available: number;
  question_text: string | null;
  student_answer: string | null;
  teacher_remark: string | null;
  mark_shapes: string[];
  prior_parts: Array<{ label: string; questionText?: string | null; studentAnswer?: string | null; marksAwarded?: number | null; marksAvailable?: number | null }>;
}

export interface EvalDeps {
  callModel: typeof realCallModel;
}

export async function runExplainEvalCase(
  args: { env: Env; sb: SupabaseClient; message: EvalExplainMessage["eval"] },
  deps: EvalDeps = { callModel: realCallModel },
): Promise<Record<string, unknown>> {
  const { env, sb, message } = args;
  const { eval_run_id: evalRunId, case_id: caseId, candidate } = message;

  const { data: row, error } = await sb.from("eval_case").select("id, source, input").eq("id", caseId).maybeSingle();
  if (error) throw new Error(`eval_case read failed: ${error.message}`);
  if (!row) throw new Error(`no such eval case ${caseId}`);

  const base = { eval_run_id: evalRunId, case_id: caseId, candidate_key: candidate.key, model: candidate.model, thinking_level: candidate.thinking_level ?? null };
  const write = async (patch: Record<string, unknown>) => {
    const { error: writeError } = await sb.from("eval_case_result").upsert({ ...base, ...patch }, { onConflict: "eval_run_id,case_id,candidate_key" });
    if (writeError) throw new Error(`eval_case_result write failed: ${writeError.message}`);
  };

  // Production cases read a student's stored text. Not enabled: it needs the owner's decision on
  // consent and on sending that text to the judge model (AXO-129). Say so rather than guess.
  if (row.source !== "synthetic") {
    await write({ status: "skipped", skipped_reason: "production_cases_not_enabled" });
    return { skipped: "production_cases_not_enabled" };
  }

  const input = row.input as SyntheticInput;
  const awarded = Number(input.marks_awarded);
  const available = Number(input.marks_available);

  // The pipeline never calls the model for a question with nothing lost; neither does the eval. It
  // is a control: explaining it would be a failure of the pipeline, not a result.
  if (!(awarded < available)) {
    await write({ status: "skipped", skipped_reason: "no_marks_lost" });
    return { skipped: "no_marks_lost" };
  }

  const priorParts = (input.prior_parts ?? []).map((p) => ({
    label: p.label,
    key: normalisePartKey(p.label),
    questionText: p.questionText ?? null,
    studentAnswer: p.studentAnswer ?? null,
    marksAwarded: p.marksAwarded ?? null,
    marksAvailable: p.marksAvailable ?? null,
  }));

  const instructionText = tier1Instruction({
    label: input.label,
    subject: input.subject,
    classLevel: input.class_level == null ? null : String(input.class_level),
    marksAwarded: awarded,
    marksAvailable: available,
    markShapes: input.mark_shapes ?? [],
    questionText: input.question_text,
    studentAnswer: input.student_answer,
    teacherRemark: input.teacher_remark,
    priorParts,
    unresolvedParts: [],
  });

  const override: RouteOverride = {
    primary_model: candidate.model,
    ...(candidate.thinking_level ? { thinking_level: candidate.thinking_level } : {}),
    ...(candidate.max_tokens ? { max_tokens: candidate.max_tokens } : {}),
  };

  let result;
  try {
    result = await deps.callModel({
      env, sb, stage: "explain",
      system: TIER1_SYSTEM, instruction: instructionText, schema: TIER1_SCHEMA,
      validate: (value) => validateTier1(value, null),
      runId: evalRunId, regionId: caseId, paperId: null, studentId: null,
      routeOverride: override,
    });
  } catch (cause) {
    const code = String((cause as { code?: unknown })?.code ?? "model_error");
    await write({ status: "failed", error_code: code.slice(0, 60), schema_valid: code === "bad_shape" || code === "empty_response" ? false : null });
    return { failed: code };
  }

  const parsed = result.parsed;
  const checks = checkExplain(parsed, { marksAwarded: awarded, marksAvailable: available });
  const grounding = gateModelAnswer({
    modelAnswer: parsed.model_answer,
    questionText: input.question_text,
    studentAnswer: input.student_answer,
    contextText: priorParts.flatMap((p) => [p.questionText, p.studentAnswer].filter(Boolean) as string[]),
    unresolvedDependencies: [],
    verifiedSchemeText: null,
  });

  // The judge sees the facts and the explanation, never the candidate's identity.
  let judgement: judge.Judgement | null = null;
  try {
    const judged = await deps.callModel({
      env, sb, stage: "explain",
      system: judge.SYSTEM,
      instruction: judge.instruction({
        question: input.question_text, studentAnswer: input.student_answer, teacherRemark: input.teacher_remark,
        marksAwarded: awarded, marksAvailable: available,
        priorParts: priorParts.map((p) => ({ label: p.label, questionText: p.questionText, studentAnswer: p.studentAnswer })),
        explanation: { can_explain: parsed.can_explain, cause: parsed.cause, body: parsed.body, do_this_next: parsed.do_this_next, model_answer: grounding.modelAnswer },
      }),
      schema: judge.SCHEMA, validate: judge.validate,
      runId: evalRunId, regionId: caseId, paperId: null, studentId: null,
      routeOverride: JUDGE_ROUTE,
    });
    judgement = judged.parsed;
  } catch { /* recorded below as a missing judgement, never a made-up score */ }

  const judgeContradiction = judgement?.flags.includes("contradicts_teacher_mark") ?? false;
  await write({
    status: "done",
    prompt_version: result.promptVersion,
    schema_valid: true,
    teacher_mark_contradiction: checks.teacher_mark_contradiction || judgeContradiction,
    loss_reasons_ok: checks.loss_reasons_ok,
    do_this_next_clears_floor: checks.do_this_next_clears_floor,
    grounding_status: grounding.status,
    can_explain: checks.can_explain,
    cause: checks.cause,
    faithfulness: judgement?.faithfulness ?? null,
    judge_flags: judgement ? [...judgement.flags, ...checks.contradiction_reasons.map((r) => `rule:${r}`)] : ["judge_failed", ...checks.contradiction_reasons.map((r) => `rule:${r}`)],
    latency_ms: result.latencyMs,
  });
  return { done: true, faithfulness: judgement?.faithfulness ?? null, contradiction: checks.teacher_mark_contradiction || judgeContradiction };
}
