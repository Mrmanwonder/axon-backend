/**
 * One candidate model on one scheme_check eval case (AXO-202). Like the topic_tag eval it runs inside
 * the explain worker, which holds the provider key, and writes exactly one thing: its
 * eval_case_result row. The case carries its own invented question, answer and scheme section
 * (evals/golden/scheme-check-v1.json in Axon-Site); it never reads a student's paper or a real
 * mark scheme.
 *
 * The call goes through the same SYSTEM / instruction / SCHEMA / validate as the live stage, so the
 * validator's promises (estimate within 0..max, no scheme notation, no copied scheme lines) are
 * measured here: a rejected answer is a failed case, not a pass.
 *
 * model_call rows carry the eval_run id as run_id and the case id as region_id, so cost and latency
 * per candidate come from SQL afterwards.
 */
import type { SupabaseClient } from "@supabase/supabase-js";
import { callModel as realCallModel, type RouteOverride } from "../model-client.js";
import type { Env } from "../env.js";
import { SYSTEM, SCHEMA, instruction, validate } from "../prompts/scheme_check.v1.js";
import { scoreSchemeCheck, type SchemeCheckExpectation } from "./scheme-check-score.js";
import type { EvalCandidate } from "./explain-run.js";

export interface EvalSchemeCheckMessage {
  eval_scheme_check: { eval_run_id: string; case_id: string; candidate: EvalCandidate };
  _retries?: number;
}

interface SchemeCheckCaseInput {
  syllabus_code: string;
  paper: string;
  label: string | null;
  marks_available: number;
  question_text: string;
  student_answer: string | null;
  scheme: string;
  conventions: string | null;
  expected: SchemeCheckExpectation;
}

export interface SchemeCheckEvalDeps {
  callModel: typeof realCallModel;
}

export async function runSchemeCheckEvalCase(
  args: { env: Env; sb: SupabaseClient; message: EvalSchemeCheckMessage["eval_scheme_check"] },
  deps: SchemeCheckEvalDeps = { callModel: realCallModel },
): Promise<Record<string, unknown>> {
  const { env, sb, message } = args;
  const { eval_run_id: evalRunId, case_id: caseId, candidate } = message;

  const { data: row, error } = await sb.from("eval_case").select("id, stage, source, input").eq("id", caseId).maybeSingle();
  if (error) throw new Error(`eval_case read failed: ${error.message}`);
  if (!row) throw new Error(`no such eval case ${caseId}`);

  const base = { eval_run_id: evalRunId, case_id: caseId, candidate_key: candidate.key, model: candidate.model, thinking_level: candidate.thinking_level ?? null };
  const write = async (patch: Record<string, unknown>) => {
    const { error: writeError } = await sb.from("eval_case_result").upsert({ ...base, ...patch }, { onConflict: "eval_run_id,case_id,candidate_key" });
    if (writeError) throw new Error(`eval_case_result write failed: ${writeError.message}`);
  };

  if (row.stage !== "scheme_check" || row.source !== "synthetic") {
    await write({ status: "skipped", skipped_reason: row.source !== "synthetic" ? "production_cases_not_enabled" : "not_a_scheme_check_case" });
    return { skipped: true };
  }
  const input = row.input as SchemeCheckCaseInput;

  const override: RouteOverride = {
    primary_model: candidate.model,
    ...(candidate.thinking_level ? { thinking_level: candidate.thinking_level } : {}),
    ...(candidate.max_tokens ? { max_tokens: candidate.max_tokens } : {}),
  };

  let result;
  try {
    result = await deps.callModel({
      env, sb, stage: "scheme_check",
      system: SYSTEM,
      instruction: instruction({
        paper: input.paper,
        label: input.label,
        marksAvailable: input.marks_available,
        questionText: input.question_text,
        studentAnswer: input.student_answer,
        scheme: input.scheme,
        conventions: input.conventions,
      }),
      schema: SCHEMA as never,
      validate: (value) => validate(value, { scheme: input.scheme, marksAvailable: input.marks_available }),
      runId: evalRunId, regionId: caseId, paperId: null, studentId: null,
      routeOverride: override,
    });
  } catch (cause) {
    const code = String((cause as { code?: unknown })?.code ?? "model_error");
    const reason = String((cause as Error)?.message ?? "").slice(0, 120);
    await write({ status: "failed", error_code: code.slice(0, 60), schema_valid: code === "bad_shape" || code === "empty_response" ? false : null, judge_flags: reason ? [reason] : [] });
    return { failed: code };
  }

  const score = scoreSchemeCheck(result.parsed, input.expected);
  await write({
    status: "done",
    prompt_version: result.promptVersion,
    schema_valid: true,
    latency_ms: result.latencyMs,
    cost_usd: result.costUsd,
    // Estimates and the score only: no question, answer or scheme text in the result row.
    output: { can_check: result.parsed.canCheck, estimated_marks: result.parsed.estimatedMarks, max_marks: result.parsed.maxMarks, confidence: result.parsed.confidence, score },
  });
  return { done: true, in_range: score.in_range, can_check_ok: score.can_check_ok };
}
