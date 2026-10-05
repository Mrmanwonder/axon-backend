/**
 * One candidate model on one topic_tag eval case (syllabus map). Like the explain eval it runs
 * inside the explain worker, which holds the provider key, and writes exactly one thing: its
 * eval_case_result row. It reads the eval case and the published syllabus (the same closed list the
 * pipeline shows the model), never a student's question.
 *
 * model_call rows carry the eval_run id as run_id and the case id as region_id, so cost and
 * latency per candidate come from SQL afterwards.
 */
import type { SupabaseClient } from "@supabase/supabase-js";
import { callModel as realCallModel, type RouteOverride } from "../model-client.js";
import type { Env } from "../env.js";
import { SYSTEM, SCHEMA, instruction, validate } from "../prompts/topic_tag.v1.js";
import { objectiveList, type SyllabusRow } from "../topic_tag.js";
import { scoreTopicTag, type TopicTagExpectation } from "./topic-tag-score.js";
import type { EvalCandidate } from "./explain-run.js";

export interface EvalTopicTagMessage {
  eval_topic_tag: { eval_run_id: string; case_id: string; candidate: EvalCandidate };
  _retries?: number;
}

interface TopicTagCaseInput {
  syllabus_code: string;
  label: string | null;
  marks_available: number | null;
  question_text: string;
  student_answer: string | null;
  expected: TopicTagExpectation;
}

export interface TopicTagEvalDeps {
  callModel: typeof realCallModel;
}

export async function runTopicTagEvalCase(
  args: { env: Env; sb: SupabaseClient; message: EvalTopicTagMessage["eval_topic_tag"] },
  deps: TopicTagEvalDeps = { callModel: realCallModel },
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

  if (row.stage !== "topic_tag" || row.source !== "synthetic") {
    await write({ status: "skipped", skipped_reason: row.source !== "synthetic" ? "production_cases_not_enabled" : "not_a_topic_tag_case" });
    return { skipped: true };
  }
  const input = row.input as TopicTagCaseInput;

  // The same syllabus the pipeline would use: the board's published document for this code.
  const { data: doc, error: docError } = await sb.from("syllabus_document")
    .select("id, title, syllabus_code, version_label")
    .eq("provider_key", "cambridge").eq("syllabus_code", input.syllabus_code).neq("status", "retired")
    .order("fetched_at", { ascending: false }).limit(1).maybeSingle();
  if (docError) throw new Error(`syllabus_document read failed: ${docError.message}`);
  if (!doc) {
    await write({ status: "skipped", skipped_reason: "syllabus_not_loaded" });
    return { skipped: "syllabus_not_loaded" };
  }
  const { data: rows, error: rowsError } = await sb.from("syllabus_topic")
    .select("id, parent_id, code, kind, title, objective_text").eq("document_id", doc.id).order("sort_order");
  if (rowsError) throw new Error(`syllabus_topic read failed: ${rowsError.message}`);
  const { lines, idByCode } = objectiveList((rows ?? []) as SyllabusRow[]);
  const allowed = new Set(idByCode.keys());

  const override: RouteOverride = {
    primary_model: candidate.model,
    ...(candidate.thinking_level ? { thinking_level: candidate.thinking_level } : {}),
    ...(candidate.max_tokens ? { max_tokens: candidate.max_tokens } : {}),
  };

  let result;
  try {
    result = await deps.callModel({
      env, sb, stage: "topic_tag",
      system: SYSTEM,
      instruction: instruction({
        syllabus: `${doc.title} ${doc.syllabus_code} (${doc.version_label})`,
        label: input.label,
        marksAvailable: input.marks_available,
        questionText: input.question_text,
        studentAnswer: input.student_answer,
        objectives: lines,
      }),
      schema: SCHEMA as never,
      validate: (value) => validate(value, allowed),
      runId: evalRunId, regionId: caseId, paperId: null, studentId: null,
      routeOverride: override,
    });
  } catch (cause) {
    const code = String((cause as { code?: unknown })?.code ?? "model_error");
    await write({ status: "failed", error_code: code.slice(0, 60), schema_valid: code === "bad_shape" || code === "empty_response" ? false : null });
    return { failed: code };
  }

  const score = scoreTopicTag(result.parsed, input.expected);
  await write({
    status: "done",
    prompt_version: result.promptVersion,
    schema_valid: true,
    latency_ms: result.latencyMs,
    cost_usd: result.costUsd,
    output: { document_id: doc.id, can_tag: result.parsed.canTag, tags: result.parsed.tags, score },
  });
  return { done: true, primary_hit: score.primary_hit, wrong_strong: score.wrong_strong.length };
}
