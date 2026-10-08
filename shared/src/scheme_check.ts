// The scheme_check stage: one unmarked Cambridge paper, checked question by
// question against the mark scheme for that exact paper (owner decision,
// 6 Oct 2026).
//
// Two message kinds on the explain queue:
//   scheme_check   — one per paper, after review: confirm the printed code,
//                    read the scheme ONCE, split it, fan out one message per
//                    question;
//   scheme_check_q — one per question: a single model call, so each delivery
//                    stays well inside the queue handler's deadline.
// The scheme section travels in the question message only for the life of the
// message. Nothing here writes scheme text to a table.
//
// Writes only paper_check and region_check. It never touches marks_awarded,
// teacher_mark, region_explanation or anything analytics reads.

import type { SupabaseClient } from "@supabase/supabase-js";
import type { Env } from "./env.js";
import { callModel } from "./model-client.js";
import { imageRef } from "./r2.js";
import { mustData, mustMaybe, mustOk } from "./db.js";
import { isRetryable } from "./errors.js";
import { placementInput, placementVerdicts } from "./placement.js";
import {
  fetchCambridgeScheme, headerMatches, schemePreamble, schemeSections, sectionFor, type SchemeRef,
} from "./cambridge_scheme.js";
import {
  SYSTEM, SCHEMA, PROMPT_VERSION, instruction, validate,
  HEADER_SYSTEM, HEADER_SCHEMA, validateHeader, type CheckResult,
} from "./prompts/scheme_check.v1.js";

export interface SchemeCheckMessage {
  scheme_check: { run_id: string };
  _retries?: number;
}

export interface SchemeCheckQuestionMessage {
  scheme_check_q: {
    run_id: string;
    region_id: string;
    paper: string;
    /** The placed label ("3(c)"); absent on messages queued before it existed. */
    label?: string | null;
    section: string | null;
    conventions: string | null;
  };
  _retries?: number;
}

/** What triage stored on the run when it accepted an unmarked Cambridge paper. */
export interface SchemeCheckRouting {
  ref: SchemeRef;
  candidates: string[];
}

function sessionWords(ref: SchemeRef): string {
  return ref.series === "w" ? "October/November" : ref.series === "s" ? "May/June" : "February/March";
}

async function setStatus(sb: SupabaseClient, runId: string, patch: Record<string, unknown>) {
  await mustOk(sb.from("paper_check").update({ ...patch, updated_at: new Date().toISOString() }).eq("run_id", runId), "paper_check update");
}

const UNCHECKED = (reason: string): CheckResult => ({
  canCheck: false, reason, estimatedMarks: null, maxMarks: null, confidence: "unsure", whatWasRight: null, whatWasMissing: [], doThisNext: null,
});

/** The paper message: confirm, read the scheme once, fan out per question. */
export async function runSchemeCheck(opts: { env: Env; sb: SupabaseClient; runId: string; attempt?: number }) {
  const { env, sb, runId } = opts;
  const run = await mustMaybe(sb.from("extraction_run").select("id, paper_id, student_id, status, tier_routing").eq("id", runId).maybeSingle(), "run read") as any;
  const routing = run?.tier_routing?.scheme_check as SchemeCheckRouting | undefined;
  if (!run || !routing?.ref) return { skipped: "not a scheme-check run" };
  const check = await mustMaybe(sb.from("paper_check").select("status").eq("run_id", runId).maybeSingle(), "paper_check read") as any;
  if (!check) return { skipped: "no paper_check row" };
  // "running" is allowed through: a redelivery after a transient failure must
  // be able to finish the paper. Re-running is safe (all writes are upserts).
  if (check.status === "done" || check.status === "unavailable") return { skipped: `already ${check.status}` };
  await setStatus(sb, runId, { status: "running", reason: null });
  const ref = routing.ref;

  // 1. Confirm the printed reference on the page itself with the strong route.
  //    A misread variant (…/12 for …/11) is a real scheme for another paper.
  const pages = await mustData(sb.from("paper_page").select("page_number, r2_bucket, r2_key").eq("paper_id", run.paper_id).not("r2_key", "is", null).order("page_number").limit(2), "pages read") as any[];
  let confirmed = false;
  for (const page of pages) {
    const image = await imageRef(env, page.r2_bucket ?? "derived", page.r2_key, "high");
    const { parsed } = await callModel({
      env, sb, stage: "scheme_check", system: HEADER_SYSTEM, instruction: "Read the paper reference on this page.",
      images: [image], schema: HEADER_SCHEMA as any, validate: validateHeader,
      runId, paperId: run.paper_id, studentId: run.student_id, attempt: opts.attempt, thinkingLevel: "low",
    });
    if (headerMatches(parsed.reference, ref)) { confirmed = true; break; }
    if (parsed.reference) break; // legible and different: stop, do not try another page
  }
  if (!confirmed) {
    await setStatus(sb, runId, { status: "unavailable", reason: `We could not confirm this is ${ref.label} from the page itself, so it was not checked.` });
    return { unavailable: "header not confirmed" };
  }

  // 2. The scheme, once for the whole paper. Never stored by Axon.
  const scheme = await fetchCambridgeScheme(env, ref, routing.candidates?.length ? routing.candidates : undefined);
  if (!scheme) {
    await setStatus(sb, runId, { status: "unavailable", reason: `The mark scheme for ${ref.label} could not be found or read.` });
    return { unavailable: "scheme not found" };
  }
  const sections = schemeSections(scheme.markdown);
  const conventions = schemePreamble(scheme.markdown);

  // 3. One message per readable question.
  const allRegions = await mustData(sb.from("question_region")
    .select("id, order_index, question_label, page_spans, confidence_tier, marks_awarded, marks_available, student_answer, question_text")
    .eq("run_id", runId)
    .order("order_index"), "regions read") as any[];
  const paper = `${ref.code}/${ref.component} ${sessionWords(ref)} ${ref.year}`;
  const messages = schemeQuestionMessages(runId, allRegions, sections, paper, conventions);
  if (!messages.length) {
    await setStatus(sb, runId, { status: "done", checked: 0, reason: "No readable question was found on this paper." });
    return { checked: 0 };
  }
  const queue = env.SELF_QUEUE ?? env.EXPLAIN_QUEUE;
  if (!queue) throw new Error("scheme_check: no queue to fan out to");
  for (let i = 0; i < messages.length; i += 50) {
    await queue.sendBatch(messages.slice(i, i + 50).map((body) => ({ body })));
  }
  return { queued: messages.length, sections: sections.size, scheme: ref.filename, source_host: scheme.sourceHost };
}

/**
 * One message per readable region, each carrying its scheme section.
 *
 * The section is found by the PLACED label (council D1). Cambridge prints
 * "3(a)" once and then a bare "(c)" further down; a bare "(c)" has no question
 * number to find a section by, so every such part came back "not found in the
 * mark scheme". The walk is the one the review screen groups parts with, run
 * over every region of the run (unreadable ones included, so the reading order
 * is the paper's), then the unreadable ones are dropped.
 */
export function schemeQuestionMessages(
  runId: string,
  rows: Array<Parameters<typeof placementInput>[0] & { id: string; confidence_tier?: string | null; question_label?: string | null }>,
  sections: Map<number, string>,
  paper: string,
  conventions: string | null,
): SchemeCheckQuestionMessage[] {
  const placement = placementVerdicts(rows.map(placementInput));
  return rows
    .map((row, i) => ({ row, label: placement[i].placedLabel ?? row.question_label ?? null }))
    .filter(({ row }) => row.confidence_tier !== "unreadable")
    .map(({ row, label }) => ({
      scheme_check_q: { run_id: runId, region_id: row.id, paper, label, section: sectionFor(sections, label), conventions },
    }));
}

/** One question: check it, store the estimate, close the paper when it is the last. */
export async function runSchemeCheckQuestion(opts: { env: Env; sb: SupabaseClient; msg: SchemeCheckQuestionMessage["scheme_check_q"]; attempt?: number }) {
  const { env, sb, msg } = opts;
  const region = await mustMaybe(sb.from("question_region")
    .select("id, paper_id, student_id, question_label, question_text, student_answer, marks_available")
    .eq("id", msg.region_id).eq("run_id", msg.run_id).maybeSingle(), "region read") as any;
  if (!region) return { skipped: "no such question" };
  const marksAvailable = region.marks_available === null ? null : Number(region.marks_available);

  let result: CheckResult;
  let model: string | null = null;
  if (!msg.section) {
    result = UNCHECKED("This question was not found in the mark scheme.");
  } else {
    try {
      const out = await callModel({
        env, sb, stage: "scheme_check", system: SYSTEM,
        instruction: instruction({ paper: msg.paper, label: msg.label ?? region.question_label, marksAvailable, questionText: region.question_text, studentAnswer: region.student_answer, scheme: msg.section, conventions: msg.conventions }),
        schema: SCHEMA as any,
        validate: (v) => validate(v, { scheme: msg.section!, marksAvailable }),
        runId: msg.run_id, paperId: region.paper_id, regionId: region.id, studentId: region.student_id, attempt: opts.attempt,
      });
      result = out.parsed;
      model = out.model;
    } catch (error) {
      // A transient provider failure goes back to the queue; anything else is
      // an honest gap on this one question.
      if (isRetryable(error)) throw error;
      console.warn("scheme_check question failed", region.id, String((error as Error)?.message ?? error).slice(0, 200));
      result = UNCHECKED("This question could not be checked.");
    }
  }
  await writeRegionCheck(sb, msg.run_id, region.id, region.student_id, result, model);
  await closeIfComplete(sb, msg.run_id);
  return { can_check: result.canCheck, estimate: result.estimatedMarks };
}

async function writeRegionCheck(sb: SupabaseClient, runId: string, regionId: string, studentId: string, result: CheckResult, model: string | null) {
  await mustOk(sb.from("region_check").upsert({
    region_id: regionId, run_id: runId, student_id: studentId,
    can_check: result.canCheck, reason: result.reason,
    estimated_marks: result.estimatedMarks, max_marks: result.maxMarks, confidence: result.confidence,
    what_was_right: result.whatWasRight, what_was_missing: result.whatWasMissing, do_this_next: result.doThisNext,
    model_version: model, prompt_version: PROMPT_VERSION,
  }, { onConflict: "region_id" }), "region_check upsert");
}

/** The paper is done once every readable question has a row. Idempotent. */
async function closeIfComplete(sb: SupabaseClient, runId: string) {
  const expected = await mustData(sb.from("question_region").select("id").eq("run_id", runId).or("confidence_tier.is.null,confidence_tier.neq.unreadable"), "regions count") as any[];
  const rows = await mustData(sb.from("region_check").select("can_check").eq("run_id", runId), "region_check count") as any[];
  if (rows.length < expected.length) return;
  const checked = rows.filter((r) => r.can_check).length;
  await setStatus(sb, runId, { status: "done", checked, reason: checked ? null : "No question on this paper could be checked." });
}

export async function failSchemeCheck(sb: SupabaseClient, msg: SchemeCheckMessage, error: unknown) {
  console.error("scheme_check failed", msg.scheme_check.run_id, String((error as Error)?.message ?? error).slice(0, 300));
  await mustOk(sb.from("paper_check").update({ status: "failed", reason: "Checking this paper failed. Try again later.", updated_at: new Date().toISOString() }).eq("run_id", msg.scheme_check.run_id), "paper_check failed");
}

export async function failSchemeCheckQuestion(sb: SupabaseClient, msg: SchemeCheckQuestionMessage, error: unknown) {
  const m = msg.scheme_check_q;
  console.error("scheme_check question failed permanently", m.region_id, String((error as Error)?.message ?? error).slice(0, 300));
  const region = await mustMaybe(sb.from("question_region").select("student_id").eq("id", m.region_id).maybeSingle(), "region read") as any;
  if (!region) return;
  await writeRegionCheck(sb, m.run_id, m.region_id, region.student_id, UNCHECKED("This question could not be checked."), null);
  await closeIfComplete(sb, m.run_id);
}
