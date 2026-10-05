// The scheme_check stage: one unmarked Cambridge paper, checked question by
// question against the mark scheme for that exact paper (owner decision,
// 6 Oct 2026). Delivered as one message on the explain queue after the
// student finishes review, so the scheme is fetched once per paper.
//
// Writes only paper_check and region_check. It never touches marks_awarded,
// teacher_mark, region_explanation or anything analytics reads.

import type { SupabaseClient } from "@supabase/supabase-js";
import type { Env } from "./env.js";
import { callModel } from "./model-client.js";
import { imageRef } from "./r2.js";
import { mustData, mustMaybe, mustOk } from "./db.js";
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

/** What triage stored on the run when it accepted an unmarked Cambridge paper. */
export interface SchemeCheckRouting {
  ref: SchemeRef;
  candidates: string[];
}

const CONCURRENCY = 4;

function sessionWords(ref: SchemeRef): string {
  return ref.series === "w" ? "October/November" : ref.series === "s" ? "May/June" : "February/March";
}

async function setStatus(sb: SupabaseClient, runId: string, patch: Record<string, unknown>) {
  await mustOk(sb.from("paper_check").update({ ...patch, updated_at: new Date().toISOString() }).eq("run_id", runId), "paper_check update");
}

async function mapLimit<T, R>(items: T[], limit: number, fn: (item: T) => Promise<R>): Promise<R[]> {
  const out: R[] = new Array(items.length);
  let next = 0;
  await Promise.all(Array.from({ length: Math.min(limit, items.length) }, async () => {
    while (next < items.length) {
      const i = next++;
      out[i] = await fn(items[i]!);
    }
  }));
  return out;
}

export async function runSchemeCheck(opts: { env: Env; sb: SupabaseClient; runId: string; attempt?: number }) {
  const { env, sb, runId } = opts;
  const run = await mustMaybe(sb.from("extraction_run").select("id, paper_id, student_id, status, tier_routing").eq("id", runId).maybeSingle(), "run read") as any;
  const routing = run?.tier_routing?.scheme_check as SchemeCheckRouting | undefined;
  if (!run || !routing?.ref) return { skipped: "not a scheme-check run" };
  const check = await mustMaybe(sb.from("paper_check").select("status").eq("run_id", runId).maybeSingle(), "paper_check read") as any;
  if (!check) return { skipped: "no paper_check row" };
  if (check.status === "done" || check.status === "unavailable") return { skipped: `already ${check.status}` };
  await setStatus(sb, runId, { status: "running", reason: null });
  const ref = routing.ref;

  // 1. Confirm the printed reference on the page itself with the strong route.
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
    if (parsed.reference) break; // a legible, different reference: stop, do not try another page
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

  // 3. Every readable question the student confirmed.
  const regions = await mustData(sb.from("question_region")
    .select("id, question_label, question_text, student_answer, marks_available, confidence_tier, order_index")
    .eq("run_id", runId)
    .neq("confidence_tier", "unreadable")
    .order("order_index"), "regions read") as any[];

  const paperName = `${ref.code}/${ref.component} ${sessionWords(ref)} ${ref.year}`;
  let checked = 0;
  await mapLimit(regions, CONCURRENCY, async (region) => {
    const section = sectionFor(sections, region.question_label);
    const marksAvailable = region.marks_available === null ? null : Number(region.marks_available);
    let result: CheckResult;
    let model: string | null = null;
    if (!section) {
      result = { canCheck: false, reason: "This question was not found in the mark scheme.", estimatedMarks: null, maxMarks: null, confidence: "unsure", whatWasRight: null, whatWasMissing: [], doThisNext: null };
    } else {
      try {
        const out = await callModel({
          env, sb, stage: "scheme_check", system: SYSTEM,
          instruction: instruction({ paper: paperName, label: region.question_label, marksAvailable, questionText: region.question_text, studentAnswer: region.student_answer, scheme: section, conventions }),
          schema: SCHEMA as any,
          validate: (v) => validate(v, { scheme: section, marksAvailable }),
          runId, paperId: run.paper_id, regionId: region.id, studentId: run.student_id, attempt: opts.attempt,
        });
        result = out.parsed;
        model = out.model;
      } catch (error) {
        console.warn("scheme_check question failed", region.id, String((error as Error)?.message ?? error).slice(0, 200));
        result = { canCheck: false, reason: "This question could not be checked.", estimatedMarks: null, maxMarks: null, confidence: "unsure", whatWasRight: null, whatWasMissing: [], doThisNext: null };
      }
    }
    await mustOk(sb.from("region_check").upsert({
      region_id: region.id, run_id: runId, student_id: run.student_id,
      can_check: result.canCheck, reason: result.reason,
      estimated_marks: result.estimatedMarks, max_marks: result.maxMarks, confidence: result.confidence,
      what_was_right: result.whatWasRight, what_was_missing: result.whatWasMissing, do_this_next: result.doThisNext,
      model_version: model, prompt_version: PROMPT_VERSION,
    }, { onConflict: "region_id" }), "region_check upsert");
    if (result.canCheck) checked++;
  });

  await setStatus(sb, runId, { status: "done", checked, reason: checked ? null : "No question on this paper could be checked." });
  return { checked, questions: regions.length, scheme: ref.filename, source_host: scheme.sourceHost };
}

export async function failSchemeCheck(sb: SupabaseClient, msg: SchemeCheckMessage, error: unknown) {
  console.error("scheme_check failed", msg.scheme_check.run_id, String((error as Error)?.message ?? error).slice(0, 300));
  await sb.from("paper_check").update({ status: "failed", reason: "Checking this paper failed. Try again later.", updated_at: new Date().toISOString() }).eq("run_id", msg.scheme_check.run_id);
}
