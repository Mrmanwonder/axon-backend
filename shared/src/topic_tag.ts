// The topic_tag stage: one committed question, one syllabus document, a closed
// list of objectives. Claimed by the sweep (claim_topic_tag_work), delivered on
// the explain queue, written back through finish_topic_tags, which refuses any
// topic outside the claimed document.

import type { SupabaseClient } from "@supabase/supabase-js";
import type { Env } from "./env.js";
import { callModel } from "./model-client.js";
import { mustData, mustMaybe, mustRpc } from "./db.js";
import { SYSTEM, SCHEMA, PROMPT_VERSION, instruction, validate, type ObjectiveLine, type TagResult } from "./prompts/topic_tag.v1.js";
import * as Shortlist from "./prompts/topic_tag_shortlist.v1.js";
import { questionTerms, refinementTerms, resolveScope, searchAny, type Candidate } from "./syllabus_search.js";

/** Most objectives ever shown to the model on the scoped path, across both searches. */
export const MAX_SHORTLIST = 20;

export interface TopicTagMessage {
  topic_tag: { region_id: string; document_id: string };
  _retries?: number;
}

export interface SyllabusRow {
  id: string;
  parent_id: string | null;
  code: string;
  kind: "unit" | "topic" | "objective";
  title: string;
  objective_text: string | null;
}

/** The closed list shown to the model, and the code → id map used to store tags. */
export function objectiveList(rows: SyllabusRow[]): { lines: ObjectiveLine[]; idByCode: Map<string, string> } {
  const topics = new Map(rows.filter((r) => r.kind === "topic").map((r) => [r.id, r]));
  const lines: ObjectiveLine[] = [];
  const idByCode = new Map<string, string>();
  for (const r of rows) {
    if (r.kind !== "objective") continue;
    const topic = r.parent_id ? topics.get(r.parent_id) : undefined;
    lines.push({ code: r.code, topic: topic ? `${topic.code} ${topic.title}` : "", text: (r.objective_text ?? r.title).replace(/\s+/g, " ").trim() });
    idByCode.set(r.code, r.id);
  }
  return { lines, idByCode };
}

/** Tags in the shape finish_topic_tags takes. Only "strong" tags count in analytics. */
export function tagRows(result: TagResult, idByCode: Map<string, string>) {
  return result.tags
    .filter((t) => idByCode.has(t.code))
    .map((t) => ({ topic_id: idByCode.get(t.code)!, confidence: t.strength === "strong" ? "likely" : "unsure", is_primary: t.primary }));
}

export async function tagRegion(opts: { env: Env; sb: SupabaseClient; regionId: string; documentId: string; attempt?: number }) {
  const { env, sb, regionId, documentId } = opts;
  const region = await mustMaybe(
    sb.from("question_region")
      .select("id, paper_id, student_id, question_label, question_text, student_answer, marks_available")
      .eq("id", regionId)
      .maybeSingle(),
    "question_region read",
  ) as any;
  if (!region || !region.question_text?.trim()) {
    await mustRpc(sb.rpc("finish_topic_tags", { p_region_id: regionId, p_document_id: documentId, p_tags: [], p_model: null, p_prompt: PROMPT_VERSION }), "finish_topic_tags(empty)");
    return { skipped: "no question text" };
  }
  if (env.SYLLABUS_SEARCH === "1") return tagRegionScoped(opts, region);
  const doc = await mustMaybe(
    sb.from("syllabus_document").select("title, syllabus_code, version_label").eq("id", documentId).maybeSingle(),
    "syllabus_document read",
  ) as any;
  const rows = await mustData(
    sb.from("syllabus_topic").select("id, parent_id, code, kind, title, objective_text").eq("document_id", documentId).order("sort_order"),
    "syllabus_topic read",
  ) as SyllabusRow[];
  const { lines, idByCode } = objectiveList(rows);
  if (!doc || !lines.length) throw new Error("topic_tag: syllabus has no objectives");

  const allowed = new Set(idByCode.keys());
  const { parsed, model } = await callModel({
    env, sb,
    stage: "topic_tag",
    system: SYSTEM,
    instruction: instruction({
      syllabus: `${doc.title} ${doc.syllabus_code} (${doc.version_label})`,
      label: region.question_label,
      marksAvailable: region.marks_available === null ? null : Number(region.marks_available),
      questionText: region.question_text,
      studentAnswer: region.student_answer,
      objectives: lines,
    }),
    schema: SCHEMA as any,
    validate: (v) => validate(v, allowed),
    paperId: region.paper_id,
    regionId,
    studentId: region.student_id,
    attempt: opts.attempt,
    // Standard tier: flex can queue for up to 15 minutes, past the queue
    // handler's deadline (HANDLE_TIMEOUT_MS).
  });

  const written = await mustRpc(sb.rpc("finish_topic_tags", {
    p_region_id: regionId, p_document_id: documentId, p_tags: tagRows(parsed, idByCode), p_model: model, p_prompt: PROMPT_VERSION,
  }), "finish_topic_tags");
  return { tags: parsed.tags.length, written };
}

/** The objectives as the shortlist prompt shows them. */
export function shortlistLines(candidates: Candidate[]): Shortlist.ShortlistLine[] {
  return candidates.map((c) => ({
    code: c.code,
    parent: c.parent ? `${c.parent.code} ${c.parent.title}` : "",
    text: `${c.title}: ${c.objective_text}`.replace(/\s+/g, " ").trim() + (c.text_truncated ? " (truncated)" : ""),
  }));
}

/**
 * AXO-226: tag against the paper's verified, approved syllabus by search, not by
 * sending the whole syllabus. One server-side search on the printed question,
 * one judgment on the shortlist, and at most one model-suggested refinement
 * search and second judgment. No approved syllabus: the job ends `unavailable`
 * with a machine reason and no tag stands on it.
 */
async function tagRegionScoped(
  opts: { env: Env; sb: SupabaseClient; regionId: string; documentId: string; attempt?: number },
  region: { paper_id: string; student_id: string; question_label: string | null; question_text: string; student_answer: string | null; marks_available: unknown },
) {
  const { env, sb, regionId, documentId } = opts;
  const unavailable = async (reason: string) => {
    await mustRpc(sb.rpc("finish_topic_tags_unavailable", { p_region_id: regionId, p_document_id: documentId, p_reason: reason }), "finish_topic_tags_unavailable");
    return { unavailable: reason };
  };

  const scope = await resolveScope(sb, region.paper_id);
  if (scope.status !== "available") return unavailable(scope.reason);
  // The claim picked a document by the older, permissive rule; only the approved one may be used.
  if (scope.document_id !== documentId) return unavailable("scope_mismatch");

  const doc = await mustMaybe(
    sb.from("syllabus_document").select("title, syllabus_code, version_label").eq("id", documentId).maybeSingle(),
    "syllabus_document read",
  ) as any;
  if (!doc) return unavailable("missing");

  const first = await searchAny(sb, region.paper_id, scope, questionTerms(region.question_text));
  if (first.status !== "available") return unavailable(first.reason);
  const byCode = new Map<string, Candidate>();
  for (const c of first.candidates) if (byCode.size < MAX_SHORTLIST && !byCode.has(c.code)) byCode.set(c.code, c);

  const judge = (finalRound: boolean) => callModel({
    env, sb,
    stage: "topic_tag",
    system: Shortlist.SYSTEM,
    instruction: Shortlist.instruction({
      syllabus: `${doc.title} ${doc.syllabus_code} (${doc.version_label})`,
      label: region.question_label,
      marksAvailable: region.marks_available === null ? null : Number(region.marks_available),
      questionText: region.question_text,
      studentAnswer: region.student_answer,
      objectives: shortlistLines([...byCode.values()]),
      finalRound,
    }),
    schema: Shortlist.SCHEMA as any,
    validate: (v) => Shortlist.validate(v, new Set(byCode.keys())),
    // model_call records the prompt that was actually used.
    routeOverride: { prompt_version: Shortlist.PROMPT_VERSION },
    paperId: region.paper_id,
    regionId,
    studentId: region.student_id,
    attempt: opts.attempt,
  });

  let { parsed, model } = await judge(false);
  let refined = false;
  if (!parsed.canTag) {
    const terms = refinementTerms(parsed.refineTerms);
    if (terms.length) {
      const more = await searchAny(sb, region.paper_id, scope, terms);
      if (more.status !== "available") return unavailable(more.reason);
      let added = 0;
      for (const c of more.candidates) {
        if (byCode.size >= MAX_SHORTLIST || byCode.has(c.code)) continue;
        byCode.set(c.code, c);
        added++;
      }
      if (added) {
        refined = true;
        ({ parsed, model } = await judge(true));
      }
    }
  }

  // Ids come from the database's own shortlist, never from the model.
  const idByCode = new Map([...byCode].map(([code, c]) => [code, c.objective_id]));
  const written = await mustRpc(sb.rpc("finish_topic_tags", {
    p_region_id: regionId, p_document_id: documentId, p_tags: tagRows(parsed, idByCode), p_model: model, p_prompt: Shortlist.PROMPT_VERSION,
  }), "finish_topic_tags");
  return { tags: parsed.tags.length, written, shortlist: byCode.size, refined };
}

export async function failTopicTag(sb: SupabaseClient, msg: TopicTagMessage, error: unknown) {
  await mustRpc(sb.rpc("finish_topic_tags", {
    p_region_id: msg.topic_tag.region_id, p_document_id: msg.topic_tag.document_id, p_tags: [], p_model: null,
    p_prompt: PROMPT_VERSION, p_error: String((error as Error)?.message ?? error).slice(0, 500),
  }), "finish_topic_tags(failed)");
}

/**
 * One sweep tick: claim due questions and put them on the explain queue.
 * A failed send leaves the claim "queued"; claim_topic_tag_work hands it out
 * again after an hour, so nothing is lost and nothing is sent twice at once.
 */
export async function queueTopicTagWork(deps: {
  claim: (limit: number) => Promise<Array<{ region_id: string; document_id: string }>>;
  send: (messages: TopicTagMessage[]) => Promise<void>;
}, limit = 25): Promise<number> {
  const due = await deps.claim(limit);
  if (!due.length) return 0;
  await deps.send(due.map((d) => ({ topic_tag: { region_id: d.region_id, document_id: d.document_id } })));
  return due.length;
}
