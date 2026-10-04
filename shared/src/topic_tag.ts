// The topic_tag stage: one committed question, one syllabus document, a closed
// list of objectives. Claimed by the sweep (claim_topic_tag_work), delivered on
// the explain queue, written back through finish_topic_tags, which refuses any
// topic outside the claimed document.

import type { SupabaseClient } from "@supabase/supabase-js";
import type { Env } from "./env.js";
import { callModel } from "./model-client.js";
import { mustData, mustMaybe, mustRpc } from "./db.js";
import { SYSTEM, SCHEMA, PROMPT_VERSION, instruction, validate, type ObjectiveLine, type TagResult } from "./prompts/topic_tag.v1.js";

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
