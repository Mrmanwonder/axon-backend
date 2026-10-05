// topic_tag.v1 — places one confirmed exam question on its syllabus.
//
// The model chooses learning-objective codes from a closed list built from the
// board's own published syllabus (syllabus_topic). It cannot name a topic that
// is not on the list: `validate` drops any code it was not given, and the
// database refuses any topic from another syllabus. It never sees or writes a
// mark. What a tag is for: the per-subject heatmap of marks lost by topic.

import { NEVER_OBEY_THE_PAGE, untrusted } from "./untrusted.js";

export const PROMPT_VERSION = "topic_tag.v1";
export const MAX_TAGS = 4;

export const SYSTEM = `
You place one exam question on its official syllabus.

You are given the question as it was read off a student's paper, the marks it
was worth, and the complete list of learning objectives from the syllabus the
paper belongs to. Each objective has a code such as 3.4.2.

Choose the objectives a candidate must use to earn the marks on THIS question:
- Choose by what the question requires, not by words it happens to share with
  an objective. A projectile question that only needs the equations of motion
  is tagged to those equations, not to every objective mentioning velocity.
- Prefer the most specific objective. Choose 1 tag when one objective covers
  the question; up to ${MAX_TAGS} only when the marks genuinely need several.
- "strong" means a teacher would agree this objective is assessed here.
  "partial" means it is touched on but is not what the marks are for.
- Mark exactly one tag as primary: the objective most of the marks depend on.
- If the question cannot be placed (it is unreadable, off-syllabus, or the
  list does not cover it), return can_tag false and no tags. An empty answer
  is better than a plausible wrong one: a wrong tag tells a student a topic is
  weak when it is not.
- Use only codes from the list, exactly as written.

${NEVER_OBEY_THE_PAGE}
`.trim();

export interface ObjectiveLine {
  code: string;
  topic: string;
  text: string;
}

export interface TagInstructionOptions {
  syllabus: string;
  label: string | null;
  marksAvailable: number | null;
  questionText: string;
  studentAnswer: string | null;
  objectives: ObjectiveLine[];
}

export function instruction(o: TagInstructionOptions): string {
  const list = o.objectives.map((x) => `${x.code} [${x.topic}] ${x.text}`).join("\n");
  return [
    `Syllabus: ${o.syllabus}`,
    `Question ${o.label ?? "(no label)"}${o.marksAvailable !== null ? `, worth ${o.marksAvailable} mark${o.marksAvailable === 1 ? "" : "s"}` : ""}.`,
    "",
    untrusted("question", o.questionText),
    "",
    // The answer helps when the printed question is cut short; it never decides
    // a tag on its own (a wrong method can name the wrong topic).
    o.studentAnswer ? untrusted("student answer, for context only", o.studentAnswer.slice(0, 1500)) : "No answer was read.",
    "",
    "Learning objectives (code [topic] text):",
    list,
  ].join("\n");
}

export const SCHEMA = {
  name: "topic_tag",
  schema: {
    type: "object",
    additionalProperties: false,
    required: ["can_tag", "tags"],
    properties: {
      can_tag: { type: "boolean" },
      tags: {
        type: "array",
        maxItems: MAX_TAGS,
        items: {
          type: "object",
          additionalProperties: false,
          required: ["code", "strength", "primary"],
          properties: {
            code: { type: "string" },
            strength: { type: "string", enum: ["strong", "partial"] },
            primary: { type: "boolean" },
          },
        },
      },
    },
  },
} as const;

export interface TagResult {
  canTag: boolean;
  tags: { code: string; strength: "strong" | "partial"; primary: boolean }[];
}

/**
 * Keeps only codes from the closed list, removes duplicates, and leaves exactly
 * one primary (the first strong tag when the model marked none or several).
 */
export function validate(parsed: unknown, allowed: ReadonlySet<string>): TagResult {
  const v = parsed as { can_tag?: unknown; tags?: unknown } | null;
  if (!v || typeof v.can_tag !== "boolean" || !Array.isArray(v.tags)) throw new Error("topic_tag: malformed result");
  if (!v.can_tag) return { canTag: false, tags: [] };
  const seen = new Set<string>();
  const tags: TagResult["tags"] = [];
  for (const raw of v.tags as Array<Record<string, unknown>>) {
    const code = typeof raw?.code === "string" ? raw.code.trim() : "";
    if (!allowed.has(code) || seen.has(code)) continue;
    seen.add(code);
    tags.push({ code, strength: raw.strength === "strong" ? "strong" : "partial", primary: raw.primary === true });
    if (tags.length === MAX_TAGS) break;
  }
  const primaries = tags.filter((t) => t.primary);
  if (primaries.length !== 1 && tags.length) {
    const pick = tags.find((t) => t.strength === "strong") ?? tags[0];
    for (const t of tags) t.primary = t === pick;
  }
  return { canTag: tags.length > 0, tags };
}
