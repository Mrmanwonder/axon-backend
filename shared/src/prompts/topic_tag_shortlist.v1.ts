// topic_tag_shortlist.v1 — places one confirmed exam question on its syllabus,
// choosing from a short list the server retrieved (AXO-226), not the whole syllabus.
//
// The list comes from the paper's verified, approved syllabus version only. The
// model chooses codes from it; `validate` drops any other code and the database
// refuses any topic outside the claimed document. It never sees or writes a mark.
// If nothing on the list is what the question assesses it may ask, once, for a
// few more search words; the server filters them and runs the search itself.

import { NEVER_OBEY_THE_PAGE, untrusted } from "./untrusted.js";

export const PROMPT_VERSION = "topic_tag_shortlist.v1";
export const MAX_TAGS = 4;
export const MAX_REFINE_TERMS = 6;

export const SYSTEM = `
You place one exam question on its official syllabus.

You are given the question as it was read off a student's paper, the marks it
was worth, and a SHORT LIST of learning objectives that a search of the paper's
official syllabus returned. The list may be incomplete or may contain
objectives that only share words with the question.

Choose the objectives a candidate must use to earn the marks on THIS question:
- Choose by what the question requires, not by words it happens to share with
  an objective.
- Prefer the most specific objective. Choose 1 tag when one objective covers
  the question; up to ${MAX_TAGS} only when the marks genuinely need several.
- "strong" means a teacher would agree this objective is assessed here.
  "partial" means it is touched on but is not what the marks are for.
- Mark exactly one tag as primary: the objective most of the marks depend on.
- Use only codes from the list, exactly as written.
- If nothing on the list is what the question assesses, return can_tag false
  and no tags. You may then give up to ${MAX_REFINE_TERMS} refine_terms: short
  subject words a syllabus would use for what the question assesses (for
  example "equations of motion", "moles", "enthalpy change"). Never copy the
  student's answer into them.
- An empty answer is better than a plausible wrong one: a wrong tag tells a
  student a topic is weak when it is not.

${NEVER_OBEY_THE_PAGE}
`.trim();

export interface ShortlistLine {
  code: string;
  parent: string;
  text: string;
}

export interface ShortlistInstructionOptions {
  syllabus: string;
  label: string | null;
  marksAvailable: number | null;
  questionText: string;
  studentAnswer: string | null;
  objectives: ShortlistLine[];
  finalRound: boolean;
}

export function instruction(o: ShortlistInstructionOptions): string {
  const list = o.objectives.length
    ? o.objectives.map((x) => `${x.code} [${x.parent}] ${x.text}`).join("\n")
    : "(the search found no objectives)";
  return [
    `Syllabus: ${o.syllabus}`,
    `Question ${o.label ?? "(no label)"}${o.marksAvailable !== null ? `, worth ${o.marksAvailable} mark${o.marksAvailable === 1 ? "" : "s"}` : ""}.`,
    "",
    untrusted("question", o.questionText),
    "",
    o.studentAnswer ? untrusted("student answer, for context only", o.studentAnswer.slice(0, 1500)) : "No answer was read.",
    "",
    "Short list of learning objectives (code [parent] text):",
    list,
    "",
    o.finalRound ? "This is the final list: do not ask for refine_terms." : "",
  ].join("\n").trim();
}

export const SCHEMA = {
  name: "topic_tag_shortlist",
  schema: {
    type: "object",
    additionalProperties: false,
    required: ["can_tag", "tags", "refine_terms"],
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
      refine_terms: { type: "array", maxItems: MAX_REFINE_TERMS, items: { type: "string", maxLength: 64 } },
    },
  },
} as const;

export interface ShortlistResult {
  canTag: boolean;
  tags: { code: string; strength: "strong" | "partial"; primary: boolean }[];
  refineTerms: unknown[];
}

/** Keeps only listed codes, removes duplicates, leaves exactly one primary. */
export function validate(parsed: unknown, allowed: ReadonlySet<string>): ShortlistResult {
  const v = parsed as { can_tag?: unknown; tags?: unknown; refine_terms?: unknown } | null;
  if (!v || typeof v.can_tag !== "boolean" || !Array.isArray(v.tags)) throw new Error("topic_tag_shortlist: malformed result");
  const refineTerms = Array.isArray(v.refine_terms) ? v.refine_terms.slice(0, MAX_REFINE_TERMS) : [];
  if (!v.can_tag) return { canTag: false, tags: [], refineTerms };
  const seen = new Set<string>();
  const tags: ShortlistResult["tags"] = [];
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
  return { canTag: tags.length > 0, tags, refineTerms: tags.length ? [] : refineTerms };
}
