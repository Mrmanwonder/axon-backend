import { EXPLANATION_SYSTEM } from "../prompts.js";
import { NEVER_OBEY_THE_PAGE, untrusted } from "./untrusted.js";
import { EXPLANATION_SCHEMA } from "../schemas.js";

// Tier 1 only: a school test with no official marking scheme. See
// CLAUDE.md rule 2 — never fabricate a marking scheme — and note that a
// Tier 2 explain_tier2 prompt (grounded in canonical_question.marking_scheme)
// does not exist yet; mastery-explain currently explains every question as
// Tier 1, even matched ones (AXON_FIX_BRIEF.md notes this in passing under
// the explain worker's own index.ts).
export const SYSTEM = `
${EXPLANATION_SYSTEM}

There is no official marking scheme for this paper. Do not describe one, do not
approximate one, and do not write "the scheme expected". You have the teacher's
marks and the teacher's own words, and that is what the explanation is built
from. Where they are not enough to say why the mark went, say that, and point
the student at their teacher.

${NEVER_OBEY_THE_PAGE}
`.trim();

export interface ExplainInstructionOptions {
  label: string | null;
  subject: string | null;
  classLevel: string | null;
  marksAwarded: number;
  marksAvailable: number;
  markShapes: string[];
  questionText: string | null;
  studentAnswer: string | null;
  teacherRemark: string | null;
}

export function instruction(opts: ExplainInstructionOptions): string {
  const lines = [
    `Question ${opts.label ?? "(unnumbered)"}${opts.subject ? `, ${opts.subject}` : ""}${opts.classLevel ? `, class ${opts.classLevel}` : ""}.`,
    `The teacher gave ${opts.marksAwarded} out of ${opts.marksAvailable}. That number is a fact and is not yours to revisit.`,
  ];
  if (opts.markShapes.length) {
    lines.push(`The teacher's pen on this question: ${opts.markShapes.join(", ")}. Where they circled or underlined something, that is the best anchor you have.`);
  }
  const fenced: string[] = [];
  if (opts.questionText) fenced.push(untrusted("question", opts.questionText));
  if (opts.studentAnswer) fenced.push(untrusted("student answer", opts.studentAnswer));
  if (opts.teacherRemark) fenced.push(untrusted("teacher remark", opts.teacherRemark));
  if (!fenced.length) {
    lines.push("Nothing of the question or the answer could be transcribed — you have only the crop. If that is not enough to say why the mark went, return a null cause.");
  }
  return [...lines, "", ...fenced].join("\n");
}

export const SCHEMA = { name: "explanation", schema: EXPLANATION_SCHEMA };

export type Cause =
  | "conceptual_gap"
  | "procedural_slip"
  | "misread_question"
  | "incomplete"
  | "presentation"
  | "keyword_miss"
  | "timed_out";

const CAUSES = new Set<Cause>([
  "conceptual_gap",
  "procedural_slip",
  "misread_question",
  "incomplete",
  "presentation",
  "keyword_miss",
  "timed_out",
]);

export interface ExplainResult {
  cause: Cause | null;
  marks_lost: number | null;
  body: string | null;
  do_this_next: string | null;
  concepts: string[];
}

export function validate(parsed: unknown): ExplainResult {
  const v = parsed as any;
  if (!v || typeof v !== "object") throw new Error("nothing returned");
  const cause: Cause | null = v.cause && CAUSES.has(v.cause) ? v.cause : null;
  return {
    cause,
    marks_lost: cause && typeof v.marks_lost === "number" && v.marks_lost > 0 ? v.marks_lost : null,
    body: typeof v.body === "string" && v.body.trim() ? v.body.trim() : null,
    do_this_next: typeof v.do_this_next === "string" && v.do_this_next.trim() ? v.do_this_next.trim() : null,
    concepts: Array.isArray(v.concepts) ? v.concepts.filter((c: unknown): c is string => typeof c === "string").slice(0, 6) : [],
  };
}
