import { EXPLANATION_SYSTEM } from "../prompts.js";
import { NEVER_OBEY_THE_PAGE, untrusted } from "./untrusted.js";
import { EXPLANATION_SCHEMA } from "../schemas.js";
import { canonicalCommandWord, type CommandWord } from "../command_words.js";

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

This is a Cambridge (CAIE) paper. Four more things are asked of you, and each
of them renders nothing at all if you cannot do it honestly.

command_word — the Cambridge command word the question is built around: State,
Explain, Calculate, Show that, Justify, Determine, Describe, Evaluate, Suggest,
Define and the rest of the standard list. Take it from the question stem; if the
stem was not readable, or the word is not one of those, return null. Do not
infer a command word from the shape of the answer. command_word_note is one
line on what that word requires of an answer — what "Explain" wants that
"State" does not. It is about the word, not about this student.

model_answer — the corrected working, written out in the same steps as the
student's own answer, so the two can be read side by side. This is what
do_this_next cannot be: do_this_next names the fix in one line, and this carries
it through to the end. Write it as working, not as advice, and keep the
student's own method where their method was sound — a correct answer by a
different route is not a correction. It is a demonstration of how the question
is answered, never a claim about what this attempt was worth: do not write that
it would have scored full marks, and do not compare it to the mark the teacher
gave. If you cannot produce complete correct working, return null.

loss_reasons — where the deduction breaks into distinct parts, one entry each,
with the marks, the cause, and a note anchored in the student's own working
("between your line 2 and line 3"). Two separate mistakes are two entries; one
mistake is an empty array, not one entry restating the flat cause. The marks
across the entries must not exceed the marks actually lost. Do not invent a
second reason to fill the array.

You have no marking scheme for this paper, so do not label these deductions with
mark-scheme notation. No M1, no A1, no B marks, no "method mark" or "accuracy
mark". That vocabulary belongs to a scheme you have not been given, and writing
it here would be reconstructing one.

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

/**
 * One distinct way a mark went on this question.
 *
 * `mark_type` is Cambridge mark-scheme vocabulary — M for method, A for
 * accuracy, B for independent, C for communication. It is deliberately NOT
 * something this prompt can set, and the model is never asked for it: on a
 * Tier 1 paper there is no official scheme, and labelling a deduction "M1"
 * without one is reconstructing scheme language, which hard rule 2 forbids in
 * the same breath as inventing the scheme itself. The field exists so a Tier 2
 * prompt grounded in a real `canonical_question.marking_scheme` can fill it,
 * and until that prompt exists it is null on every row.
 */
export interface LossReason {
  mark_type: "M" | "A" | "B" | "C" | null;
  marks: number;
  cause: Cause | null;
  note: string | null;
}

export interface ExplainResult {
  cause: Cause | null;
  marks_lost: number | null;
  body: string | null;
  do_this_next: string | null;
  concepts: string[];
  command_word: CommandWord | null;
  command_word_note: string | null;
  model_answer: string | null;
  loss_reasons: LossReason[];
}

/** Trimmed, or null for anything that is not a non-empty string. */
function text(value: unknown): string | null {
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

export function validate(parsed: unknown): ExplainResult {
  const v = parsed as any;
  if (!v || typeof v !== "object") throw new Error("nothing returned");
  const cause: Cause | null = v.cause && CAUSES.has(v.cause) ? v.cause : null;
  const commandWord = canonicalCommandWord(v.command_word);
  return {
    cause,
    marks_lost: cause && typeof v.marks_lost === "number" && v.marks_lost > 0 ? v.marks_lost : null,
    // EXPLANATION_SCHEMA calls this field `explanation`; the column it lands in
    // is `region_explanation.body`. This function read `v.body` — a key the
    // model is never asked for and never sends — so the prose every student was
    // meant to read was dropped here, silently, on every question ever
    // explained. All nine rows in production have body = null. The model was
    // writing it the whole time.
    //
    // `body` stays accepted as a fallback: it costs nothing, and it is the name
    // the rest of the pipeline uses, so a future prompt that emits it is not a
    // second silent outage.
    body: text(v.explanation) ?? text(v.body),
    do_this_next: text(v.do_this_next),
    concepts: Array.isArray(v.concepts) ? v.concepts.filter((c: unknown): c is string => typeof c === "string").slice(0, 6) : [],
    // Unrecognised words are dropped rather than passed through, so the tag
    // stays countable. The note goes with the word: a note explaining a word we
    // did not accept would be a caption on a missing picture.
    command_word: commandWord,
    command_word_note: commandWord ? text(v.command_word_note) : null,
    model_answer: text(v.model_answer),
    loss_reasons: lossReasons(v.loss_reasons),
  };
}

/**
 * The decomposed diagnoses, or an empty array.
 *
 * Empty is a normal outcome, not a failure: a single-cause question has nothing
 * to decompose, and QuestionDetail falls back to the flat cause when this is
 * empty. A reason with no cause we recognise, or no marks, is dropped — the
 * same refusal to store an eighth cause that `validate` makes above.
 */
function lossReasons(value: unknown): LossReason[] {
  if (!Array.isArray(value)) return [];
  const reasons: LossReason[] = [];
  for (const raw of value) {
    if (!raw || typeof raw !== "object") continue;
    const r = raw as any;
    const cause: Cause | null = r.cause && CAUSES.has(r.cause) ? r.cause : null;
    const marks = typeof r.marks === "number" && r.marks > 0 ? r.marks : null;
    if (!cause || marks === null) continue;
    reasons.push({
      // Never set here. See LossReason — this is scheme vocabulary and this is
      // the prompt for papers that have no scheme.
      mark_type: null,
      marks,
      cause,
      note: text(r.note),
    });
  }
  // Six is already more decomposition than a question can carry; past that the
  // model is splitting hairs rather than diagnosing.
  return reasons.slice(0, 6);
}
