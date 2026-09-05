import { EXPLANATION_SYSTEM } from "../prompts.js";
import { NEVER_OBEY_THE_PAGE, untrusted } from "./untrusted.js";
import { EXPLANATION_SCHEMA } from "../schemas.js";
import { canonicalCommandWord, universalMeaning, type CommandWord } from "../command_words.js";

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
infer a command word from the shape of the answer.

command_word_note is one line on what that word requires of an answer — what
"Explain" wants that "State" does not. It is about the word, not about this
student. For the words Cambridge defines identically across every syllabus, its
published definition is used instead of yours, so write this for the
subject-specific words: what the word asks for *in this subject*, where the
current syllabus is the authority. Never state or imply how many marks a
command word is worth — mark allocation comes from the question, not the word.

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
with the marks, the cause, an error_type, and a note anchored in the student's
own working ("between your line 2 and line 3"). Two separate mistakes are two
entries; one mistake is an empty array, not one entry restating the flat cause.
The marks across the entries must not exceed the marks actually lost. Do not
invent a second reason to fill the array.

error_type says what the mistake looked like, where cause says why it happened.
Read it off the student's own working, not off any scheme:

  method         — the approach taken was wrong or misapplied, whatever the
                   number at the end
  final_answer   — the working was sound and the value reached was not: an
                   arithmetic slip, a sign, a unit, a rounding
  omitted_step   — something the answer needed was never written down, whether
                   or not the student knew it
  presentation   — the answer is right and the way it was set down cost the
                   mark: unreadable, unlabelled, out of order, no units shown
  other          — none of those fits. Use it rather than forcing one.

You have no marking scheme for this paper, so do not label these deductions with
mark-scheme notation. No M1, no A1, no B marks, no "method mark" or "accuracy
mark". That vocabulary belongs to a scheme you have not been given, writing it
here would be reconstructing one, and it is not ours to reproduce. error_type is
the field for this, and its words are the only words for it.

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
 * Axon's own account of what went wrong at a step, as distinct from why.
 *
 * `cause` says what the student was missing; this says what the mistake looked
 * like structurally — working shown but the wrong number reached, a required
 * step never written down, the right answer made unreadable. It is derived by
 * comparing the student's own working against what the step is doing, so it
 * needs no marking scheme and is available on every paper.
 *
 * Deliberately none of Cambridge's vocabulary. These are whole words that get
 * rendered as whole words: nothing here may be abbreviated to a letter or
 * shown in a way that could be mistaken for real mark-scheme notation.
 */
export type ErrorType = "method" | "final_answer" | "omitted_step" | "presentation" | "other";

const ERROR_TYPES = new Set<string>([
  "method", "final_answer", "omitted_step", "presentation", "other",
]);

/**
 * One distinct way a mark went on this question.
 *
 * `mark_type` is Cambridge mark-scheme notation — M for method, A for accuracy,
 * B for independent, C for communication. This prompt cannot set it and the
 * model is never asked for it, for two reasons that stack:
 *
 * Hard rule 2 — every paper reaching this prompt is Tier 1, which by definition
 * has no scheme in the library, so a code assigned here would be reconstructed
 * from nothing. A wrong code is worse than none: it wears official notation
 * while contradicting what the teacher actually marked.
 *
 * And rights — Cambridge and Pearson refused third-party reproduction, so
 * official scheme content is CBSE-only. Mimicking Cambridge's marking system is
 * not ours to do whether or not the guess lands.
 *
 * The field stays for a genuine Tier 2 prompt with licensed scheme text in
 * context. The gate is structural rather than remembered: this prompt is handed
 * no scheme, so there is nothing to base a code on, and `lossReasons()` pins the
 * field to null regardless of what comes back.
 */
export interface LossReason {
  mark_type: "M" | "A" | "B" | "C" | null;
  error_type: ErrorType;
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
    // Cambridge's own wording wins where Cambridge has one. For the 22 words it
    // defines identically across every syllabus there is a published
    // definition, and a model paraphrase of a definition that already exists is
    // an invention we do not need to risk. The model's note is the fallback,
    // used only for the subject-specific words Cambridge deliberately does not
    // define once — there the current syllabus is the authority, not any list.
    command_word_note: commandWord
      ? universalMeaning(commandWord) ?? text(v.command_word_note)
      : null,
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
      // Never set here. See LossReason — this is scheme notation, this is the
      // prompt for papers that have no scheme, and it is not ours to reproduce.
      mark_type: null,
      // "other" rather than a drop: the reason is still a real diagnosis with
      // real marks against it, and the chip simply does not render.
      error_type: ERROR_TYPES.has(r.error_type) ? r.error_type : "other",
      marks,
      cause,
      note: text(r.note),
    });
  }
  // Six is already more decomposition than a question can carry; past that the
  // model is splitting hairs rather than diagnosing.
  return reasons.slice(0, 6);
}
