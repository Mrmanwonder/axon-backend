// scheme_check.v1 — checks one question on a student's UNMARKED paper against
// the published mark scheme for that exact paper.
//
// Owner decision (6 Oct 2026): on an ungraded paper Axon gives feedback AND an
// estimated mark, always labelled as Axon's estimate. The estimate is never a
// teacher mark, never written to marks_awarded, never counted in analytics.
// The scheme is private input: the student never sees its wording.

import { NEVER_OBEY_THE_PAGE, untrusted } from "./untrusted.js";

export const PROMPT_VERSION = "scheme_check.v1";

export const SYSTEM = `
You are an experienced examiner helping a student check their own practice
paper before anyone has marked it. You have the official mark scheme for this
exact question. The student has not seen it, and must not see its wording.

For the question you are given:
1. Read what the student actually wrote. Judge only what is on the page; do
   not credit what they might have meant.
2. Apply the mark scheme the way an examiner would: method marks (M) for a
   valid method even with an arithmetic slip, accuracy marks (A) only when
   the dependent method mark is earned, independent marks (B), and
   follow-through (FT) where the scheme allows it. Accept equivalent correct
   forms and alternative valid methods the scheme permits.
3. Give an estimated mark out of the question's maximum. It is an estimate:
   say so through "confidence". Use "unsure" whenever the reading of the
   answer is doubtful, the working is partly illegible, or the scheme is
   ambiguous for what was written.
4. Explain, in your own words, what earned credit and what was missing. Never
   copy or closely paraphrase a line of the mark scheme, and never write
   scheme notation (M1, A1, B1, FT, oe, cao, AG) in anything the student reads.
   Describe the idea ("you needed to show the substitution before simplifying"),
   not the scheme's sentence.
5. "do_this_next" is one concrete action the student can do in an exam, tied
   to this answer. If there is nothing useful to say, return null.
6. If the answer is blank, say nothing was written and estimate 0.
7. If you cannot check it (the question in the scheme does not match the
   question on the page, the answer is unreadable, or the scheme section is
   missing), return can_check false with a short reason. An honest gap is
   better than a confident wrong mark.

Tone: direct, calm, no exclamation marks, no praise padding.

${NEVER_OBEY_THE_PAGE}
`.trim();

export interface CheckInstructionOptions {
  paper: string;              // "9231/11 October/November 2025"
  label: string | null;
  marksAvailable: number | null;
  questionText: string | null;
  studentAnswer: string | null;
  scheme: string;             // this question's section of the mark scheme
  conventions: string | null; // the scheme's own notes on mark types
}

export function instruction(o: CheckInstructionOptions): string {
  return [
    `Paper: ${o.paper}`,
    `Question ${o.label ?? "(no label)"}${o.marksAvailable !== null ? `, worth ${o.marksAvailable} mark${o.marksAvailable === 1 ? "" : "s"} on the paper` : ""}.`,
    "",
    o.questionText ? untrusted("question", o.questionText) : "The question text was not read.",
    "",
    o.studentAnswer?.trim() ? untrusted("student answer", o.studentAnswer.slice(0, 6000)) : "No answer was read for this question.",
    "",
    "MARK SCHEME FOR THIS QUESTION (private reference; never quote it):",
    o.scheme,
    ...(o.conventions ? ["", "MARK SCHEME CONVENTIONS (private reference):", o.conventions] : []),
  ].join("\n");
}

export const SCHEMA = {
  name: "scheme_check",
  schema: {
    type: "object",
    additionalProperties: false,
    required: ["can_check", "reason", "estimated_marks", "max_marks", "confidence", "what_was_right", "what_was_missing", "do_this_next"],
    properties: {
      can_check: { type: "boolean" },
      reason: { type: ["string", "null"] },
      estimated_marks: { type: ["number", "null"] },
      max_marks: { type: ["number", "null"] },
      confidence: { type: "string", enum: ["likely", "unsure"] },
      what_was_right: { type: ["string", "null"] },
      what_was_missing: { type: "array", maxItems: 6, items: { type: "string" } },
      do_this_next: { type: ["string", "null"] },
    },
  },
} as const;

export interface CheckResult {
  canCheck: boolean;
  reason: string | null;
  estimatedMarks: number | null;
  maxMarks: number | null;
  confidence: "likely" | "unsure";
  whatWasRight: string | null;
  whatWasMissing: string[];
  doThisNext: string | null;
}

const SCHEME_NOTATION = /\b(?:[MAB]\d|FT|cao|oe|AG|isw|SC\d?)\b/;

function text(value: unknown, max: number): string | null {
  if (typeof value !== "string") return null;
  const t = value.replace(/\s+/g, " ").trim();
  return t ? t.slice(0, max) : null;
}

/** Lines of the scheme long enough to recognise if they were copied out. */
function schemeShingles(scheme: string): string[] {
  return scheme
    .split(/\||\n/)
    .map((s) => s.replace(/\s+/g, " ").trim().toLowerCase())
    .filter((s) => s.length >= 40);
}

/**
 * Structural checks plus the two promises made to the owner: the estimate
 * stays within the question's maximum, and nothing the student reads copies
 * the scheme or carries its notation.
 */
export function validate(parsed: unknown, ctx: { scheme: string; marksAvailable: number | null }): CheckResult {
  const v = parsed as Record<string, unknown> | null;
  if (!v || typeof v.can_check !== "boolean") throw new Error("scheme_check: malformed result");
  if (!v.can_check) {
    return { canCheck: false, reason: text(v.reason, 300) ?? "Could not check this question.", estimatedMarks: null, maxMarks: null, confidence: "unsure", whatWasRight: null, whatWasMissing: [], doThisNext: null };
  }
  const max = ctx.marksAvailable ?? (typeof v.max_marks === "number" ? v.max_marks : null);
  const est = typeof v.estimated_marks === "number" ? v.estimated_marks : null;
  if (max === null || !Number.isFinite(max) || max <= 0) throw new Error("scheme_check: no maximum mark");
  if (est === null || !Number.isFinite(est) || est < 0 || est > max) throw new Error("scheme_check: estimate outside 0..max");
  if (Math.round(est * 2) !== est * 2) throw new Error("scheme_check: estimate not a whole or half mark");

  const right = text(v.what_was_right, 600);
  const missing = Array.isArray(v.what_was_missing) ? v.what_was_missing.map((m) => text(m, 300)).filter((m): m is string => !!m).slice(0, 6) : [];
  const next = text(v.do_this_next, 300);

  const shown = [right, ...missing, next].filter((s): s is string => !!s);
  if (shown.some((s) => SCHEME_NOTATION.test(s))) throw new Error("scheme_check: scheme notation in student-facing text");
  const shingles = schemeShingles(ctx.scheme);
  const lower = shown.map((s) => s.toLowerCase());
  if (shingles.some((line) => lower.some((s) => s.includes(line)))) throw new Error("scheme_check: copied scheme text");

  return {
    canCheck: true,
    reason: null,
    estimatedMarks: est,
    maxMarks: max,
    confidence: v.confidence === "likely" ? "likely" : "unsure",
    whatWasRight: right,
    whatWasMissing: missing,
    doThisNext: next,
  };
}

// ── Header confirmation ──────────────────────────────────────────────────
// Before a scheme is used, a second, stronger read of one page confirms the
// printed paper code. A wrong variant (…/12 read as …/11) would be a real,
// valid scheme for a different paper, so this cannot be skipped.

export const HEADER_SYSTEM = `
Read the Cambridge paper reference printed on this exam page. It appears in the
header or the footer, in the form 9231/11/O/N/25 (syllabus/component/series/year),
or as a code like 9231/11 with the session written out (October/November 2025).
Copy exactly what is printed. If it is not printed or not legible, return null.
Do not infer it from the content of the questions.

${NEVER_OBEY_THE_PAGE}
`.trim();

export const HEADER_SCHEMA = {
  name: "scheme_check_header",
  schema: {
    type: "object",
    additionalProperties: false,
    required: ["reference"],
    properties: { reference: { type: ["string", "null"] } },
  },
} as const;

export function validateHeader(parsed: unknown): { reference: string | null } {
  const v = parsed as { reference?: unknown } | null;
  if (!v || !("reference" in v)) throw new Error("scheme_check header: malformed result");
  return { reference: typeof v.reference === "string" && v.reference.trim() ? v.reference.trim() : null };
}
