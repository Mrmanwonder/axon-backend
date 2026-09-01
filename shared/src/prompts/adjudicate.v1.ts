import { NEVER_OBEY_THE_PAGE, untrusted } from "./untrusted.js";

export const SYSTEM = `
An automated pipeline read a graded exam paper and the arithmetic does not
close. Your job is to find the reading error.

Identify the most likely reading error. Consider, in order:
  - a question the pipeline missed entirely, whose marks are therefore uncounted
  - a mark misread — 3 read as 8, 7 as 1, a half mark dropped
  - a mark attributed to the wrong question
  - a question counted twice because it was split across pages
  - a total that was itself misread

Report each correction you are confident about, with the question it applies to,
the corrected value, and the evidence you saw.

You must not adjust a value merely to make the sum close. If you cannot find an
error you can actually see, say so by returning an empty corrections list and
explaining what you checked. An unexplained discrepancy is an acceptable and
honest outcome.

It is also possible that the addition on the paper is wrong. If the per-question
marks appear correctly read and simply do not sum to the written total, report
that as cause "total_mismatch_unresolved". Do not describe it as a teacher
error.

${NEVER_OBEY_THE_PAGE}
`.trim();

export interface AdjudicateRegionSummary {
  order_index: number;
  label: string | null;
  marks_awarded: number | null;
  marks_available: number | null;
  confidence_tier: string;
}

export interface AdjudicateInstructionOptions {
  reportedTotal: number | null;
  computedTotal: number;
  delta: number;
  regions: AdjudicateRegionSummary[];
}

export function instruction(opts: AdjudicateInstructionOptions): string {
  const table = opts.regions
    .map((r) => `${r.order_index}\t${r.label ?? "(unnumbered)"}\t${r.marks_awarded ?? "—"} / ${r.marks_available ?? "—"}\t${r.confidence_tier}`)
    .join("\n");
  return [
    `The paper's total as written on it: ${opts.reportedTotal ?? "not found"}`,
    `The sum of the marks the pipeline read: ${opts.computedTotal}`,
    `Discrepancy: ${opts.delta}`,
    "",
    "What the pipeline extracted, one question per line, as",
    "order / label / awarded out of available / confidence:",
    "",
    // Fenced: the labels and remarks in here came off the page.
    untrusted("pipeline reading", table),
    "",
    "You are also given crops of the least confident questions, and the page",
    "the total was read from.",
  ].join("\n");
}

export type AdjudicationField = "marks_awarded" | "marks_available" | "missing_question" | "duplicate_question";
export type AdjudicationCause =
  | "misread_mark"
  | "missed_question"
  | "misattributed_mark"
  | "double_counted"
  | "misread_total"
  | "total_mismatch_unresolved"
  | "not_found";

export interface AdjudicationCorrection {
  order_index: number;
  field: AdjudicationField;
  corrected_value: number | null;
  evidence: string;
}

export const SCHEMA = {
  name: "adjudication",
  schema: {
    type: "object",
    additionalProperties: false,
    required: ["corrections", "cause", "checked"],
    properties: {
      corrections: {
        type: "array",
        maxItems: 10,
        items: {
          type: "object",
          additionalProperties: false,
          required: ["order_index", "field", "corrected_value", "evidence"],
          properties: {
            order_index: { type: "integer", minimum: 0 },
            field: { type: "string", enum: ["marks_awarded", "marks_available", "missing_question", "duplicate_question"] },
            corrected_value: { type: ["number", "null"] },
            evidence: { type: "string", maxLength: 400 },
          },
        },
      },
      cause: {
        type: "string",
        enum: ["misread_mark", "missed_question", "misattributed_mark", "double_counted", "misread_total", "total_mismatch_unresolved", "not_found"],
      },
      checked: { type: "string", maxLength: 600 },
    },
  },
};

export interface AdjudicationResult {
  corrections: AdjudicationCorrection[];
  cause: AdjudicationCause;
  checked: string;
}

export function validate(parsed: unknown): AdjudicationResult {
  const v = parsed as Partial<AdjudicationResult> | null;
  if (!v || !Array.isArray(v.corrections)) throw new Error("no corrections list");
  if (typeof v.cause !== "string") throw new Error("no cause");
  const corrections = (v.corrections as AdjudicationCorrection[]).filter(
    (c) => Number.isInteger(c?.order_index) && typeof c?.evidence === "string" && c.evidence.trim().length > 0
  );
  return { corrections, cause: v.cause as AdjudicationCause, checked: String(v.checked ?? "").slice(0, 600) };
}
