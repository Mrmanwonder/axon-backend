import { CONTENT_SYSTEM, contentInstruction, type ContentInstructionOptions } from "../prompts.js";
import { NEVER_OBEY_THE_PAGE, NULL_IS_AN_ANSWER } from "./untrusted.js";
import { CONTENT_SCHEMA } from "../schemas.js";
import type { ValueWithBox } from "./structure.v1.js";

export const SYSTEM = `${CONTENT_SYSTEM}

${NULL_IS_AN_ANSWER}

${NEVER_OBEY_THE_PAGE}`;

export const instruction = contentInstruction;
export type { ContentInstructionOptions };

export const SCHEMA = { name: "content", schema: CONTENT_SCHEMA };

export type RegionType = "prose" | "math" | "diagram" | "table" | "mcq" | "mixed";

export interface ContentResult {
  question_label?: ValueWithBox<string> | null;
  question_text: ValueWithBox<string> | null;
  student_answer: ValueWithBox<string> | null;
  marks_awarded: ValueWithBox<number> | null;
  marks_available: ValueWithBox<number> | null;
  teacher_remark: ValueWithBox<string> | null;
  region_type: RegionType;
  recognition_confidence: "high" | "medium" | "low";
  unreadable: boolean;
  unreadable_reason: string | null;
}

/**
 * A mark that is not a whole number was not read off a Cambridge paper.
 *
 * CAIE awards integers at IGCSE, AS and A Level, so 0.5 is not a mark a teacher
 * could have written — it is a misread, and the honest thing is to have read
 * nothing rather than to store a number that cannot exist. Dropped to null and
 * the recognition downgraded, which routes the question to review with its crop
 * so a person can say what the pen actually says.
 *
 * Caught here rather than at the database, where a CHECK now refuses it
 * (20260906091000_whole_marks_only). Both belong: the constraint is the
 * guarantee, and this is what keeps one bad digit from failing the whole run
 * instead of the one question it actually concerns.
 */
function wholeMark(field: ValueWithBox<number> | null | undefined): {
  field: ValueWithBox<number> | null;
  dropped: boolean;
} {
  if (!field || typeof field.value !== "number") return { field: field ?? null, dropped: false };
  if (Number.isInteger(field.value)) return { field, dropped: false };
  return { field: { ...field, value: null as unknown as number }, dropped: true };
}

export function validate(parsed: unknown): ContentResult {
  const v = parsed as Partial<ContentResult> | null;
  if (!v || typeof v.unreadable !== "boolean") throw new Error("no readability verdict");

  const a = wholeMark(v.marks_awarded);
  const m = wholeMark(v.marks_available);
  if (a.dropped || m.dropped) {
    v.marks_awarded = a.field;
    v.marks_available = m.field;
    // Never silently: the question goes to the student with its crop rather
    // than being quietly dropped or quietly rounded.
    v.recognition_confidence = "low";
  }

  const awarded = v.marks_awarded?.value;
  const available = v.marks_available?.value;
  if (typeof awarded === "number" && typeof available === "number" && awarded > available) {
    throw new Error(`read ${awarded} out of ${available}`);
  }
  return v as ContentResult;
}
