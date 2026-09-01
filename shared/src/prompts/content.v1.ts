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

export function validate(parsed: unknown): ContentResult {
  const v = parsed as Partial<ContentResult> | null;
  if (!v || typeof v.unreadable !== "boolean") throw new Error("no readability verdict");
  const awarded = v.marks_awarded?.value;
  const available = v.marks_available?.value;
  if (typeof awarded === "number" && typeof available === "number" && awarded > available) {
    throw new Error(`read ${awarded} out of ${available}`);
  }
  return v as ContentResult;
}
