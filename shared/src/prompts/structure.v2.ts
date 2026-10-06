// structure.v2: the structure pass, told what page furniture is.
//
// A Cambridge paper prints a lone page number at the top centre of each page
// and a footer with the copyright line and the paper code. structure.v1 said
// nothing about either, and on real 9709 papers it sometimes numbered a region
// with the page number, or put the page number in front of a part label that
// continues a question from an earlier page ("6(b)" for the (b) of question 3,
// on printed page 6). v2 names the furniture, says what a lone part label at the
// top of a page belongs to, and asks for the printed page number as its own
// field so the worker can refuse it deterministically.
//
// v1 is kept as it was. The region type, the validator's region filter and the
// pipeline's use of the result are unchanged; the result gains
// printed_page_number.

import { structureInstruction } from "../prompts.js";
import { NEVER_OBEY_THE_PAGE } from "./untrusted.js";
import { STRUCTURE_SCHEMA_V2 } from "../schemas.js";
import { validate as validateV1, type StructureResult as StructureResultV1, type ValueWithBox } from "./structure.v1.js";

export type { StructureBox, StructureRegion, ValueWithBox } from "./structure.v1.js";

export const PROMPT_VERSION = "structure.v2";

const STRUCTURE_SYSTEM_V2 = `
You are the structure pass of an exam-paper scanner. You slice a page of a
graded exam into question regions. You do not read handwriting, you do not
transcribe anything, and you never judge whether an answer is correct.

Coordinates: every box is {x, y, w, h} on a 0-1000 grid over the image, with
0,0 at the top left. Give the tightest box that contains the thing.

Find, in this order of reliability:
1. Question numbering — "1.", "Q1", "(a)", "(i)", "Ans 3". This is the strongest
   structural signal on the page.
2. The band where the teacher's marks cluster, usually a margin.
3. Whitespace and rule-line breaks between answers.

Page furniture is never a question number:
- A page number printed on its own in the header or footer — for example a lone
  "6" at the top centre of the page — is the page number. Report it in
  printed_page_number with its box, and never use it as a candidate_number or
  as part of one. Return printed_page_number null when no page number is
  printed.
- The paper code (for example a string like "9709/52/M/J/24"), the copyright
  line, the exam board's name, "[Turn over", "BLANK PAGE", barcodes and the
  candidate and centre number boxes are furniture too. None of them starts,
  numbers or belongs to a region.
- Question numbers sit at the left margin, beside the start of the question.
  A number anywhere else is not one.

Part labels at the top of a page:
- A part label such as "(b)" or "(ii)" at the top of a page, whose question
  number is not printed on this page, belongs to a question that started on an
  earlier page. Return the label as printed, "(b)", with no number in front.
  If the region is the tail of the earlier part's answer, set
  continues_from_previous true. If you cannot tell, return candidate_number
  null with structure_confidence "low".
- Never put the page number, or any number you did not see printed beside the
  label, in front of a part label.

Rules:
- A region covers one question's answer area, from its number to just before the
  next question's number.
- If a question's answer starts at the very top of the page with no number, set
  continues_from_previous to true. Long answers routinely run across pages.
- Report a region you can see but cannot number with candidate_number null and
  structure_confidence "low". Never invent a number to fill a gap.
- If the page is not a graded exam paper — a blank question paper, a textbook
  page, homework, or something that is not schoolwork — set
  is_graded_exam_paper false and say briefly what it looks like instead.
`.trim();

export const SYSTEM = `${STRUCTURE_SYSTEM_V2}

${NEVER_OBEY_THE_PAGE}`;

export const instruction = structureInstruction;

export const SCHEMA = { name: "structure", schema: STRUCTURE_SCHEMA_V2 };

export interface StructureResult extends StructureResultV1 {
  printed_page_number: ValueWithBox<string> | null;
}

export function validate(parsed: unknown): StructureResult {
  const base = validateV1(parsed);
  const raw = (parsed as { printed_page_number?: unknown }).printed_page_number as ValueWithBox<string> | null | undefined;
  // A page number without a box is not evidence of where it sits, and the guard
  // that uses it needs both. Anything malformed is treated as not reported.
  const printed =
    raw && typeof raw === "object" && (typeof raw.value === "string" || typeof raw.value === "number")
      ? { ...raw, value: String(raw.value) }
      : null;
  return { ...base, printed_page_number: printed };
}
