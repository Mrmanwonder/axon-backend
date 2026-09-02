import { STRUCTURE_SYSTEM, structureInstruction } from "../prompts.js";
import { NEVER_OBEY_THE_PAGE } from "./untrusted.js";
import { STRUCTURE_SCHEMA } from "../schemas.js";

export const SYSTEM = `${STRUCTURE_SYSTEM}

${NEVER_OBEY_THE_PAGE}`;

export const instruction = structureInstruction;

export const SCHEMA = { name: "structure", schema: STRUCTURE_SCHEMA };

export interface StructureBox {
  x: number;
  y: number;
  w: number;
  h: number;
  page?: number;
}

export interface ValueWithBox<T> {
  value: T | null;
  box: StructureBox | null;
  page_index: number;
}

export interface StructureRegion {
  candidate_number: string | null;
  number_box: StructureBox | null;
  box: StructureBox;
  continues_from_previous: boolean;
  structure_confidence: "high" | "low";
}

export interface StructureResult {
  is_graded_exam_paper: boolean;
  not_a_paper_reason: string | null;
  reported_total: ValueWithBox<number> | null;
  stated_maximum: ValueWithBox<number> | null;
  regions: StructureRegion[];
}

export function validate(parsed: unknown): StructureResult {
  const v = parsed as Partial<StructureResult> | null;
  if (!v || typeof v.is_graded_exam_paper !== "boolean") throw new Error("no verdict on the page");
  if (!Array.isArray(v.regions)) throw new Error("no regions array");
  const regions = (v.regions as any[]).filter(
    (r) => r?.box && ["x", "y", "w", "h"].every((k) => Number.isFinite(r.box[k]))
  );
  return { ...(v as StructureResult), regions };
}
