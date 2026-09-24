import type { Box } from "../../schemas";

export type RegionClass = "question_number" | "subquestion_number" | "printed_question" | "student_answer" | "teacher_annotation" | "teacher_comment" | "marginal_mark" | "marks_available" | "reported_total" | "diagram" | "table" | "working_area" | "header" | "footer" | "page_number" | "crossed_out_work" | "continuation_region";
export interface LayoutRegion { id: string; pageId: string; class: RegionClass; box: Box; confidence: number; text?: string }
export type InkClass = "PRINTED" | "STUDENT" | "TEACHER" | "UNKNOWN";
export interface InkSignals { printedProbability: number; colourDistanceFromPrint: number; strokeDifference: number; marginTendency: number; annotationOverlap: number; handwritingDifference: number }
