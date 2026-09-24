import type { Evidence } from "../schemas";
import type { LayoutRegion } from "./layout/types";
import type { PageQualityMetrics, QualityAssessment } from "./quality";
import type { QuestionGraph } from "./question-graph";
import type { MarkAssignment } from "./mark-matcher";
import type { RegionRead } from "./recognition";

export type PaperStage = "INGESTED" | "QUALITY_ASSESSED" | "LAYOUT_DETECTED" | "QUESTION_GRAPH_BUILT" | "INK_SEPARATED" | "MARKS_MATCHED" | "CONTENT_READ" | "RECONCILED" | "ACADEMICALLY_VERIFIED" | "REVIEW_REQUIRED" | "TRUSTED_COMMIT";
export interface PaperState {
  paperId: string; pageId: string; originalHash: string; conditionedHash?: string; stage: PaperStage;
  qualityMetrics?: PageQualityMetrics; quality?: QualityAssessment; regions?: LayoutRegion[];
  questionGraph?: QuestionGraph; markAssignments?: MarkAssignment[]; reads?: RegionRead[];
  evidence: Evidence[]; reviewReasons: string[];
}

const ALLOWED: Readonly<Record<PaperStage, readonly PaperStage[]>> = {
  INGESTED: ["QUALITY_ASSESSED"], QUALITY_ASSESSED: ["LAYOUT_DETECTED", "REVIEW_REQUIRED"],
  LAYOUT_DETECTED: ["QUESTION_GRAPH_BUILT", "REVIEW_REQUIRED"], QUESTION_GRAPH_BUILT: ["INK_SEPARATED", "REVIEW_REQUIRED"],
  INK_SEPARATED: ["MARKS_MATCHED", "REVIEW_REQUIRED"], MARKS_MATCHED: ["CONTENT_READ", "REVIEW_REQUIRED"],
  CONTENT_READ: ["RECONCILED", "REVIEW_REQUIRED"], RECONCILED: ["ACADEMICALLY_VERIFIED", "REVIEW_REQUIRED"],
  ACADEMICALLY_VERIFIED: ["TRUSTED_COMMIT", "REVIEW_REQUIRED"], REVIEW_REQUIRED: ["TRUSTED_COMMIT"], TRUSTED_COMMIT: []
};

export function transitionPaper(state: PaperState, next: PaperStage, patch: Partial<Omit<PaperState, "paperId" | "pageId" | "originalHash" | "stage">> = {}): PaperState {
  if (!ALLOWED[state.stage].includes(next)) throw new Error(`Invalid paper transition ${state.stage} -> ${next}`);
  if (next === "TRUSTED_COMMIT" && (state.reviewReasons.length > 0 || state.evidence.some((item) => item.verification === "unverified"))) {
    throw new Error("Cannot trusted-commit unresolved or unverified paper evidence");
  }
  return { ...state, ...patch, stage: next };
}
