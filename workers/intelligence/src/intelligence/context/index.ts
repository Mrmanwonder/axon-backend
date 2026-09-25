import type { Evidence, Intent } from "../../schemas";

export interface VerifiedObservation { conceptId: string; paperId: string; correct: boolean; confidence: number; trustState: "AUTO_VERIFIED" | "STUDENT_VERIFIED" | "UNVERIFIED" | "UNKNOWN" }
export interface PatternResult { established: boolean; relevantObservations: number; distinctPapers: number; confidence: number }

export function detectPattern(observations: readonly VerifiedObservation[], minimumCount = 4, minimumPapers = 2, threshold = 0.8): PatternResult {
  const trusted = observations.filter((item) => item.trustState === "AUTO_VERIFIED" || item.trustState === "STUDENT_VERIFIED");
  const distinctPapers = new Set(trusted.map((item) => item.paperId)).size;
  const confidence = trusted.length ? trusted.reduce((sum, item) => sum + item.confidence, 0) / trusted.length : 0;
  return { established: trusted.length >= minimumCount && distinctPapers >= minimumPapers && confidence >= threshold, relevantObservations: trusted.length, distinctPapers, confidence };
}

export interface ContextInput { intent: Intent; subject?: string; topic?: string; paperId?: string; evidence: readonly Evidence[] }

export function assembleContext(input: ContextInput): Evidence[] {
  return input.evidence.filter((item) => {
    if (input.paperId && item.provenance.paperId === input.paperId) return true;
    if (item.source === "student" || item.source === "tool" || item.source === "retrieval" || item.source === "official_source" || item.source === "stable_knowledge") return true;
    if (input.intent === "paper_feedback" && item.source === "teacher") return true;
    return item.informationClass === "OBSERVED" || item.informationClass === "DERIVED";
  }).slice(0, 100);
}
