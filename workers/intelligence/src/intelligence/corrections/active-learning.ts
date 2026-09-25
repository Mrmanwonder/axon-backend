import type { CorrectionEvent } from "../../schemas";

export interface ActiveLearningCandidate { priority: number; reasons: string[] }

export function prioritizeCorrection(event: CorrectionEvent): ActiveLearningCandidate {
  const reasons = ["student_correction"];
  let priority = 0.5;
  const confidence = event.contextMetadata["confidence"];
  if (typeof confidence === "number" && confidence >= 0.85) { priority += 0.25; reasons.push("high_confidence_error"); }
  if (/mark|question|teacher/i.test(event.field)) { priority += 0.15; reasons.push("high_risk_field"); }
  if (event.contextMetadata["ambiguous"] === true) { priority += 0.05; reasons.push("ambiguous_input"); }
  return { priority: Math.min(1, priority), reasons };
}
