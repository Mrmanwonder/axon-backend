import type { InkClass, InkSignals } from "../layout/types";

export interface InkClassification { class: InkClass; confidence: number; scores: Record<InkClass, number> }

export function classifyInk(signals: InkSignals): InkClassification {
  const scores: Record<InkClass, number> = {
    PRINTED: signals.printedProbability * 0.8 + (1 - signals.strokeDifference) * 0.2,
    TEACHER: signals.colourDistanceFromPrint * 0.16 + signals.strokeDifference * 0.2 + signals.marginTendency * 0.24 + signals.annotationOverlap * 0.22 + signals.handwritingDifference * 0.18,
    STUDENT: (1 - signals.marginTendency) * 0.22 + (1 - signals.annotationOverlap) * 0.18 + signals.strokeDifference * 0.2 + (1 - signals.handwritingDifference) * 0.25 + (1 - signals.printedProbability) * 0.15,
    UNKNOWN: 0.25
  };
  const sorted = (Object.entries(scores) as Array<[InkClass, number]>).sort((a, b) => b[1] - a[1]);
  const [best, second] = sorted;
  if (!best || !second || best[1] - second[1] < 0.12) return { class: "UNKNOWN", confidence: 1 - (best?.[1] ?? 0), scores };
  return { class: best[0], confidence: Math.min(1, best[1]), scores };
}
