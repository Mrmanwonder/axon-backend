export type PageQualityClass = "CLEAR" | "RECOVERABLE" | "AMBIGUOUS" | "UNREADABLE";
export interface PageQualityMetrics {
  blur: number;
  glareFraction: number;
  perspectiveDegrees: number;
  resolution: number;
  compression: number;
  cropCompleteness: number;
  shadowFraction: number;
}
export interface QualityAssessment { classification: PageQualityClass; score: number; requiredAction: "normal" | "condition" | "independent_reader" | "rescan"; reasons: string[] }

const clamp = (value: number): number => Math.max(0, Math.min(1, value));

export function assessPageQuality(metrics: PageQualityMetrics): QualityAssessment {
  const penalties = {
    blur: clamp(metrics.blur), glare: clamp(metrics.glareFraction * 4), perspective: clamp(metrics.perspectiveDegrees / 15),
    resolution: clamp(1 - metrics.resolution), compression: clamp(metrics.compression), crop: clamp(1 - metrics.cropCompleteness), shadow: clamp(metrics.shadowFraction * 2)
  };
  const score = clamp(1 - (penalties.blur * 0.24 + penalties.glare * 0.16 + penalties.perspective * 0.12 + penalties.resolution * 0.2 + penalties.compression * 0.1 + penalties.crop * 0.12 + penalties.shadow * 0.06));
  const reasons = Object.entries(penalties).filter(([, value]) => value >= 0.35).map(([name]) => name);
  if (score >= 0.82) return { classification: "CLEAR", score, requiredAction: "normal", reasons };
  if (score >= 0.62) return { classification: "RECOVERABLE", score, requiredAction: "condition", reasons };
  if (score >= 0.38) return { classification: "AMBIGUOUS", score, requiredAction: "independent_reader", reasons };
  return { classification: "UNREADABLE", score, requiredAction: "rescan", reasons };
}

export interface ConditioningPlan { rotateDegrees: 0 | 90 | 180 | 270; deskew: boolean; correctPerspective: boolean; normalizeIllumination: boolean; normalizeContrast: boolean; generativeEnhancement: false }

export function conditioningPlan(metrics: PageQualityMetrics, orientationDegrees: number): ConditioningPlan {
  const normalized = ((Math.round(orientationDegrees / 90) * 90) % 360 + 360) % 360;
  return {
    rotateDegrees: normalized as 0 | 90 | 180 | 270,
    deskew: Math.abs(metrics.perspectiveDegrees) > 0.5,
    correctPerspective: Math.abs(metrics.perspectiveDegrees) > 1.5,
    normalizeIllumination: metrics.shadowFraction > 0.03 || metrics.glareFraction > 0.01,
    normalizeContrast: metrics.compression > 0.15 || metrics.resolution < 0.8,
    generativeEnhancement: false
  };
}
