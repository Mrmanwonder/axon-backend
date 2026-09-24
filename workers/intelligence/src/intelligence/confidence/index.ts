export interface ConfidenceFeatures {
  imageQuality?: number; layoutProbability?: number; readerAgreement?: number; markAssignmentMargin?: number;
  arithmeticConsistency?: number; academicToolValidation?: number; retrievalAuthority?: number;
  sourceAgreement?: number; historicalAccuracy?: number;
}

const WEIGHTS: Readonly<Record<keyof ConfidenceFeatures, number>> = {
  imageQuality: 0.14, layoutProbability: 0.12, readerAgreement: 0.15, markAssignmentMargin: 0.12,
  arithmeticConsistency: 0.13, academicToolValidation: 0.13, retrievalAuthority: 0.08,
  sourceAgreement: 0.08, historicalAccuracy: 0.05
};

export function evidenceConfidence(features: ConfidenceFeatures): number {
  let weighted = 0; let weight = 0;
  for (const [name, configuredWeight] of Object.entries(WEIGHTS) as Array<[keyof ConfidenceFeatures, number]>) {
    const value = features[name];
    if (value === undefined) continue;
    weighted += Math.max(0, Math.min(1, value)) * configuredWeight;
    weight += configuredWeight;
  }
  return weight === 0 ? 0 : weighted / weight;
}

export interface CalibrationPoint { confidence: number; correct: boolean }
export interface CalibrationMetrics { brierScore: number; expectedCalibrationError: number; bins: Array<{ lower: number; upper: number; count: number; meanConfidence: number; accuracy: number }> }

export function calibrationMetrics(points: readonly CalibrationPoint[], binCount = 10): CalibrationMetrics {
  if (points.length === 0) return { brierScore: 0, expectedCalibrationError: 0, bins: [] };
  const bins = Array.from({ length: binCount }, (_, index) => ({ lower: index / binCount, upper: (index + 1) / binCount, points: [] as CalibrationPoint[] }));
  for (const point of points) bins[Math.min(binCount - 1, Math.floor(Math.max(0, Math.min(0.999999, point.confidence)) * binCount))].points.push(point);
  const summarized = bins.filter((bin) => bin.points.length > 0).map((bin) => ({
    lower: bin.lower, upper: bin.upper, count: bin.points.length,
    meanConfidence: bin.points.reduce((sum, point) => sum + point.confidence, 0) / bin.points.length,
    accuracy: bin.points.filter((point) => point.correct).length / bin.points.length
  }));
  const brierScore = points.reduce((sum, point) => sum + (point.confidence - (point.correct ? 1 : 0)) ** 2, 0) / points.length;
  const expectedCalibrationError = summarized.reduce((sum, bin) => sum + bin.count / points.length * Math.abs(bin.accuracy - bin.meanConfidence), 0);
  return { brierScore, expectedCalibrationError, bins: summarized };
}

export type EscalationAction = "ACCEPT_FAST" | "STRONG_READER" | "INDEPENDENT_ADJUDICATION" | "STUDENT_REVIEW";
export function escalationAction(confidence: number, attempt: 0 | 1 | 2): EscalationAction {
  if (confidence >= 0.92) return attempt === 0 ? "ACCEPT_FAST" : "STRONG_READER";
  if (attempt === 0) return "STRONG_READER";
  if (attempt === 1) return "INDEPENDENT_ADJUDICATION";
  return "STUDENT_REVIEW";
}
