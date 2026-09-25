import type { CorrectionEvent } from "../../schemas";
import { oneWayHash } from "../security/privacy";

export type LearningTarget =
  | "BENCHMARK_EXPANSION"
  | "CONFIDENCE_RECALIBRATION"
  | "LAYOUT_TRAINING"
  | "HTR_DATASET"
  | "PROMPT_REGRESSION"
  | "ERROR_CLUSTERING";

export interface ActiveLearningCandidate { priority: number; reasons: string[] }
export interface CorrectionLearningPlan {
  predictionCorrect: boolean;
  targets: LearningTarget[];
  calibration?: { confidence: number; bucket: number; predictionCorrect: boolean };
  errorCluster?: { signature: string; field: string; reasons: string[]; highConfidence: boolean };
}

function canonical(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(canonical);
  if (value !== null && typeof value === "object") {
    return Object.fromEntries(Object.entries(value as Record<string, unknown>).sort(([left], [right]) => left.localeCompare(right)).map(([key, item]) => [key, canonical(item)]));
  }
  return value;
}

export function predictionMatchesAccepted(event: CorrectionEvent): boolean {
  return JSON.stringify(canonical(event.predicted)) === JSON.stringify(canonical(event.acceptedValue));
}

function confidenceOf(event: CorrectionEvent): number | undefined {
  const value = event.contextMetadata["confidence"];
  return typeof value === "number" && Number.isFinite(value) && value >= 0 && value <= 1 ? value : undefined;
}

function category(event: CorrectionEvent, key: string): string | undefined {
  const value = event.contextMetadata[key];
  if (typeof value !== "string") return undefined;
  const normalized = value.trim().toLowerCase().replace(/[^a-z0-9_.:-]+/g, "_").slice(0, 64);
  return normalized || undefined;
}

function fieldCategory(field: string): string {
  const normalized = field.trim().toLowerCase();
  if (/recognized.?text/.test(normalized)) return "recognized_text";
  if (/handwrit/.test(normalized)) return "handwriting";
  if (/mark/.test(normalized)) return "mark";
  if (/question/.test(normalized)) return "question";
  if (/teacher/.test(normalized)) return "teacher";
  if (/answer/.test(normalized)) return "answer";
  if (/comment/.test(normalized)) return "comment";
  if (/annotat/.test(normalized)) return "annotation";
  if (/layout/.test(normalized)) return "layout";
  if (/region/.test(normalized)) return "region";
  if (/page/.test(normalized)) return "page";
  if (/box/.test(normalized)) return "box";
  return "other";
}

export function prioritizeCorrection(event: CorrectionEvent): ActiveLearningCandidate {
  const predictionCorrect = predictionMatchesAccepted(event);
  const reasons = [predictionCorrect ? "prediction_confirmed" : "student_correction"];
  let priority = predictionCorrect ? 0.1 : 0.5;
  const confidence = confidenceOf(event);
  if (!predictionCorrect && confidence !== undefined && confidence >= 0.85) { priority += 0.25; reasons.push("high_confidence_error"); }
  if (/mark|question|teacher/i.test(event.field)) { priority += predictionCorrect ? 0.05 : 0.15; reasons.push("high_risk_field"); }
  if (event.contextMetadata["ambiguous"] === true) { priority += 0.05; reasons.push("ambiguous_input"); }
  return { priority: Math.min(1, priority), reasons };
}

export async function planCorrectionLearning(event: CorrectionEvent, reasons: readonly string[]): Promise<CorrectionLearningPlan> {
  const predictionCorrect = predictionMatchesAccepted(event);
  const confidence = confidenceOf(event);
  const targets = new Set<LearningTarget>(["BENCHMARK_EXPANSION"]);
  if (confidence !== undefined) targets.add("CONFIDENCE_RECALIBRATION");
  if (!predictionCorrect) {
    targets.add("PROMPT_REGRESSION");
    targets.add("ERROR_CLUSTERING");
    const regionClass = category(event, "regionClass") ?? "";
    const layer = category(event, "layer") ?? "";
    if (/layout|box|region|question|mark|page/i.test(event.field) || /question|mark|header|footer|diagram|table/.test(regionClass)) targets.add("LAYOUT_TRAINING");
    if (/handwriting|text|answer|comment|annotation|mark/i.test(event.field) || layer === "student" || layer === "teacher") targets.add("HTR_DATASET");
  }
  const clusterMetadata = {
    field: fieldCategory(event.field),
    reasons: [...new Set(reasons)].sort(),
    stage: category(event, "stage"),
    intent: category(event, "intent"),
    layer: category(event, "layer"),
    regionClass: category(event, "regionClass"),
    sourceType: category(event, "sourceType")
  };
  return {
    predictionCorrect,
    targets: [...targets],
    ...(confidence !== undefined ? { calibration: { confidence, bucket: Math.min(9, Math.floor(confidence * 10)), predictionCorrect } } : {}),
    ...(!predictionCorrect ? {
      errorCluster: {
        signature: await oneWayHash(JSON.stringify(clusterMetadata)),
        field: clusterMetadata.field,
        reasons: clusterMetadata.reasons,
        highConfidence: confidence !== undefined && confidence >= 0.85
      }
    } : {})
  };
}
