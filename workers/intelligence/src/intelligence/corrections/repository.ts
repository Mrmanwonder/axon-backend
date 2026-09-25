import { Type } from "@sinclair/typebox";
import { parseSchema } from "../../schemas";
import type { CorrectionLearningPlan, LearningTarget } from "./active-learning";
import type { LearningConsentDecision } from "./consent";

const ReviewSchema = Type.Object({
  status: Type.Union([Type.Literal("LABELLED"), Type.Literal("PROMOTED"), Type.Literal("REJECTED")]),
  reviewer: Type.String({ minLength: 1, maxLength: 128 }),
  evidenceUri: Type.String({ minLength: 1, maxLength: 2_000 }),
  evalRunId: Type.Optional(Type.String({ minLength: 1, maxLength: 128 }))
}, { additionalProperties: false });

const ALLOWED: Readonly<Record<string, readonly string[]>> = {
  QUEUED: ["LABELLED", "REJECTED"], LABELLED: ["PROMOTED", "REJECTED"], PROMOTED: [], REJECTED: []
};

const LEARNING_TARGETS = new Set<LearningTarget>([
  "BENCHMARK_EXPANSION", "CONFIDENCE_RECALIBRATION", "LAYOUT_TRAINING",
  "HTR_DATASET", "PROMPT_REGRESSION", "ERROR_CLUSTERING"
]);
const TARGET_STATUSES = new Set(["QUEUED", "LABELLED", "READY", "EXPORTED", "REJECTED"]);

export function correctionLearningStatements(
  db: D1Database,
  correctionId: string,
  plan: CorrectionLearningPlan,
  createdAt: string
): D1PreparedStatement[] {
  const statements = plan.targets.map((target) => db.prepare(`INSERT INTO active_learning_target
    (id, correction_id, target, status, created_at) VALUES (?, ?, ?, 'QUEUED', ?)`)
    .bind(crypto.randomUUID(), correctionId, target, createdAt));
  if (plan.calibration) {
    statements.push(db.prepare(`INSERT INTO confidence_calibration_observation
      (id, correction_id, confidence, confidence_bucket, prediction_correct, created_at) VALUES (?, ?, ?, ?, ?, ?)`)
      .bind(crypto.randomUUID(), correctionId, plan.calibration.confidence, plan.calibration.bucket, plan.calibration.predictionCorrect ? 1 : 0, createdAt));
  }
  if (plan.errorCluster) {
    statements.push(db.prepare(`INSERT INTO correction_error_cluster
      (signature, field, reasons_json, correction_count, high_confidence_count, first_seen_at, last_seen_at)
      VALUES (?, ?, ?, 1, ?, ?, ?)
      ON CONFLICT(signature) DO UPDATE SET
        correction_count = correction_error_cluster.correction_count + 1,
        high_confidence_count = correction_error_cluster.high_confidence_count + excluded.high_confidence_count,
        last_seen_at = excluded.last_seen_at`)
      .bind(plan.errorCluster.signature, plan.errorCluster.field, JSON.stringify(plan.errorCluster.reasons), plan.errorCluster.highConfidence ? 1 : 0, createdAt, createdAt));
  }
  return statements;
}

export async function listActiveLearning(db: D1Database, status = "QUEUED", limit = 50): Promise<unknown[]> {
  if (!["QUEUED", "LABELLED", "PROMOTED", "REJECTED"].includes(status)) throw new Error("Invalid active-learning status");
  const boundedLimit = Math.max(1, Math.min(200, limit));
  const rows = await db.prepare(`SELECT al.id, al.correction_id, al.priority, al.reasons_json, al.status, al.reviewer, al.evidence_uri, al.eval_run_id, al.created_at, al.updated_at,
    sc.field, sc.artifact_id, sc.pipeline_version, sc.model, sc.prompt_hash
    FROM active_learning_queue al JOIN student_correction sc ON sc.id = al.correction_id
    WHERE al.status = ? ORDER BY al.priority DESC, al.created_at LIMIT ?`).bind(status, boundedLimit).all();
  return rows.results;
}

export async function reviewActiveLearning(
  db: D1Database,
  id: string,
  input: unknown,
  consent: { decision: LearningConsentDecision; studentPseudonym?: string }
): Promise<void> {
  const request = parseSchema(ReviewSchema, input);
  const current = await db.prepare(`SELECT al.status, al.correction_id, sc.student_id, sc.learning_consent_granted
    FROM active_learning_queue al JOIN student_correction sc ON sc.id = al.correction_id WHERE al.id = ?`)
    .bind(id).first<{ status: string; correction_id: string; student_id: string | null; learning_consent_granted: number }>();
  if (!current) throw new Error("Active-learning item not found");
  if (!(ALLOWED[current.status] ?? []).includes(request.status)) throw new Error(`Invalid active-learning transition ${current.status} -> ${request.status}`);
  if (request.status === "PROMOTED") {
    if (current.learning_consent_granted !== 1 || consent.decision.state !== "GRANTED" || !consent.studentPseudonym || consent.studentPseudonym !== current.student_id) {
      throw new Error("Promotion requires current improve_extraction consent for the correction owner");
    }
    if (!request.evalRunId) throw new Error("Promotion requires a passing evaluation run");
    const run = await db.prepare("SELECT passed FROM eval_run WHERE id = ?").bind(request.evalRunId).first<{ passed: number }>();
    if (run?.passed !== 1) throw new Error("Promotion requires a passing evaluation run");
  }
  const updatedAt = new Date().toISOString();
  const targetStatus = request.status === "PROMOTED" ? "READY" : request.status;
  await db.batch([
    db.prepare("UPDATE active_learning_queue SET status = ?, reviewer = ?, evidence_uri = ?, eval_run_id = ?, updated_at = ? WHERE id = ?")
      .bind(request.status, request.reviewer, request.evidenceUri, request.evalRunId ?? null, updatedAt, id),
    db.prepare("UPDATE active_learning_target SET status = ?, updated_at = ? WHERE correction_id = ?")
      .bind(targetStatus, updatedAt, current.correction_id)
  ]);
}

export async function listLearningTargets(db: D1Database, target: string | null, status: string, limit = 100): Promise<unknown[]> {
  if (target !== null && !LEARNING_TARGETS.has(target as LearningTarget)) throw new Error("Invalid learning target");
  if (!TARGET_STATUSES.has(status)) throw new Error("Invalid learning target status");
  const boundedLimit = Math.max(1, Math.min(500, limit));
  const whereTarget = target === null ? "" : " AND alt.target = ?";
  const statement = db.prepare(`SELECT alt.id, alt.correction_id, alt.target, alt.status, alt.created_at, alt.updated_at,
    sc.artifact_id, sc.pipeline_version, sc.model, sc.prompt_hash,
    al.priority, al.reviewer, al.evidence_uri, al.eval_run_id
    FROM active_learning_target alt
    JOIN student_correction sc ON sc.id = alt.correction_id
    JOIN active_learning_queue al ON al.correction_id = alt.correction_id
    WHERE alt.status = ?${whereTarget}
    ORDER BY al.priority DESC, alt.created_at LIMIT ?`);
  const rows = target === null
    ? await statement.bind(status, boundedLimit).all()
    : await statement.bind(status, target, boundedLimit).all();
  return rows.results;
}

export async function listErrorClusters(db: D1Database, limit = 100): Promise<unknown[]> {
  const rows = await db.prepare(`SELECT signature, field, reasons_json, correction_count, high_confidence_count, first_seen_at, last_seen_at
    FROM correction_error_cluster ORDER BY correction_count DESC, high_confidence_count DESC, last_seen_at DESC LIMIT ?`)
    .bind(Math.max(1, Math.min(500, limit))).all();
  return rows.results;
}

export async function calibrationSummary(db: D1Database): Promise<unknown[]> {
  const rows = await db.prepare(`SELECT confidence_bucket, COUNT(*) AS observations,
    AVG(confidence) AS mean_confidence, AVG(prediction_correct) AS empirical_accuracy,
    AVG(confidence) - AVG(prediction_correct) AS calibration_gap
    FROM confidence_calibration_observation
    GROUP BY confidence_bucket ORDER BY confidence_bucket`).all();
  return rows.results;
}
