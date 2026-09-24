import { Type } from "@sinclair/typebox";
import { parseSchema } from "../../schemas";

const ReviewSchema = Type.Object({
  status: Type.Union([Type.Literal("LABELLED"), Type.Literal("PROMOTED"), Type.Literal("REJECTED")]),
  reviewer: Type.String({ minLength: 1, maxLength: 128 }),
  evidenceUri: Type.String({ minLength: 1, maxLength: 2_000 }),
  evalRunId: Type.Optional(Type.String({ minLength: 1, maxLength: 128 }))
}, { additionalProperties: false });

const ALLOWED: Readonly<Record<string, readonly string[]>> = {
  QUEUED: ["LABELLED", "REJECTED"], LABELLED: ["PROMOTED", "REJECTED"], PROMOTED: [], REJECTED: []
};

export async function listActiveLearning(db: D1Database, status = "QUEUED", limit = 50): Promise<unknown[]> {
  if (!["QUEUED", "LABELLED", "PROMOTED", "REJECTED"].includes(status)) throw new Error("Invalid active-learning status");
  const boundedLimit = Math.max(1, Math.min(200, limit));
  const rows = await db.prepare(`SELECT al.id, al.correction_id, al.priority, al.reasons_json, al.status, al.reviewer, al.evidence_uri, al.eval_run_id, al.created_at, al.updated_at,
    sc.field, sc.artifact_id, sc.pipeline_version, sc.model, sc.prompt_hash
    FROM active_learning_queue al JOIN student_correction sc ON sc.id = al.correction_id
    WHERE al.status = ? ORDER BY al.priority DESC, al.created_at LIMIT ?`).bind(status, boundedLimit).all();
  return rows.results;
}

export async function reviewActiveLearning(db: D1Database, id: string, input: unknown): Promise<void> {
  const request = parseSchema(ReviewSchema, input);
  const current = await db.prepare("SELECT status FROM active_learning_queue WHERE id = ?").bind(id).first<{ status: string }>();
  if (!current) throw new Error("Active-learning item not found");
  if (!(ALLOWED[current.status] ?? []).includes(request.status)) throw new Error(`Invalid active-learning transition ${current.status} -> ${request.status}`);
  if (request.status === "PROMOTED") {
    if (!request.evalRunId) throw new Error("Promotion requires a passing evaluation run");
    const run = await db.prepare("SELECT passed FROM eval_run WHERE id = ?").bind(request.evalRunId).first<{ passed: number }>();
    if (run?.passed !== 1) throw new Error("Promotion requires a passing evaluation run");
  }
  await db.prepare("UPDATE active_learning_queue SET status = ?, reviewer = ?, evidence_uri = ?, eval_run_id = ?, updated_at = ? WHERE id = ?")
    .bind(request.status, request.reviewer, request.evidenceUri, request.evalRunId ?? null, new Date().toISOString(), id).run();
}
