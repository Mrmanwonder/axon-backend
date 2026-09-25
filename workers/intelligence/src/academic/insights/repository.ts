import { Type, type Static } from "@sinclair/typebox";
import { parseSchema } from "../../schemas";
import { detectPattern, type VerifiedObservation } from "../../intelligence/context";
import { pseudonymizeIdentifier } from "../../intelligence/security/privacy";

export const InsightObservationRequestSchema = Type.Object({
  trustedFieldId: Type.String({ minLength: 1, maxLength: 256 }),
  conceptId: Type.String({ minLength: 1, maxLength: 256 }),
  paperId: Type.String({ minLength: 1, maxLength: 256 })
}, { additionalProperties: false });
export type InsightObservationRequest = Static<typeof InsightObservationRequestSchema>;

export async function recordInsightObservation(db: D1Database, input: unknown): Promise<string> {
  const request = parseSchema(InsightObservationRequestSchema, input);
  const field = await db.prepare(`SELECT tf.field_name, tf.value_json, tf.trust_state, tf.evidence_ids_json, lr.confidence, pp.student_id, pp.paper_id
    FROM trusted_field tf
    JOIN layout_region lr ON lr.id = tf.artifact_id
    JOIN paper_page pp ON pp.page_id = lr.page_id
    WHERE tf.id = ? AND pp.paper_id = ?`)
    .bind(request.trustedFieldId, request.paperId).first<{ field_name: string; value_json: string; trust_state: string; evidence_ids_json: string; confidence: number; student_id: string; paper_id: string }>();
  if (!field) throw new Error("Trusted correctness field not found for paper");
  if (field.field_name !== "correctness") throw new Error("Insight observations require a correctness field");
  if (field.trust_state !== "AUTO_VERIFIED" && field.trust_state !== "STUDENT_VERIFIED") throw new Error("Insight observations require verified trust");
  const evidenceIds = JSON.parse(field.evidence_ids_json) as unknown;
  if (!Array.isArray(evidenceIds) || evidenceIds.length === 0) throw new Error("Insight observations require evidence");
  const correct = JSON.parse(field.value_json) as unknown;
  if (typeof correct !== "boolean") throw new Error("Correctness field must contain a boolean");
  const concept = await db.prepare("SELECT id FROM concept_taxonomy WHERE id = ?").bind(request.conceptId).first<{ id: string }>();
  if (!concept) throw new Error("Unknown concept id");
  const id = crypto.randomUUID();
  await db.prepare(`INSERT INTO insight_observation
    (id, student_id, concept_id, paper_id, correct, confidence, trust_state, evidence_ids_json, created_at, trusted_field_id)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`)
    .bind(id, field.student_id, request.conceptId, field.paper_id, correct ? 1 : 0, field.confidence, field.trust_state, JSON.stringify(evidenceIds), new Date().toISOString(), request.trustedFieldId).run();
  return id;
}

export async function readInsightPatterns(db: D1Database, sourceStudentId: string, pseudonymKey: string): Promise<Array<{ conceptId: string; pattern: ReturnType<typeof detectPattern>; correctRate: number }>> {
  const studentId = await pseudonymizeIdentifier(sourceStudentId, pseudonymKey);
  const rows = await db.prepare("SELECT concept_id, paper_id, correct, confidence, trust_state FROM insight_observation WHERE student_id = ? ORDER BY created_at")
    .bind(studentId).all<{ concept_id: string; paper_id: string; correct: number; confidence: number; trust_state: VerifiedObservation["trustState"] }>();
  const groups = new Map<string, VerifiedObservation[]>();
  for (const row of rows.results) {
    const group = groups.get(row.concept_id) ?? [];
    group.push({ conceptId: row.concept_id, paperId: row.paper_id, correct: row.correct === 1, confidence: row.confidence, trustState: row.trust_state });
    groups.set(row.concept_id, group);
  }
  return [...groups.entries()].map(([conceptId, observations]) => ({
    conceptId, pattern: detectPattern(observations), correctRate: observations.filter((item) => item.correct).length / observations.length
  }));
}
