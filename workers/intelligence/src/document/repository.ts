import { Type, type Static } from "@sinclair/typebox";
import { parseSchema } from "../schemas";

const ReviewDecisionSchema = Type.Object({
  fieldId: Type.String({ minLength: 1, maxLength: 256 }), value: Type.Unknown(),
  evidenceIds: Type.Array(Type.String({ minLength: 1, maxLength: 256 }), { minItems: 1, maxItems: 20 })
}, { additionalProperties: false });

export const PaperReviewRequestSchema = Type.Object({
  decisions: Type.Array(ReviewDecisionSchema, { maxItems: 2_000 }),
  resolvedReasons: Type.Array(Type.String({ minLength: 1, maxLength: 128 }), { maxItems: 50 })
}, { additionalProperties: false });
export type PaperReviewRequest = Static<typeof PaperReviewRequestSchema>;

export async function getPaperPage(db: D1Database, pageId: string): Promise<Record<string, unknown> | null> {
  const page = await db.prepare("SELECT * FROM paper_page WHERE page_id = ?").bind(pageId).first<Record<string, unknown>>();
  if (!page) return null;
  const [regions, marks, reads, trustedFields, events] = await Promise.all([
    db.prepare("SELECT * FROM layout_region WHERE page_id = ? ORDER BY id").bind(pageId).all(),
    db.prepare("SELECT * FROM mark_assignment WHERE mark_region_id IN (SELECT id FROM layout_region WHERE page_id = ?) ORDER BY mark_region_id").bind(pageId).all(),
    db.prepare("SELECT * FROM recognition_read WHERE region_id IN (SELECT id FROM layout_region WHERE page_id = ?) ORDER BY region_id").bind(pageId).all(),
    db.prepare("SELECT * FROM trusted_field WHERE artifact_id IN (SELECT id FROM layout_region WHERE page_id = ?) ORDER BY id").bind(pageId).all(),
    db.prepare("SELECT stage, status, input_hash, output_hash, details_json, created_at FROM paper_stage_event WHERE page_id = ? ORDER BY created_at").bind(pageId).all()
  ]);
  return { page, regions: regions.results, marks: marks.results, reads: reads.results, trustedFields: trustedFields.results, events: events.results };
}

export async function reviewPaperPage(db: D1Database, pageId: string, input: unknown): Promise<{ updated: number; remainingReasons: string[] }> {
  const request = parseSchema(PaperReviewRequestSchema, input);
  const page = await db.prepare("SELECT processing_state, review_reasons_json FROM paper_page WHERE page_id = ?").bind(pageId).first<{ processing_state: string; review_reasons_json: string }>();
  if (!page) throw new Error("Paper page not found");
  if (page.processing_state !== "REVIEW_REQUIRED") throw new Error("Paper page is not awaiting review");
  const updates: D1PreparedStatement[] = [];
  for (const decision of request.decisions) {
    const existing = await db.prepare("SELECT id FROM trusted_field WHERE id = ? AND artifact_id IN (SELECT id FROM layout_region WHERE page_id = ?)")
      .bind(decision.fieldId, pageId).first<{ id: string }>();
    if (!existing) throw new Error(`Invalid trusted field for page: ${decision.fieldId}`);
    updates.push(db.prepare("UPDATE trusted_field SET value_json = ?, trust_state = 'STUDENT_VERIFIED', evidence_ids_json = ?, updated_at = ? WHERE id = ?")
      .bind(JSON.stringify(decision.value), JSON.stringify(decision.evidenceIds), new Date().toISOString(), decision.fieldId));
  }
  if (updates.length > 0) await db.batch(updates);
  const reasons = JSON.parse(page.review_reasons_json) as unknown;
  const current = Array.isArray(reasons) ? reasons.filter((reason): reason is string => typeof reason === "string") : [];
  const resolved = new Set(request.resolvedReasons);
  const remainingReasons = current.filter((reason) => !resolved.has(reason));
  await db.prepare("UPDATE paper_page SET review_reasons_json = ?, updated_at = ? WHERE page_id = ?")
    .bind(JSON.stringify(remainingReasons), new Date().toISOString(), pageId).run();
  return { updated: updates.length, remainingReasons };
}

export async function commitReviewedPaperPage(db: D1Database, pageId: string): Promise<void> {
  const page = await db.prepare("SELECT processing_state, review_reasons_json, original_hash FROM paper_page WHERE page_id = ?")
    .bind(pageId).first<{ processing_state: string; review_reasons_json: string; original_hash: string }>();
  if (!page) throw new Error("Paper page not found");
  if (page.processing_state !== "REVIEW_REQUIRED") throw new Error("Paper page is not awaiting review");
  const reasons = JSON.parse(page.review_reasons_json) as unknown;
  if (Array.isArray(reasons) && reasons.length > 0) throw new Error("Paper page has unresolved review reasons");
  const counts = await db.prepare(`SELECT COUNT(*) AS total,
    SUM(CASE WHEN trust_state IN ('AUTO_VERIFIED','STUDENT_VERIFIED') AND json_array_length(evidence_ids_json) > 0 THEN 1 ELSE 0 END) AS trusted
    FROM trusted_field WHERE artifact_id IN (SELECT id FROM layout_region WHERE page_id = ?)`)
    .bind(pageId).first<{ total: number; trusted: number | null }>();
  if (!counts || counts.total === 0 || counts.trusted !== counts.total) throw new Error("Paper page has unresolved trusted fields");
  const now = new Date().toISOString();
  await db.batch([
    db.prepare("UPDATE paper_page SET processing_state = 'TRUSTED_COMMIT', updated_at = ? WHERE page_id = ?").bind(now, pageId),
    db.prepare("INSERT INTO paper_stage_event (id, page_id, stage, status, input_hash, details_json, created_at) VALUES (?, ?, 'STUDENT_REVIEW', 'COMPLETED', ?, ?, ?)")
      .bind(crypto.randomUUID(), pageId, page.original_hash, JSON.stringify({ trustedFields: counts.total }), now)
  ]);
}
