/**
 * Deletion parity (AXO-126). When a paper is deleted or a student erased, the Tutor's own
 * provenance rows for that paper must go too. Tutor provenance is keyed by paper_id on ai_trace,
 * with claim, evidence and claim_evidence hanging off the trace. A Tutor question asked with no
 * paper carries no student link at all, so there is nothing to purge for it.
 *
 * Idempotent: purging a paper with no rows is a no-op, so the caller can safely retry.
 */
export interface PurgeResult {
  paperIds: number;
  traces: number;
}

const MAX_PAPERS = 50;
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

export function parsePurgeRequest(body: unknown): string[] {
  const ids = (body as { paperIds?: unknown } | null)?.paperIds;
  if (!Array.isArray(ids) || ids.length === 0) throw new Error("paperIds must be a non-empty array");
  if (ids.length > MAX_PAPERS) throw new Error(`paperIds must be at most ${MAX_PAPERS}`);
  for (const id of ids) {
    if (typeof id !== "string" || !UUID.test(id)) throw new Error("paperIds must be UUIDs");
  }
  return [...new Set(ids.map((id) => (id as string).toLowerCase()))];
}

export async function purgeTutorDataForPapers(db: D1Database, paperIds: string[]): Promise<PurgeResult> {
  let traces = 0;
  for (const paperId of paperIds) {
    const results = await db.batch([
      db.prepare("DELETE FROM claim_evidence WHERE claim_id IN (SELECT c.id FROM claim c JOIN ai_trace t ON t.trace_id = c.trace_id WHERE t.paper_id = ?1) OR evidence_id IN (SELECT e.id FROM evidence e JOIN ai_trace t ON t.trace_id = e.trace_id WHERE t.paper_id = ?1)").bind(paperId),
      db.prepare("DELETE FROM claim WHERE trace_id IN (SELECT trace_id FROM ai_trace WHERE paper_id = ?1)").bind(paperId),
      db.prepare("DELETE FROM evidence WHERE trace_id IN (SELECT trace_id FROM ai_trace WHERE paper_id = ?1)").bind(paperId),
      db.prepare("DELETE FROM ai_trace WHERE paper_id = ?1").bind(paperId),
    ]);
    traces += results[3]?.meta.changes ?? 0;
  }
  return { paperIds: paperIds.length, traces };
}
