/**
 * AXO-126 deletion parity. The database queues a purge for every deleted paper and every erased
 * student's papers (tutor_purge). This carries a claimed batch to axon-intelligence, which deletes
 * the Tutor's provenance rows for those papers, then stamps each row. A failed call records why
 * and leaves the rows to be claimed again; nothing is marked done unless the Tutor said so.
 * Carries paper ids only.
 */
export interface PurgeClaim {
  id: number;
  paper_id: string;
}

export interface PurgeSink {
  claim(limit: number): Promise<PurgeClaim[]>;
  finish(id: number, error: string | null): Promise<void>;
}

export async function runTutorPurges(
  sink: PurgeSink,
  call: ((paperIds: string[]) => Promise<{ ok: boolean; status: number }>) | undefined,
  limit = 20,
): Promise<number> {
  // With no Tutor to call, claim nothing: the rows wait, with their attempt counts untouched.
  if (!call) return 0;
  const claims = await sink.claim(limit);
  if (claims.length === 0) return 0;
  let error: string | null = null;
  try {
    const res = await call(claims.map((c) => c.paper_id));
    if (!res.ok) error = `tutor purge returned ${res.status}`;
  } catch (cause) {
    error = String(cause).slice(0, 200);
  }
  for (const claim of claims) await sink.finish(claim.id, error);
  return error === null ? claims.length : 0;
}
