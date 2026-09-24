import { oneWayHash } from "./privacy";

export interface IdempotentResult { status: number; payload: unknown }

function canonical(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`;
  if (value && typeof value === "object") {
    const record = value as Record<string, unknown>;
    return `{${Object.keys(record).sort().map((key) => `${JSON.stringify(key)}:${canonical(record[key])}`).join(",")}}`;
  }
  return JSON.stringify(value) ?? "null";
}

export async function lookupIdempotentResult(db: D1Database, request: Request, route: string, body: unknown): Promise<{ keyHash?: string; requestHash: string; cached?: IdempotentResult }> {
  const requestHash = await oneWayHash(canonical(body));
  const key = request.headers.get("idempotency-key");
  if (!key) return { requestHash };
  if (key.length < 8 || key.length > 256) throw new Error("Idempotency-Key must be between 8 and 256 characters");
  const keyHash = await oneWayHash(key);
  const row = await db.prepare("SELECT request_hash, status_code, response_json, expires_at FROM request_idempotency WHERE idempotency_key_hash = ? AND route = ?")
    .bind(keyHash, route).first<{ request_hash: string; status_code: number; response_json: string; expires_at: string }>();
  if (!row || Date.parse(row.expires_at) <= Date.now()) return { keyHash, requestHash };
  if (row.request_hash !== requestHash) throw new Error("Idempotency-Key was already used for a different request");
  return { keyHash, requestHash, cached: { status: row.status_code, payload: JSON.parse(row.response_json) as unknown } };
}

export async function storeIdempotentResult(db: D1Database, route: string, lookup: { keyHash?: string; requestHash: string }, result: IdempotentResult): Promise<void> {
  if (!lookup.keyHash) return;
  const createdAt = new Date();
  const expiresAt = new Date(createdAt.getTime() + 24 * 60 * 60 * 1_000);
  await db.prepare("INSERT OR REPLACE INTO request_idempotency (idempotency_key_hash, route, request_hash, status_code, response_json, created_at, expires_at) VALUES (?, ?, ?, ?, ?, ?, ?)")
    .bind(lookup.keyHash, route, lookup.requestHash, result.status, JSON.stringify(result.payload), createdAt.toISOString(), expiresAt.toISOString()).run();
}
