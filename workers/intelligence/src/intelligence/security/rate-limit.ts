import { oneWayHash } from "./privacy";

export async function enforceRateLimit(db: D1Database, request: Request, route: string, limit: number, now = Date.now()): Promise<{ allowed: boolean; remaining: number; retryAfterSeconds: number }> {
  const principal = request.headers.get("x-axon-client-id") ?? request.headers.get("x-axon-student-id") ?? request.headers.get("cf-connecting-ip") ?? "internal-client";
  const identityHash = await oneWayHash(principal);
  const windowStart = Math.floor(now / 60_000) * 60_000;
  const row = await db.prepare(`INSERT INTO api_rate_window (identity_hash, route, window_start, request_count)
    VALUES (?, ?, ?, 1)
    ON CONFLICT(identity_hash, route, window_start) DO UPDATE SET request_count = request_count + 1
    RETURNING request_count`)
    .bind(identityHash, route, windowStart).first<{ request_count: number }>();
  const count = row?.request_count ?? limit + 1;
  return { allowed: count <= limit, remaining: Math.max(0, limit - count), retryAfterSeconds: Math.max(1, Math.ceil((windowStart + 60_000 - now) / 1_000)) };
}
