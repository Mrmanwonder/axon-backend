/**
 * Tutor staged rollout (AXO-126).
 *
 *   off       — nobody (the default; an unset variable is off)
 *   internal  — only the auth users listed in TUTOR_INTERNAL_USERS
 *   ga        — every authenticated Student Mode session
 *
 * The beta stage (opt-in Student Mode users) is deliberately not a value yet:
 * it is gated on the 60-case AXO-40 golden set passing AXO-44 thresholds, and
 * shipping the switch before the evidence would make it one config edit away.
 *
 * Rollback is the same switch: set TUTOR_ROLLOUT=off (or remove the
 * AXON_INTERNAL_TOKEN secret, which also returns 503) — no code change.
 *
 * This is a release gate, not an authority boundary. It runs only after the
 * session's Student Mode scope was verified by the database, so the `sub` it
 * reads belongs to a JWT that PostgREST has already accepted.
 */

export type RolloutDecision = { allowed: true } | { allowed: false; reason: "off" | "not_in_stage" };

export function jwtSubject(authorization: string | null): string | null {
  const token = authorization?.match(/^Bearer\s+(.+)$/i)?.[1];
  const payload = token?.split(".")[1];
  if (!payload) return null;
  try {
    const json = JSON.parse(atob(payload.replace(/-/g, "+").replace(/_/g, "/")));
    return typeof json?.sub === "string" ? json.sub : null;
  } catch {
    return null;
  }
}

export function tutorRollout(env: { TUTOR_ROLLOUT?: string; TUTOR_INTERNAL_USERS?: string }, subject: string | null): RolloutDecision {
  const stage = (env.TUTOR_ROLLOUT ?? "off").trim().toLowerCase();
  if (stage === "ga") return { allowed: true };
  if (stage === "internal") {
    const users = new Set((env.TUTOR_INTERNAL_USERS ?? "").split(",").map((u) => u.trim()).filter(Boolean));
    return subject && users.has(subject) ? { allowed: true } : { allowed: false, reason: "not_in_stage" };
  }
  return { allowed: false, reason: "off" };
}
