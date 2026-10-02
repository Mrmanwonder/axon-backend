/**
 * Guardian verification provider callback (AXO-57).
 *
 * The browser starts a check with `begin_guardian_verification()` and is
 * redirected to the provider carrying the one-time `state`. The provider then
 * calls this endpoint server-to-server. Nothing in that request is trusted
 * until the provider's signature verifies; after that, the database decides
 * whose check it was (the state is bound to the guardian and session that
 * started it) and refuses replay, expiry, a different provider or a stale
 * result.
 *
 * No provider is registered until AXO-56 is decided. `hmacWebhookProvider` is
 * the generic shape most identity aggregators use (HMAC-SHA256 over
 * `timestamp.body` in a header). The concrete adapter for the chosen provider
 * replaces or wraps it; it must never be configured without its secret.
 */

export interface VerifiedCallback {
  state: string;
  reference: string;
  identity: boolean;
  adulthood: boolean;
  relationship: boolean;
  issuedAt: string;
}

export interface VerificationProvider {
  id: string;
  /** Throws on any signature, format or freshness problem. */
  verify(req: Request): Promise<VerifiedCallback>;
}

export class CallbackRejected extends Error {}

const enc = new TextEncoder();

async function hmacHex(secret: string, message: string): Promise<string> {
  const key = await crypto.subtle.importKey("raw", enc.encode(secret), { name: "HMAC", hash: "SHA-256" }, false, ["sign"]);
  const sig = await crypto.subtle.sign("HMAC", key, enc.encode(message));
  return [...new Uint8Array(sig)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

function constantTimeEqual(a: string, b: string): boolean {
  if (a.length !== b.length) return false;
  let diff = 0;
  for (let i = 0; i < a.length; i++) diff |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return diff === 0;
}

export function hmacWebhookProvider(id: string, secret: string, now: () => number = Date.now): VerificationProvider {
  if (!secret || secret.length < 32) throw new Error("verification webhook secret missing or too short");
  return {
    id,
    async verify(req: Request): Promise<VerifiedCallback> {
      const timestamp = req.headers.get("x-verification-timestamp") ?? "";
      const signature = (req.headers.get("x-verification-signature") ?? "").toLowerCase();
      const ts = Number(timestamp);
      if (!Number.isInteger(ts) || Math.abs(now() / 1000 - ts) > 300) throw new CallbackRejected("timestamp outside window");
      const body = await req.text();
      if (body.length > 8_000) throw new CallbackRejected("body too large");
      if (!constantTimeEqual(await hmacHex(secret, `${timestamp}.${body}`), signature)) throw new CallbackRejected("bad signature");
      let p: any;
      try { p = JSON.parse(body); } catch { throw new CallbackRejected("malformed body"); }
      if (typeof p?.state !== "string" || typeof p?.reference !== "string" || typeof p?.issued_at !== "string") {
        throw new CallbackRejected("missing fields");
      }
      // Absent claims are false, never assumed.
      return {
        state: p.state,
        reference: p.reference,
        identity: p.identity_verified === true,
        adulthood: p.adulthood_verified === true,
        relationship: p.relationship_verified === true,
        issuedAt: p.issued_at,
      };
    },
  };
}

export function configuredProvider(env: { GUARDIAN_VERIFICATION_PROVIDER?: string; GUARDIAN_VERIFICATION_WEBHOOK_SECRET?: string }, id: string): VerificationProvider | null {
  if (!env.GUARDIAN_VERIFICATION_PROVIDER || env.GUARDIAN_VERIFICATION_PROVIDER !== id) return null;
  if (!env.GUARDIAN_VERIFICATION_WEBHOOK_SECRET) return null;
  return hmacWebhookProvider(id, env.GUARDIAN_VERIFICATION_WEBHOOK_SECRET);
}

/** Database refusal hints → HTTP status for the provider. */
export function statusForRefusal(hint: string | undefined): number {
  switch (hint) {
    case "replayed_state": return 409;
    case "expired_state": case "stale_result": return 410;
    default: return 403;
  }
}
