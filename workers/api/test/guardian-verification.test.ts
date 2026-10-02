import { beforeEach, describe, expect, test, vi } from "vitest";

/* AXO-57: the provider callback trusts nothing before the signature verifies,
   and leaves binding/replay/expiry to the database. Synthetic payloads. */

const fixture = vi.hoisted(() => ({ rpc: vi.fn() }));

vi.mock("@mastery/shared/http.js", () => {
  const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status });
  return {
    CORS: {}, corsFor: () => ({}), withCors: (_r: Request, res: Response) => res, json,
    failure: (message: string, status = 400) => json({ error: message }, status),
    clientFor: vi.fn(), serviceClient: () => ({ rpc: fixture.rpc }),
    readJson: async (req: Request) => req.json(),
  };
});
vi.mock("@mastery/shared/r2.js", () => ({ presignPut: vi.fn(), headObject: vi.fn(), signAssetUrl: vi.fn(), verifyAssetSignature: vi.fn(), objectKey: vi.fn(), BUCKET_FOR: {} }));
vi.mock("@mastery/shared/contract.js", () => ({ CAPTURE: { MAX_PAGES: 60 }, PIPELINE_VERSION: "test", SAFE_OBJECT_NAME: /^[a-z]+$/ }));

import worker from "../src/index.js";
import { hmacWebhookProvider, statusForRefusal } from "../src/guardian_verification.js";

const SECRET = "s".repeat(48);
const env = { GUARDIAN_VERIFICATION_PROVIDER: "digilocker", GUARDIAN_VERIFICATION_WEBHOOK_SECRET: SECRET } as any;

async function sign(ts: string, body: string, secret = SECRET) {
  const key = await crypto.subtle.importKey("raw", new TextEncoder().encode(secret), { name: "HMAC", hash: "SHA-256" }, false, ["sign"]);
  const sig = await crypto.subtle.sign("HMAC", key, new TextEncoder().encode(`${ts}.${body}`));
  return [...new Uint8Array(sig)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

const payload = { state: "st", reference: "txn-1", identity_verified: true, adulthood_verified: true, relationship_verified: true, issued_at: new Date().toISOString() };

async function callback(opts: { provider?: string; body?: string; ts?: number; secret?: string; tamper?: boolean } = {}) {
  const body = opts.body ?? JSON.stringify(payload);
  const ts = String(opts.ts ?? Math.floor(Date.now() / 1000));
  const sig = await sign(ts, body, opts.secret);
  return worker.fetch(new Request(`https://api.test/guardian-verification/callback/${opts.provider ?? "digilocker"}`, {
    method: "POST",
    headers: { "x-verification-timestamp": ts, "x-verification-signature": sig, "content-type": "application/json" },
    body: opts.tamper ? body.replace("txn-1", "txn-9") : body,
  }), env);
}

beforeEach(() => { vi.clearAllMocks(); fixture.rpc.mockResolvedValue({ data: "assertion-1", error: null }); });

describe("AXO-57 provider callback", () => {
  test("a validly signed callback is recorded through the service-role RPC with the parsed claims", async () => {
    const res = await callback();
    expect(res.status).toBe(200);
    expect(fixture.rpc).toHaveBeenCalledWith("record_guardian_verification_callback", expect.objectContaining({
      p_state: "st", p_provider: "digilocker", p_reference: "txn-1", p_identity: true, p_adulthood: true, p_relationship: true,
    }));
  });

  test("a tampered body is rejected before the database is touched", async () => {
    expect((await callback({ tamper: true })).status).toBe(401);
    expect(fixture.rpc).not.toHaveBeenCalled();
  });

  test("a signature made with another secret is rejected", async () => {
    expect((await callback({ secret: "x".repeat(48) })).status).toBe(401);
    expect(fixture.rpc).not.toHaveBeenCalled();
  });

  test("an old signed request (outside 5 minutes) is rejected", async () => {
    expect((await callback({ ts: Math.floor(Date.now() / 1000) - 3600 })).status).toBe(401);
    expect(fixture.rpc).not.toHaveBeenCalled();
  });

  test("a provider that is not configured does not exist", async () => {
    expect((await callback({ provider: "stub" })).status).toBe(404);
    expect((await worker.fetch(new Request("https://api.test/guardian-verification/callback/digilocker", { method: "POST", body: "{}" }), {} as any)).status).toBe(404);
  });

  test("absent claims are false, never assumed", async () => {
    await callback({ body: JSON.stringify({ state: "st", reference: "txn-2", issued_at: new Date().toISOString() }) });
    expect(fixture.rpc).toHaveBeenCalledWith("record_guardian_verification_callback", expect.objectContaining({
      p_identity: false, p_adulthood: false, p_relationship: false,
    }));
  });

  test("database refusals map to provider-facing statuses", async () => {
    fixture.rpc.mockResolvedValueOnce({ data: null, error: { code: "42501", hint: "replayed_state" } });
    expect((await callback()).status).toBe(409);
    fixture.rpc.mockResolvedValueOnce({ data: null, error: { code: "23505" } });
    expect((await callback()).status).toBe(409);
    expect(statusForRefusal("expired_state")).toBe(410);
    expect(statusForRefusal("stale_result")).toBe(410);
    expect(statusForRefusal("wrong_provider")).toBe(403);
  });

  test("a provider cannot be configured with a weak secret", () => {
    expect(() => hmacWebhookProvider("digilocker", "short")).toThrow();
  });
});
