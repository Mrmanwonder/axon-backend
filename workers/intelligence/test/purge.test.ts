import { describe, expect, it } from "vitest";
import { env, exports } from "cloudflare:workers";
import { parsePurgeRequest } from "../src/intelligence/security/purge";

const PAPER_A = "aaaaaaaa-0000-4000-8000-0000000000a1";
const PAPER_B = "aaaaaaaa-0000-4000-8000-0000000000a2";

async function seedTrace(traceId: string, paperId: string | null) {
  await env.DB.batch([
    env.DB.prepare("INSERT INTO ai_trace (trace_id, paper_id, stage, capability, deployment_sha, config_revision, pipeline_version, verification_status) VALUES (?, ?, 'tutor', 'tutor', 'sha', 'rev', '3.0.0', 'verified')").bind(traceId, paperId),
    env.DB.prepare("INSERT INTO evidence (id, trace_id, information_class, source, authority, value_json, provenance_json, verification) VALUES (?, ?, 'x', 'x', 'x', '{}', '{}', 'verified')").bind(`ev-${traceId}`, traceId),
    env.DB.prepare("INSERT INTO claim (id, trace_id, text, type, risk, verification_status) VALUES (?, ?, 't', 'x', 'low', 'verified')").bind(`cl-${traceId}`, traceId),
    env.DB.prepare("INSERT INTO claim_evidence (claim_id, evidence_id) VALUES (?, ?)").bind(`cl-${traceId}`, `ev-${traceId}`),
  ]);
}

const count = async (table: string) =>
  (await env.DB.prepare(`SELECT COUNT(*) AS n FROM ${table}`).first<{ n: number }>())!.n;

const purge = (body: unknown, token = "test-admin-token") =>
  exports.default.fetch(new Request("https://axon.test/v1/admin/purge", {
    method: "POST",
    headers: { "content-type": "application/json", authorization: `Bearer ${token}` },
    body: JSON.stringify(body),
  }));

describe("Tutor deletion parity", () => {
  it("removes a paper's trace and everything hanging off it, and nothing else", async () => {
    await seedTrace("trace-a", PAPER_A);
    await seedTrace("trace-b", PAPER_B);
    await seedTrace("trace-none", null);

    const response = await purge({ paperIds: [PAPER_A] });
    expect(response.status).toBe(200);
    await expect(response.json()).resolves.toEqual({ paperIds: 1, traces: 1 });

    const remaining = await env.DB.prepare("SELECT trace_id FROM ai_trace ORDER BY trace_id").all<{ trace_id: string }>();
    expect(remaining.results.map((r) => r.trace_id)).toEqual(["trace-b", "trace-none"]);
    expect(await count("evidence")).toBe(2);
    expect(await count("claim")).toBe(2);
    expect(await count("claim_evidence")).toBe(2);
  });

  it("is idempotent: purging a paper with no rows succeeds and changes nothing", async () => {
    const before = await count("ai_trace");
    const response = await purge({ paperIds: [PAPER_A] });
    expect(response.status).toBe(200);
    await expect(response.json()).resolves.toEqual({ paperIds: 1, traces: 0 });
    expect(await count("ai_trace")).toBe(before);
  });

  it("is admin-only: the internal Tutor token cannot purge", async () => {
    expect((await purge({ paperIds: [PAPER_B] }, "test-token")).status).toBe(401);
    expect(await count("ai_trace")).toBeGreaterThan(0);
  });

  it("rejects malformed requests before touching data", async () => {
    expect(() => parsePurgeRequest({})).toThrow(/non-empty array/);
    expect(() => parsePurgeRequest({ paperIds: [] })).toThrow(/non-empty array/);
    expect(() => parsePurgeRequest({ paperIds: ["not-a-uuid"] })).toThrow(/UUIDs/);
    expect(() => parsePurgeRequest({ paperIds: Array.from({ length: 51 }, () => PAPER_A) })).toThrow(/at most 50/);
    expect(parsePurgeRequest({ paperIds: [PAPER_A, PAPER_A.toUpperCase()] })).toEqual([PAPER_A]);
    expect((await purge({ paperIds: ["nope"] })).status).toBe(400);
  });
});
