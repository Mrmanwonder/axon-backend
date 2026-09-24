import { describe, expect, it } from "vitest";
import { env, exports } from "cloudflare:workers";
import { pseudonymizeIdentifier } from "../src/intelligence/security/privacy";

describe("Worker", () => {
  it("reports immutable deployment provenance", async () => {
    const response = await exports.default.fetch(new Request("https://axon.test/health"));
    expect(response.status).toBe(200);
    await expect(response.json()).resolves.toMatchObject({ status: "ok", pipelineVersion: "3.0.0", configRevision: "v3.default" });
  });
  it("captures corrections without overwriting prediction history", async () => {
    const correction = {
      field: "marksAwarded", predicted: 2, corrected: 3, acceptedValue: 3,
      artifactId: "paper:q1", pipelineVersion: "3.0.0", model: "test-model",
      promptHash: "abc", contextMetadata: { pageId: "p1" }
    };
    const response = await exports.default.fetch(new Request("https://axon.test/v1/corrections", {
      method: "POST", headers: { "content-type": "application/json", authorization: "Bearer test-token" }, body: JSON.stringify(correction)
    }));
    expect(response.status).toBe(201);
    const row = await env.DB.prepare("SELECT predicted_json, corrected_json, accepted_json FROM student_correction WHERE artifact_id = ?").bind("paper:q1").first<{ predicted_json: string; corrected_json: string; accepted_json: string }>();
    expect(row).toEqual({ predicted_json: "2", corrected_json: "3", accepted_json: "3" });
    const learning = await env.DB.prepare("SELECT priority, status FROM active_learning_queue WHERE correction_id IN (SELECT id FROM student_correction WHERE artifact_id = ?)").bind("paper:q1").first<{ priority: number; status: string }>();
    expect(learning).toEqual({ priority: 0.65, status: "QUEUED" });
  });
  it("replays correction results idempotently", async () => {
    const correction = {
      field: "answer", predicted: "x", corrected: "y", acceptedValue: "y", artifactId: `artifact:${crypto.randomUUID()}`,
      pipelineVersion: "3.0.0", model: "test-model", promptHash: "hash", contextMetadata: {}
    };
    const request = () => new Request("https://axon.test/v1/corrections", {
      method: "POST", headers: { "content-type": "application/json", authorization: "Bearer test-token", "idempotency-key": "same-correction-request" }, body: JSON.stringify(correction)
    });
    const first = await exports.default.fetch(request());
    const second = await exports.default.fetch(request());
    expect(await second.json()).toEqual(await first.json());
    const count = await env.DB.prepare("SELECT COUNT(*) AS count FROM student_correction WHERE artifact_id = ?").bind(correction.artifactId).first<{ count: number }>();
    expect(count?.count).toBe(1);
  });
  it("deduplicates immutable paper pages before inference", async () => {
    const body = new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 1, 2, 3]);
    const request = () => new Request("https://axon.test/v1/papers/ingest", {
      method: "POST",
      headers: { "content-type": "image/png", "content-length": String(body.byteLength), "x-axon-student-id": "student-1", "x-axon-paper-id": "paper-1", authorization: "Bearer test-token" },
      body
    });
    const first = await exports.default.fetch(request());
    const second = await exports.default.fetch(request());
    expect(first.status).toBe(202);
    expect(second.status).toBe(202);
    await expect(second.json()).resolves.toMatchObject({ duplicate: true });
    const pseudonym = await pseudonymizeIdentifier("student-1", "test-pseudonym-key");
    const count = await env.DB.prepare("SELECT COUNT(*) AS count FROM paper_page WHERE student_id = ?").bind(pseudonym).first<{ count: number }>();
    expect(count?.count).toBe(1);
  });
  it("rejects unauthenticated student-data requests", async () => {
    const response = await exports.default.fetch(new Request("https://axon.test/v1/corrections", { method: "POST", body: "{}" }));
    expect(response.status).toBe(401);
  });
  it("reports fail-closed production readiness until external evidence exists", async () => {
    const response = await exports.default.fetch(new Request("https://axon.test/v1/admin/readiness", { headers: { authorization: "Bearer test-admin-token" } }));
    expect(response.status).toBe(503);
    await expect(response.json()).resolves.toMatchObject({ status: "not_ready", checks: { geminiZdr: false, releaseCertified: false } });
  });
  it("does not allow ordinary callers to use admin endpoints", async () => {
    const response = await exports.default.fetch(new Request("https://axon.test/v1/admin/provider-health", { headers: { authorization: "Bearer test-token" } }));
    expect(response.status).toBe(401);
  });
  it("fails closed and records telemetry when model privacy is unattested", async () => {
    const response = await exports.default.fetch(new Request("https://axon.test/v1/tutor", {
      method: "POST",
      headers: { "content-type": "application/json", authorization: "Bearer test-token" },
      body: JSON.stringify({ studentId: "student-private", message: "Explain mitosis" })
    }));
    expect(response.status).toBe(200);
    const payload = await response.json<{ traceId: string; status: string; answer: string }>();
    expect(payload.status).toBe("controlled_failure");
    expect(payload.answer).toContain("data was not sent");
    const trace = await env.DB.prepare("SELECT verification_status, error FROM ai_trace WHERE trace_id = ?").bind(payload.traceId).first<{ verification_status: string; error: string }>();
    expect(trace).toEqual({ verification_status: "controlled_failure", error: "NO_COMPLIANT_PROVIDER" });
  });
});
