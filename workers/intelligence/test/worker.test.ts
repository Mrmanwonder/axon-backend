import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { env, exports } from "cloudflare:workers";
import { pseudonymizeIdentifier } from "../src/intelligence/security/privacy";
import { mockSupabaseConsent, TEST_STUDENT_ID } from "./supabase-consent.mock";

describe("Worker", () => {
  beforeEach(() => { mockSupabaseConsent(); });
  afterEach(() => { vi.restoreAllMocks(); });

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
      method: "POST", headers: { "content-type": "application/json", authorization: "Bearer test-token", "x-axon-student-id": TEST_STUDENT_ID }, body: JSON.stringify(correction)
    }));
    expect(response.status).toBe(201);
    const row = await env.DB.prepare("SELECT predicted_json, corrected_json, accepted_json FROM student_correction WHERE artifact_id = ?").bind("paper:q1").first<{ predicted_json: string; corrected_json: string; accepted_json: string }>();
    expect(row).toEqual({ predicted_json: "2", corrected_json: "3", accepted_json: "3" });
    const learning = await env.DB.prepare("SELECT priority, status FROM active_learning_queue WHERE correction_id IN (SELECT id FROM student_correction WHERE artifact_id = ?)").bind("paper:q1").first<{ priority: number; status: string }>();
    expect(learning).toEqual({ priority: 0.65, status: "QUEUED" });
    const targets = await env.DB.prepare("SELECT target FROM active_learning_target WHERE correction_id IN (SELECT id FROM student_correction WHERE artifact_id = ?) ORDER BY target").bind("paper:q1").all<{ target: string }>();
    expect(targets.results.map((item) => item.target)).toEqual(["BENCHMARK_EXPANSION", "ERROR_CLUSTERING", "HTR_DATASET", "LAYOUT_TRAINING", "PROMPT_REGRESSION"]);
    const cluster = await env.DB.prepare("SELECT field, correction_count, high_confidence_count FROM correction_error_cluster").first<{ field: string; correction_count: number; high_confidence_count: number }>();
    expect(cluster).toEqual({ field: "mark", correction_count: 1, high_confidence_count: 0 });
    const provenance = await env.DB.prepare("SELECT student_id, learning_consent_granted, learning_consent_seq, learning_consent_notice_version FROM student_correction WHERE artifact_id = ?").bind("paper:q1").first<Record<string, unknown>>();
    expect(provenance).toEqual({
      student_id: await pseudonymizeIdentifier(TEST_STUDENT_ID, "test-pseudonym-key"),
      learning_consent_granted: 1,
      learning_consent_seq: 42,
      learning_consent_notice_version: "privacy.v1"
    });
  });

  it("preserves a correction but creates no learning data when optional consent is denied", async () => {
    vi.restoreAllMocks();
    mockSupabaseConsent(false);
    const artifactId = `paper:${crypto.randomUUID()}`;
    const response = await exports.default.fetch(new Request("https://axon.test/v1/corrections", {
      method: "POST",
      headers: { "content-type": "application/json", authorization: "Bearer test-token", "x-axon-student-id": TEST_STUDENT_ID },
      body: JSON.stringify({
        field: "recognized_text", predicted: "private prediction", corrected: "private correction", acceptedValue: "private correction",
        artifactId, pipelineVersion: "3.0.0", model: "vision", promptHash: "prompt-hash", contextMetadata: { confidence: 0.99 }
      })
    }));
    expect(response.status).toBe(201);
    await expect(response.json()).resolves.toMatchObject({ accepted: true, activeLearningId: null, learningConsent: "DENIED" });
    const correction = await env.DB.prepare("SELECT learning_consent_granted FROM student_correction WHERE artifact_id = ?").bind(artifactId).first<{ learning_consent_granted: number }>();
    expect(correction).toEqual({ learning_consent_granted: 0 });
    const queued = await env.DB.prepare("SELECT COUNT(*) AS count FROM active_learning_queue WHERE correction_id IN (SELECT id FROM student_correction WHERE artifact_id = ?)").bind(artifactId).first<{ count: number }>();
    expect(queued?.count).toBe(0);
  });

  it("routes correction evidence into calibration and private learning ledgers without copying values", async () => {
    const artifactId = `paper:${crypto.randomUUID()}`;
    const correction = {
      field: "recognized_text", predicted: "x = 7", corrected: "x = 1", acceptedValue: "x = 1",
      artifactId, pipelineVersion: "3.0.0", model: "vision", promptHash: "prompt-hash",
      contextMetadata: { confidence: 0.92, layer: "STUDENT", regionClass: "student_answer", stage: "document_reading" }
    };
    const response = await exports.default.fetch(new Request("https://axon.test/v1/corrections", {
      method: "POST", headers: { "content-type": "application/json", authorization: "Bearer test-token", "x-axon-student-id": TEST_STUDENT_ID }, body: JSON.stringify(correction)
    }));
    expect(response.status).toBe(201);
    const targetResponse = await exports.default.fetch(new Request("https://axon.test/v1/admin/learning/targets?status=QUEUED", { headers: { authorization: "Bearer test-admin-token" } }));
    const targetPayload = await targetResponse.json<{ items: Array<Record<string, unknown>> }>();
    const relevantTargets = targetPayload.items.filter((item) => item["artifact_id"] === artifactId).map((item) => item["target"]).sort();
    expect(relevantTargets).toEqual(["BENCHMARK_EXPANSION", "CONFIDENCE_RECALIBRATION", "ERROR_CLUSTERING", "HTR_DATASET", "PROMPT_REGRESSION"]);
    expect(JSON.stringify(targetPayload)).not.toContain("x = 7");
    expect(JSON.stringify(targetPayload)).not.toContain("x = 1");
    const calibration = await exports.default.fetch(new Request("https://axon.test/v1/admin/learning/calibration", { headers: { authorization: "Bearer test-admin-token" } }));
    await expect(calibration.json()).resolves.toMatchObject({ buckets: [{ confidence_bucket: 9, observations: 1, mean_confidence: 0.92, empirical_accuracy: 0 }] });
    const clusters = await exports.default.fetch(new Request("https://axon.test/v1/admin/learning/error-clusters", { headers: { authorization: "Bearer test-admin-token" } }));
    const clusterPayload = await clusters.json<{ clusters: Array<Record<string, unknown>> }>();
    expect(clusterPayload.clusters.find((item) => item["field"] === "recognized_text")).toMatchObject({ field: "recognized_text", correction_count: 1, high_confidence_count: 1 });
    expect(JSON.stringify(clusterPayload)).not.toContain("x = 7");
    expect(JSON.stringify(clusterPayload)).not.toContain("x = 1");
  });
  it("replays correction results idempotently", async () => {
    const correction = {
      field: "answer", predicted: "x", corrected: "y", acceptedValue: "y", artifactId: `artifact:${crypto.randomUUID()}`,
      pipelineVersion: "3.0.0", model: "test-model", promptHash: "hash", contextMetadata: {}
    };
    const request = () => new Request("https://axon.test/v1/corrections", {
      method: "POST", headers: { "content-type": "application/json", authorization: "Bearer test-token", "idempotency-key": "same-correction-request", "x-axon-student-id": TEST_STUDENT_ID }, body: JSON.stringify(correction)
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
  it("requires the correction owner before storing student data", async () => {
    const response = await exports.default.fetch(new Request("https://axon.test/v1/corrections", {
      method: "POST", headers: { authorization: "Bearer test-token", "content-type": "application/json" },
      body: JSON.stringify({ field: "answer", predicted: "x", corrected: "y", acceptedValue: "y", artifactId: "paper:owner", pipelineVersion: "3", model: "vision", promptHash: "hash", contextMetadata: {} })
    }));
    expect(response.status).toBe(400);
    await expect(response.json()).resolves.toMatchObject({ error: "INVALID_REQUEST", message: "Missing or invalid x-axon-student-id" });
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
