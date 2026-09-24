import { describe, expect, it } from "vitest";
import { env, exports } from "cloudflare:workers";
import { processPaperPage } from "../src/document/orchestrator";
import type { DocumentVisionProvider, VisionAnalysis } from "../src/document/vision/provider";
import { providerIsAvailable, recordProviderObservation, writeTutorAudit } from "../src/intelligence/telemetry/repository";
import { runShadowTutor } from "../src/evaluation/shadow";
import type { AIProvider } from "../src/providers/types";
import { enforceRateLimit } from "../src/intelligence/security/rate-limit";
import { pseudonymizeIdentifier } from "../src/intelligence/security/privacy";

const printed = { printedProbability: 0.98, colourDistanceFromPrint: 0.02, strokeDifference: 0.02, marginTendency: 0.05, annotationOverlap: 0.05, handwritingDifference: 0.05 };
const student = { printedProbability: 0.02, colourDistanceFromPrint: 0.1, strokeDifference: 0.9, marginTendency: 0.05, annotationOverlap: 0.05, handwritingDifference: 0.05 };
const teacher = { printedProbability: 0.02, colourDistanceFromPrint: 0.9, strokeDifference: 0.9, marginTendency: 0.98, annotationOverlap: 0.98, handwritingDifference: 0.9 };
const region = (id: string, kind: VisionAnalysis["regions"][number]["class"], x: number, y: number, text: string, inkSignals: typeof printed) => ({ id, class: kind, box: { x, y, width: 0.2, height: 0.05 }, confidence: 0.98, text, inkSignals });
const read = (id: string, value: string, x: number, y: number) => ({ regionId: id, reads: ["reader-a", "reader-b"].map((readerId) => ({ value, alternatives: [], status: "read" as const, region: { x, y, width: 0.2, height: 0.05 }, readerIds: [readerId] })) });

describe("operational pipeline", () => {
  it("runs privacy-compliant document analysis through deterministic trusted commit", async () => {
    const pageId = crypto.randomUUID();
    const paperId = crypto.randomUUID();
    const objectKey = `papers/${paperId}/original/${pageId}`;
    const bytes = new TextEncoder().encode("synthetic page");
    await env.PAPER_ARTIFACTS.put(objectKey, bytes);
    await env.DB.prepare("INSERT INTO paper_page (paper_id, page_id, student_id, original_hash, source_type, object_key, processing_state, created_at) VALUES (?, ?, 'student', 'hash', 'image/png', ?, 'QUEUED', ?)")
      .bind(paperId, pageId, objectKey, new Date().toISOString()).run();
    const analysis: VisionAnalysis = {
      qualityMetrics: { blur: 0.02, glareFraction: 0, perspectiveDegrees: 0, resolution: 1, compression: 0.01, cropCompleteness: 1, shadowFraction: 0 },
      orientationDegrees: 0,
      regions: [
        region("q", "question_number", 0.1, 0.1, "1", printed),
        region("qt", "printed_question", 0.15, 0.15, "2 + 2", printed),
        region("a", "student_answer", 0.2, 0.25, "4", student),
        region("m", "marginal_mark", 0.85, 0.25, "1", teacher)
      ],
      reads: [read("q", "1", 0.1, 0.1), read("qt", "2 + 2", 0.15, 0.15), read("a", "4", 0.2, 0.25), read("m", "1", 0.85, 0.25)]
    };
    const provider: DocumentVisionProvider = { id: "test-vision", privacyMode: "zdr", analyze: () => Promise.resolve({ analysis, latencyMs: 3 }) };
    await processPaperPage(env, { paperId, pageId, studentId: "student", originalHash: "hash", sourceType: "image/png", timestamp: new Date().toISOString(), objectKey, pageIndex: 0 }, provider);
    const page = await env.DB.prepare("SELECT processing_state, quality_class FROM paper_page WHERE page_id = ?").bind(pageId).first<{ processing_state: string; quality_class: string }>();
    expect(page).toEqual({ processing_state: "TRUSTED_COMMIT", quality_class: "CLEAR" });
    const fields = await env.DB.prepare("SELECT COUNT(*) AS count FROM trusted_field WHERE artifact_id LIKE ? AND trust_state = 'AUTO_VERIFIED'").bind(`${pageId}:%`).first<{ count: number }>();
    expect(fields?.count).toBe(4);
  });

  it("persists immutable prompt and route artifacts", async () => {
    await exports.default.fetch(new Request("https://axon.test/health"));
    const prompts = await env.DB.prepare("SELECT COUNT(*) AS count FROM prompt_artifact").first<{ count: number }>();
    const routes = await env.DB.prepare("SELECT COUNT(*) AS count FROM ai_route WHERE config_revision = 'v3.default'").first<{ count: number }>();
    const evalCases = await env.DB.prepare("SELECT COUNT(*) AS count FROM eval_case WHERE suite_id = 'axon-golden-v3'").first<{ count: number }>();
    const stableFacts = await env.DB.prepare("SELECT COUNT(*) AS count FROM stable_knowledge WHERE active = 1").first<{ count: number }>();
    expect(prompts?.count).toBeGreaterThanOrEqual(5);
    expect(routes?.count).toBe(5);
    expect(evalCases?.count).toBeGreaterThanOrEqual(17);
    expect(stableFacts?.count).toBeGreaterThanOrEqual(3);
  });

  it("opens the provider health gate after repeated failures", async () => {
    const provider = `provider-${crypto.randomUUID()}`;
    for (let index = 0; index < 5; index += 1) await recordProviderObservation(env.DB, provider, "model", { success: false, serverError: true, latencyMs: 100 });
    await expect(providerIsAvailable(env.DB, provider, "model")).resolves.toBe(false);
  });

  it("persists semantic observability separately from transport success", async () => {
    const traceId = crypto.randomUUID();
    await writeTutorAudit(env.DB, {
      traceId, stage: "tutor", capability: "current_information", intent: "current_information",
      deploymentSha: "test-sha", configRevision: "test-config", pipelineVersion: "3.0.0",
      toolCalls: ["tavily.search"], retrievalUsed: true, groundingUsed: true,
      verificationStatus: "failed", verificationFailures: ["unsupported_claim c1"],
      repairAttempted: true, answerStatus: "controlled_failure", transportSuccess: true,
      schemaSuccess: true, semanticValidationSuccess: false, inputArtifactHashes: []
    });
    const stored = await env.DB.prepare(`SELECT intent, verification_failures, tool_calls, retrieval_used,
      grounding_used, verification_status, repair_attempted, answer_status,
      transport_success, schema_success, semantic_validation_success
      FROM ai_trace WHERE trace_id = ?`).bind(traceId).first<Record<string, unknown>>();
    expect(stored).toMatchObject({
      intent: "current_information", verification_failures: '["unsupported_claim c1"]',
      tool_calls: '["tavily.search"]', retrieval_used: 1, grounding_used: 1,
      verification_status: "failed", repair_attempted: 1, answer_status: "controlled_failure",
      transport_success: 1, schema_success: 1, semantic_validation_success: 0
    });
  });

  it("enforces a durable per-route request window without storing raw identity", async () => {
    const request = new Request("https://axon.test/v1/tutor", { headers: { "x-axon-client-id": crypto.randomUUID() } });
    await expect(enforceRateLimit(env.DB, request, "/unit-rate", 2, 1_000)).resolves.toMatchObject({ allowed: true, remaining: 1 });
    await expect(enforceRateLimit(env.DB, request, "/unit-rate", 2, 1_001)).resolves.toMatchObject({ allowed: true, remaining: 0 });
    await expect(enforceRateLimit(env.DB, request, "/unit-rate", 2, 1_002)).resolves.toMatchObject({ allowed: false, remaining: 0 });
  });

  it("records an unattested capability probe without sending a model request", async () => {
    const response = await exports.default.fetch(new Request("https://axon.test/v1/admin/capabilities/probe", { method: "POST", headers: { authorization: "Bearer test-admin-token" } }));
    expect(response.status).toBe(200);
    const payload = await response.json<{ results: Array<{ capability: string; passed: boolean }> }>();
    expect(payload.results.find((item) => item.capability === "zero_data_retention")?.passed).toBe(false);
    expect(payload.results.find((item) => item.capability === "structured_output")?.passed).toBe(false);
  });

  it("runs shadow output invisibly and stores only metrics plus a hash", async () => {
    const provider: AIProvider = {
      id: "gemini-zdr",
      generate: (request) => Promise.resolve({
        requestedModel: request.model, servedModel: request.model, latencyMs: 2, usage: {},
        output: { status: "supported", intent: "concept_explanation", claims: [{ id: "c1", text: "Mitosis produces daughter cells genetically similar to the parent cell.", type: "stable", evidenceIds: ["stable:biology.cells.cell_division.mitosis"], risk: "low", verificationStatus: "pending" }], conceptIds: ["biology.cells.cell_division.mitosis"], teachingStrategy: "direct" }
      })
    };
    const response = await runShadowTutor({ db: env.DB, liveTraceId: crypto.randomUUID(), request: { studentId: "synthetic", message: "Explain mitosis" }, provider, model: "candidate-model", candidateConfigRevision: "candidate-v1" });
    expect(response.verification.passed).toBe(true);
    const stored = await env.DB.prepare("SELECT output_hash, metrics_json FROM shadow_result WHERE candidate_config_revision = 'candidate-v1' ORDER BY created_at DESC LIMIT 1").first<{ output_hash: string; metrics_json: string }>();
    expect(stored?.output_hash).toMatch(/^[a-f0-9]{64}$/);
    expect(JSON.parse(stored?.metrics_json ?? "{}")).toMatchObject({ passed: true });
  });

  it("requires explicit resolution and verified fields before review commit", async () => {
    const pageId = crypto.randomUUID();
    const paperId = crypto.randomUUID();
    const regionId = `${pageId}:r1`;
    const fieldId = `${regionId}:recognized_text`;
    await env.DB.batch([
      env.DB.prepare("INSERT INTO paper_page (paper_id, page_id, student_id, original_hash, source_type, object_key, processing_state, review_reasons_json, created_at) VALUES (?, ?, 'student', 'review-hash', 'image/png', 'key', 'REVIEW_REQUIRED', '[\"AMBIGUOUS_CONTENT_READ\"]', ?)").bind(paperId, pageId, new Date().toISOString()),
      env.DB.prepare("INSERT INTO layout_region (id, page_id, class, box_json, confidence, trust_state) VALUES (?, ?, 'student_answer', '{}', 0.5, 'UNVERIFIED')").bind(regionId, pageId),
      env.DB.prepare("INSERT INTO trusted_field (id, artifact_id, field_name, value_json, trust_state, evidence_ids_json, updated_at) VALUES (?, ?, 'recognized_text', 'null', 'UNVERIFIED', '[]', ?)").bind(fieldId, regionId, new Date().toISOString())
    ]);
    const review = await exports.default.fetch(new Request(`https://axon.test/v1/papers/pages/${pageId}/review`, {
      method: "POST", headers: { authorization: "Bearer test-token", "content-type": "application/json" },
      body: JSON.stringify({ decisions: [{ fieldId, value: "student answer", evidenceIds: ["student-review"] }], resolvedReasons: ["AMBIGUOUS_CONTENT_READ"] })
    }));
    expect(review.status).toBe(200);
    const commit = await exports.default.fetch(new Request(`https://axon.test/v1/papers/pages/${pageId}/commit`, { method: "POST", headers: { authorization: "Bearer test-token" } }));
    expect(commit.status).toBe(200);
    const page = await env.DB.prepare("SELECT processing_state FROM paper_page WHERE page_id = ?").bind(pageId).first<{ processing_state: string }>();
    expect(page?.processing_state).toBe("TRUSTED_COMMIT");
  });

  it("admits only lineage-backed correctness into student insights", async () => {
    const pageId = crypto.randomUUID();
    const paperId = crypto.randomUUID();
    const regionId = `${pageId}:answer`;
    const fieldId = `${regionId}:correctness`;
    const studentId = await pseudonymizeIdentifier("insight-student", "test-pseudonym-key");
    await env.DB.batch([
      env.DB.prepare("INSERT OR IGNORE INTO concept_taxonomy (id, subject, parent_id, aliases_json, curricula_json, revision) VALUES ('mathematics.algebra', 'mathematics', NULL, '[]', '[\"general\"]', 'test')"),
      env.DB.prepare("INSERT INTO paper_page (paper_id, page_id, student_id, original_hash, source_type, object_key, processing_state, created_at) VALUES (?, ?, ?, ?, 'image/png', 'key', 'TRUSTED_COMMIT', ?)").bind(paperId, pageId, studentId, crypto.randomUUID(), new Date().toISOString()),
      env.DB.prepare("INSERT INTO layout_region (id, page_id, class, box_json, confidence, trust_state) VALUES (?, ?, 'student_answer', '{}', 0.97, 'AUTO_VERIFIED')").bind(regionId, pageId),
      env.DB.prepare("INSERT INTO trusted_field (id, artifact_id, field_name, value_json, trust_state, evidence_ids_json, updated_at) VALUES (?, ?, 'correctness', 'false', 'AUTO_VERIFIED', '[\"academic-tool:1\"]', ?)").bind(fieldId, regionId, new Date().toISOString())
    ]);
    const recorded = await exports.default.fetch(new Request("https://axon.test/v1/insights/observations", {
      method: "POST", headers: { authorization: "Bearer test-token", "content-type": "application/json" }, body: JSON.stringify({ trustedFieldId: fieldId, conceptId: "mathematics.algebra", paperId })
    }));
    expect(recorded.status).toBe(201);
    const patterns = await exports.default.fetch(new Request("https://axon.test/v1/insights/patterns", { headers: { authorization: "Bearer test-token", "x-axon-student-id": "insight-student" } }));
    await expect(patterns.json()).resolves.toMatchObject({ patterns: [{ conceptId: "mathematics.algebra", pattern: { established: false, relevantObservations: 1 }, correctRate: 0 }] });
  });

  it("requires human evidence and a passing eval before active-learning promotion", async () => {
    const correction = {
      field: "recognized_text", predicted: "x", corrected: "y", acceptedValue: "y", artifactId: crypto.randomUUID(),
      pipelineVersion: "3.0.0", model: "vision", promptHash: "prompt", contextMetadata: { confidence: 0.2 }
    };
    const created = await exports.default.fetch(new Request("https://axon.test/v1/corrections", {
      method: "POST", headers: { authorization: "Bearer test-token", "content-type": "application/json" }, body: JSON.stringify(correction)
    }));
    const { activeLearningId } = await created.json<{ activeLearningId: string }>();
    const labelled = await exports.default.fetch(new Request(`https://axon.test/v1/admin/active-learning/${activeLearningId}`, {
      method: "POST", headers: { authorization: "Bearer test-admin-token", "content-type": "application/json" },
      body: JSON.stringify({ status: "LABELLED", reviewer: "reviewer-1", evidenceUri: "urn:axon:review:evidence" })
    }));
    expect(labelled.status).toBe(200);
    const promoted = await exports.default.fetch(new Request(`https://axon.test/v1/admin/active-learning/${activeLearningId}`, {
      method: "POST", headers: { authorization: "Bearer test-admin-token", "content-type": "application/json" },
      body: JSON.stringify({ status: "PROMOTED", reviewer: "reviewer-1", evidenceUri: "urn:axon:review:evidence" })
    }));
    expect(promoted.status).toBe(400);
    const item = await env.DB.prepare("SELECT status, reviewer FROM active_learning_queue WHERE id = ?").bind(activeLearningId).first<{ status: string; reviewer: string }>();
    expect(item).toEqual({ status: "LABELLED", reviewer: "reviewer-1" });
  });
});
