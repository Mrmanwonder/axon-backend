import { parseSchema, CorrectionEventSchema, TutorRequestSchema } from "./schemas";
import { GeminiProvider } from "./providers/gemini";
import { TavilyRetrievalService } from "./providers/tavily";
import { TutorOrchestrator } from "./intelligence/tutor/orchestrator";
import { ingestPaperPage, processPaperBatch, type PaperJob } from "./document/ingest";
import { providerIsAvailable, recordConceptTaxonomy, recordDeploymentProvenance, recordProviderObservation, recordRuntimeArtifacts, writeTutorAudit } from "./intelligence/telemetry/repository";
import { readBoundedJsonBody } from "./shared/bounded-json";
import { authenticateInternalRequest } from "./intelligence/security/auth";
import { planCorrectionLearning, prioritizeCorrection } from "./intelligence/corrections/active-learning";
import { ConceptTaxonomy } from "./academic/concepts/taxonomy";
import { promptRegistry } from "./prompts";
import { commitReviewedPaperPage, getPaperPage, reviewPaperPage } from "./document/repository";
import { RUNTIME_CONFIG_V3 } from "./config/runtime.v3";
import { persistCapabilityProbe, probeReleaseCapabilities } from "./providers/capabilities";
import { runShadowTutor } from "./evaluation/shadow";
import { enforceRateLimit } from "./intelligence/security/rate-limit";
import { lookupIdempotentResult, storeIdempotentResult } from "./intelligence/security/idempotency";
import { readInsightPatterns, recordInsightObservation } from "./academic/insights/repository";
import { configuredModelRates, estimateModelCost } from "./intelligence/telemetry/cost";
import { persistEvalSuite } from "./evaluation/repository";
import { GOLDEN_CASES } from "./evaluation/golden";
import {
  calibrationSummary, correctionLearningStatements, listActiveLearning, listErrorClusters,
  listLearningTargets, reviewActiveLearning
} from "./intelligence/corrections/repository";
import { recordStableKnowledge, resolveStableKnowledge, resolveStableKnowledgeFromDb } from "./academic/stable-knowledge";
import { pseudonymizeIdentifier } from "./intelligence/security/privacy";
import { resolveLearningConsent, validStudentId, type LearningConsentDecision } from "./intelligence/corrections/consent";

const JSON_HEADERS = { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" };
const json = (value: unknown, status = 200): Response => new Response(JSON.stringify(value), { status, headers: JSON_HEADERS });

async function readBoundedJson(request: Request, maxBytes = 100_000): Promise<unknown> {
  const length = Number(request.headers.get("content-length") ?? "0");
  if (length > maxBytes) throw new Error("Request body too large");
  return readBoundedJsonBody(request.body, maxBytes);
}

async function handle(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
  const url = new URL(request.url);
  if (request.method === "GET" && url.pathname === "/health") {
    return json({ status: "ok", pipelineVersion: env.AXON_PIPELINE_VERSION, deploymentSha: env.AXON_DEPLOYMENT_SHA, configRevision: env.AXON_CONFIG_REVISION });
  }
  const expectedToken = url.pathname.startsWith("/v1/admin/") ? env.AXON_ADMIN_TOKEN : env.AXON_INTERNAL_TOKEN;
  if (url.pathname.startsWith("/v1/") && !await authenticateInternalRequest(request, expectedToken)) {
    return json({ error: "UNAUTHORIZED" }, 401);
  }
  if (url.pathname.startsWith("/v1/")) {
    const limit = url.pathname === "/v1/tutor" ? 60 : url.pathname === "/v1/papers/ingest" ? 20 : 120;
    const rate = await enforceRateLimit(env.DB, request, url.pathname, limit);
    if (!rate.allowed) return new Response(JSON.stringify({ error: "RATE_LIMITED", retryAfterSeconds: rate.retryAfterSeconds }), {
      status: 429, headers: { ...JSON_HEADERS, "retry-after": String(rate.retryAfterSeconds), "x-ratelimit-remaining": "0" }
    });
  }
  if (request.method === "POST" && url.pathname === "/v1/tutor") {
    const input = parseSchema(TutorRequestSchema, await readBoundedJson(request));
    const provider = new GeminiProvider(env, String(env.GEMINI_PRIVACY_MODE) === "zdr" ? "zdr" : "unverified");
    const retrieval = env.TAVILY_API_KEY ? new TavilyRetrievalService(env.TAVILY_API_KEY, env.TAVILY_API_BASE, env.PIPELINE_CACHE) : undefined;
    const orchestrator = new TutorOrchestrator({
      provider,
      ...(retrieval ? { retrieval } : {}),
      stableKnowledge: async (message) => {
        const stored = await resolveStableKnowledgeFromDb(env.DB, message);
        return stored.length > 0 ? stored : resolveStableKnowledge(message);
      },
      providerAvailable: () => providerIsAvailable(env.DB, provider.id, RUNTIME_CONFIG_V3.primaryModel),
      trace: (trace) => {
        const rates = configuredModelRates(env.GEMINI_INPUT_USD_PER_MILLION, env.GEMINI_OUTPUT_USD_PER_MILLION);
        const estimatedCost = rates && trace.inputTokens !== undefined && trace.outputTokens !== undefined ? estimateModelCost(trace.inputTokens, trace.outputTokens, rates) : undefined;
        const audit = writeTutorAudit(env.DB, {
          traceId: trace.traceId, stage: "tutor", capability: trace.intent,
          deploymentSha: env.AXON_DEPLOYMENT_SHA, configRevision: env.AXON_CONFIG_REVISION,
          pipelineVersion: env.AXON_PIPELINE_VERSION, ...(trace.provider ? { provider: trace.provider } : {}),
          ...(trace.requestedModel ? { requestedModel: trace.requestedModel } : {}), ...(trace.servedModel ? { servedModel: trace.servedModel } : {}),
          ...(trace.thinkingLevel ? { thinkingLevel: trace.thinkingLevel } : {}), ...(trace.promptId ? { promptId: trace.promptId } : {}),
          ...(trace.promptHash ? { promptHash: trace.promptHash } : {}), ...(trace.schemaId ? { schemaId: trace.schemaId } : {}),
          ...(trace.schemaHash ? { schemaHash: trace.schemaHash } : {}), toolCalls: trace.toolCalls, retrievalUsed: trace.retrievalUsed,
          verificationStatus: trace.verificationStatus, repairAttempted: trace.repairAttempted,
          intent: trace.intent, verificationFailures: trace.verificationFailures ?? [],
          groundingUsed: trace.groundingUsed ?? false, ...(trace.answerStatus ? { answerStatus: trace.answerStatus } : {}),
          ...(trace.latencyMs !== undefined ? { latencyMs: trace.latencyMs } : {}),
          ...(trace.inputTokens !== undefined ? { inputTokens: trace.inputTokens } : {}), ...(trace.outputTokens !== undefined ? { outputTokens: trace.outputTokens } : {}),
          ...(estimatedCost !== undefined ? { estimatedCost } : {}),
          ...(trace.transportSuccess !== undefined ? { transportSuccess: trace.transportSuccess } : {}),
          ...(trace.schemaSuccess !== undefined ? { schemaSuccess: trace.schemaSuccess } : {}),
          ...(trace.semanticValidationSuccess !== undefined ? { semanticValidationSuccess: trace.semanticValidationSuccess } : {}),
          inputArtifactHashes: [...new Set((trace.evidence ?? []).flatMap((item) => item.provenance.artifactHash ? [item.provenance.artifactHash] : []))], ...(trace.error ? { error: trace.error } : {})
        }, trace.evidence ?? [], trace.claims ?? []);
        const providerHealth = trace.provider ? recordProviderObservation(env.DB, trace.provider, trace.servedModel ?? trace.requestedModel ?? RUNTIME_CONFIG_V3.primaryModel, {
          success: trace.verificationStatus === "verified", latencyMs: trace.latencyMs ?? 0,
          timeout: trace.error === "MODEL_TIMEOUT", rateLimited: trace.error === "MODEL_RATE_LIMIT",
          serverError: trace.error === "MODEL_SERVER_ERROR", schemaFailure: trace.error?.includes("INVALID_SCHEMA") ?? false,
          semanticFailure: trace.verificationStatus === "failed"
        }) : Promise.resolve();
        ctx.waitUntil(Promise.all([audit, providerHealth]).then(() => undefined));
      }
    });
    const response = await orchestrator.respond(input);
    if (env.AXON_SHADOW_MODEL && String(env.GEMINI_PRIVACY_MODE) === "zdr") {
      ctx.waitUntil(runShadowTutor({
        db: env.DB, liveTraceId: response.traceId, request: input, provider, model: env.AXON_SHADOW_MODEL,
        candidateConfigRevision: env.AXON_SHADOW_CONFIG_REVISION ?? `shadow:${env.AXON_SHADOW_MODEL}`,
        ...(retrieval ? { retrieval } : {})
      }).then(() => undefined).catch((error) => console.error(JSON.stringify({ event: "shadow_failed", traceId: response.traceId, error: error instanceof Error ? error.message : String(error) }))));
    }
    return json(response);
  }
  if (request.method === "POST" && url.pathname === "/v1/papers/ingest") return json(await ingestPaperPage(request, env), 202);
  const paperPageMatch = url.pathname.match(/^\/v1\/papers\/pages\/([^/]+)$/);
  if (request.method === "GET" && paperPageMatch?.[1]) {
    const page = await getPaperPage(env.DB, decodeURIComponent(paperPageMatch[1]));
    return page ? json(page) : json({ error: "NOT_FOUND" }, 404);
  }
  const paperReviewMatch = url.pathname.match(/^\/v1\/papers\/pages\/([^/]+)\/review$/);
  if (request.method === "POST" && paperReviewMatch?.[1]) return json(await reviewPaperPage(env.DB, decodeURIComponent(paperReviewMatch[1]), await readBoundedJson(request)), 200);
  const paperCommitMatch = url.pathname.match(/^\/v1\/papers\/pages\/([^/]+)\/commit$/);
  if (request.method === "POST" && paperCommitMatch?.[1]) {
    await commitReviewedPaperPage(env.DB, decodeURIComponent(paperCommitMatch[1]));
    return json({ committed: true });
  }
  if (request.method === "POST" && url.pathname === "/v1/corrections") {
    const correction = parseSchema(CorrectionEventSchema, await readBoundedJson(request));
    const sourceStudentId = request.headers.get("x-axon-student-id");
    if (!validStudentId(sourceStudentId)) throw new Error("Missing or invalid x-axon-student-id");
    const studentPseudonym = await pseudonymizeIdentifier(sourceStudentId, env.AXON_PSEUDONYM_KEY);
    const idempotency = await lookupIdempotentResult(env.DB, request, url.pathname, { correction, studentPseudonym });
    if (idempotency.cached) return json(idempotency.cached.payload, idempotency.cached.status);
    const consent = await resolveLearningConsent(env, sourceStudentId);
    const id = crypto.randomUUID();
    const createdAt = new Date().toISOString();
    const statements: D1PreparedStatement[] = [
      env.DB.prepare(`INSERT INTO student_correction
        (id, field, predicted_json, corrected_json, accepted_json, artifact_id, pipeline_version, model, prompt_hash, context_metadata_json, created_at,
         student_id, learning_consent_granted, learning_consent_seq, learning_consent_notice_version)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`)
        .bind(id, correction.field, JSON.stringify(correction.predicted), JSON.stringify(correction.corrected), JSON.stringify(correction.acceptedValue), correction.artifactId, correction.pipelineVersion, correction.model, correction.promptHash, JSON.stringify(correction.contextMetadata), createdAt,
          studentPseudonym, consent.granted ? 1 : 0, consent.seq ?? null, consent.noticeVersion ?? null)
    ];
    let learningId: string | null = null;
    if (consent.granted) {
      const learning = prioritizeCorrection(correction);
      const learningPlan = await planCorrectionLearning(correction, learning.reasons);
      learningId = crypto.randomUUID();
      statements.push(
        env.DB.prepare("INSERT INTO active_learning_queue (id, correction_id, priority, reasons_json, status, created_at) VALUES (?, ?, ?, ?, 'QUEUED', ?)")
          .bind(learningId, id, learning.priority, JSON.stringify(learning.reasons), createdAt),
        ...correctionLearningStatements(env.DB, id, learningPlan, createdAt)
      );
    }
    await env.DB.batch(statements);
    const result = { id, accepted: true, activeLearningId: learningId, learningConsent: consent.state };
    await storeIdempotentResult(env.DB, url.pathname, idempotency, { status: 201, payload: result });
    return json(result, 201);
  }
  if (request.method === "POST" && url.pathname === "/v1/insights/observations") {
    const id = await recordInsightObservation(env.DB, await readBoundedJson(request));
    return json({ id, accepted: true }, 201);
  }
  if (request.method === "GET" && url.pathname === "/v1/insights/patterns") {
    const studentId = request.headers.get("x-axon-student-id");
    if (!studentId) throw new Error("Missing x-axon-student-id");
    return json({ patterns: await readInsightPatterns(env.DB, studentId, env.AXON_PSEUDONYM_KEY) });
  }
  if (request.method === "POST" && url.pathname === "/v1/admin/capabilities/probe") {
    const provider = new GeminiProvider(env, String(env.GEMINI_PRIVACY_MODE) === "zdr" ? "zdr" : "unverified");
    // Certification must exercise the live search and extract endpoints, not a
    // previously cached result that merely proves an older deployment worked.
    const retrieval = env.TAVILY_API_KEY ? new TavilyRetrievalService(env.TAVILY_API_KEY, env.TAVILY_API_BASE, undefined, 5_000) : undefined;
    const result = await probeReleaseCapabilities({
      provider,
      model: RUNTIME_CONFIG_V3.primaryModel,
      ...(retrieval ? { retrieval } : {}),
      visionService: env.DOCUMENT_VISION,
      visionPrivacyAttested: String(env.AXON_VISION_PRIVACY_MODE) === "zdr",
      deploymentSha: env.AXON_DEPLOYMENT_SHA,
      configRevision: env.AXON_CONFIG_REVISION
    });
    const provenance = { deploymentSha: env.AXON_DEPLOYMENT_SHA, configRevision: env.AXON_CONFIG_REVISION };
    await Promise.all([
      persistCapabilityProbe(env.DB, provider.id, result.artifact.model, result.tutor, provenance),
      persistCapabilityProbe(env.DB, "tavily", "search-extract", [result.tavily], provenance),
      persistCapabilityProbe(env.DB, "axon-document-vision", "private-v1", [result.vision], provenance)
    ]);
    return json({ ...result.artifact, details: { tutor: result.tutor, tavily: result.tavily, vision: result.vision } });
  }
  if (request.method === "GET" && url.pathname === "/v1/admin/active-learning") {
    return json({ items: await listActiveLearning(env.DB, url.searchParams.get("status") ?? "QUEUED", Number(url.searchParams.get("limit") ?? "50")) });
  }
  if (request.method === "GET" && url.pathname === "/v1/admin/learning/targets") {
    return json({ items: await listLearningTargets(env.DB, url.searchParams.get("target"), url.searchParams.get("status") ?? "QUEUED", Number(url.searchParams.get("limit") ?? "100")) });
  }
  if (request.method === "GET" && url.pathname === "/v1/admin/learning/error-clusters") {
    return json({ clusters: await listErrorClusters(env.DB, Number(url.searchParams.get("limit") ?? "100")) });
  }
  if (request.method === "GET" && url.pathname === "/v1/admin/learning/calibration") {
    return json({ buckets: await calibrationSummary(env.DB) });
  }
  const activeLearningMatch = url.pathname.match(/^\/v1\/admin\/active-learning\/([^/]+)$/);
  if (request.method === "POST" && activeLearningMatch?.[1]) {
    const sourceStudentId = request.headers.get("x-axon-student-id");
    let decision: LearningConsentDecision = { state: "UNVERIFIED", granted: false, reason: "INVALID_STUDENT_ID" };
    let studentPseudonym: string | undefined;
    if (validStudentId(sourceStudentId)) {
      [decision, studentPseudonym] = await Promise.all([
        resolveLearningConsent(env, sourceStudentId),
        pseudonymizeIdentifier(sourceStudentId, env.AXON_PSEUDONYM_KEY)
      ]);
    }
    await reviewActiveLearning(env.DB, decodeURIComponent(activeLearningMatch[1]), await readBoundedJson(request), { decision, ...(studentPseudonym ? { studentPseudonym } : {}) });
    return json({ updated: true });
  }
  if (request.method === "GET" && url.pathname === "/v1/admin/provider-health") {
    const rows = await env.DB.prepare("SELECT * FROM provider_health ORDER BY provider, model").all();
    return json({ providers: rows.results });
  }
  if (request.method === "GET" && url.pathname === "/v1/admin/readiness") {
    const [promptCount, openProviders, structuredProbe, thinkingProbe, retrievalProbe, visionProbe, visionServiceReady] = await Promise.all([
      env.DB.prepare("SELECT COUNT(*) AS count FROM prompt_artifact").first<{ count: number }>(),
      env.DB.prepare("SELECT COUNT(*) AS count FROM provider_health WHERE state = 'OPEN'").first<{ count: number }>(),
      env.DB.prepare("SELECT model, passed FROM capability_probe WHERE provider = 'gemini-zdr' AND capability = 'structured_output' AND deployment_sha = ? AND config_revision = ? ORDER BY probed_at DESC LIMIT 1").bind(env.AXON_DEPLOYMENT_SHA, env.AXON_CONFIG_REVISION).first<{ model: string; passed: number }>(),
      env.DB.prepare("SELECT model, passed FROM capability_probe WHERE provider = 'gemini-zdr' AND capability = 'thinking' AND deployment_sha = ? AND config_revision = ? ORDER BY probed_at DESC LIMIT 1").bind(env.AXON_DEPLOYMENT_SHA, env.AXON_CONFIG_REVISION).first<{ model: string; passed: number }>(),
      env.DB.prepare("SELECT passed FROM capability_probe WHERE provider = 'tavily' AND capability = 'native_search' AND deployment_sha = ? AND config_revision = ? ORDER BY probed_at DESC LIMIT 1").bind(env.AXON_DEPLOYMENT_SHA, env.AXON_CONFIG_REVISION).first<{ passed: number }>(),
      env.DB.prepare("SELECT passed FROM capability_probe WHERE provider = 'axon-document-vision' AND capability = 'image_input' AND deployment_sha = ? AND config_revision = ? ORDER BY probed_at DESC LIMIT 1").bind(env.AXON_DEPLOYMENT_SHA, env.AXON_CONFIG_REVISION).first<{ passed: number }>(),
      String(env.AXON_VISION_PRIVACY_MODE) === "zdr"
        ? env.DOCUMENT_VISION.fetch("https://axon-document-vision/health").then((response) => response.ok).catch(() => false)
        : Promise.resolve(false)
    ]);
    const rates = configuredModelRates(env.GEMINI_INPUT_USD_PER_MILLION, env.GEMINI_OUTPUT_USD_PER_MILLION);
    const checks = {
      geminiZdr: String(env.GEMINI_PRIVACY_MODE) === "zdr",
      visionZdr: String(env.AXON_VISION_PRIVACY_MODE) === "zdr" && visionServiceReady,
      retrievalConfigured: Boolean(env.TAVILY_API_KEY),
      retrievalProbePassed: retrievalProbe?.passed === 1,
      visionProbePassed: visionProbe?.passed === 1,
      pseudonymizationConfigured: Boolean(env.AXON_PSEUDONYM_KEY),
      supabaseAdminKeyConfigured: Boolean(env.SUPABASE_SECRET_KEY ?? env.SUPABASE_SERVICE_ROLE_KEY),
      promptArtifactsPersisted: (promptCount?.count ?? 0) > 0,
      providerCircuitsClosed: (openProviders?.count ?? 0) === 0,
      structuredOutputProbePassed: structuredProbe?.model === RUNTIME_CONFIG_V3.primaryModel && structuredProbe.passed === 1,
      thinkingProbePassed: thinkingProbe?.model === RUNTIME_CONFIG_V3.primaryModel && thinkingProbe.passed === 1,
      costRatesConfigured: Boolean(rates),
      releaseCertified: String(env.AXON_RELEASE_CERTIFIED) === "true"
    };
    const ready = Object.values(checks).every(Boolean);
    return json({ status: ready ? "ready" : "not_ready", checks }, ready ? 200 : 503);
  }
  return json({ error: "NOT_FOUND" }, 404);
}

export default {
  async fetch(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
    ctx.waitUntil(recordDeploymentProvenance(env.DB, { deploymentSha: env.AXON_DEPLOYMENT_SHA, configRevision: env.AXON_CONFIG_REVISION, pipelineVersion: env.AXON_PIPELINE_VERSION }));
    ctx.waitUntil(recordConceptTaxonomy(env.DB, new ConceptTaxonomy().all(), env.AXON_CONFIG_REVISION));
    ctx.waitUntil(promptRegistry.compileAll().then((prompts) => recordRuntimeArtifacts(env.DB, prompts, env.AXON_DEPLOYMENT_SHA, env.AXON_CONFIG_REVISION)));
    ctx.waitUntil(persistEvalSuite(env.DB, { id: "axon-golden-v3", name: "AXON v3 synthetic regression suite", version: "3.0.0", category: "all", cases: GOLDEN_CASES }));
    ctx.waitUntil(recordStableKnowledge(env.DB, env.AXON_CONFIG_REVISION));
    try { return await handle(request, env, ctx); }
    catch (error) {
      const message = error instanceof Error ? error.message : "Unknown error";
      console.error(JSON.stringify({ event: "request_failed", path: new URL(request.url).pathname, error: message }));
      const clientError = /Missing|validation|too large|must be|require|Invalid|unsupported|does not match|not found|not awaiting|unresolved/i.test(message);
      return json({ error: clientError ? "INVALID_REQUEST" : "INTERNAL_ERROR", message: clientError ? message : "The request could not be completed." }, clientError ? 400 : 500);
    }
  },
  async queue(batch: MessageBatch<PaperJob>, env: Env): Promise<void> { await processPaperBatch(batch, env); },
  scheduled(_controller: ScheduledController, env: Env, ctx: ExecutionContext): void {
    const now = Date.now();
    ctx.waitUntil(env.DB.batch([
      env.DB.prepare("DELETE FROM api_rate_window WHERE window_start < ?").bind(now - 2 * 24 * 60 * 60 * 1_000),
      env.DB.prepare("DELETE FROM request_idempotency WHERE expires_at < ?").bind(new Date(now).toISOString()),
      env.DB.prepare("DELETE FROM provider_observation WHERE created_at < ?").bind(new Date(now - 30 * 24 * 60 * 60 * 1_000).toISOString())
    ]).then(() => undefined));
  }
} satisfies ExportedHandler<Env, PaperJob>;
