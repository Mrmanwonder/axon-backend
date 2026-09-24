import type { Claim, Evidence } from "../../schemas";
import type { Concept } from "../../academic/concepts/taxonomy";
import type { CompiledPrompt } from "../../prompts";
import { RUNTIME_CONFIG_V3 } from "../../config/runtime.v3";
import type { ProviderObservation } from "../routing/circuit-breaker";

export interface TraceRecord {
  traceId: string; stage: string; capability: string; deploymentSha: string; configRevision: string; pipelineVersion: string;
  paperId?: string; questionId?: string; confidence?: number;
  provider?: string; requestedModel?: string; servedModel?: string; thinkingLevel?: string; promptId?: string; promptHash?: string;
  schemaId?: string; schemaHash?: string; toolCalls: string[]; retrievalUsed: boolean; verificationStatus: string;
  repairAttempted: boolean; latencyMs?: number; inputTokens?: number; outputTokens?: number; estimatedCost?: number;
  transportSuccess?: boolean; schemaSuccess?: boolean; semanticValidationSuccess?: boolean;
  inputArtifactHashes: string[]; error?: string;
}

export async function writeTrace(db: D1Database, trace: TraceRecord): Promise<void> {
  await db.prepare(`INSERT INTO ai_trace (
    trace_id, paper_id, question_id, stage, capability, deployment_sha, config_revision, pipeline_version, provider,
    requested_model, served_model, thinking_level, prompt_id, prompt_hash, schema_id, schema_hash,
    tool_calls, retrieval_used, verification_status, repair_attempted, confidence, latency_ms, input_tokens, output_tokens, estimated_cost,
    transport_success, schema_success, semantic_validation_success, input_artifact_hashes, error
  ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`)
    .bind(trace.traceId, trace.paperId ?? null, trace.questionId ?? null, trace.stage, trace.capability, trace.deploymentSha, trace.configRevision, trace.pipelineVersion,
      trace.provider ?? null, trace.requestedModel ?? null, trace.servedModel ?? null, trace.thinkingLevel ?? null,
      trace.promptId ?? null, trace.promptHash ?? null, trace.schemaId ?? null, trace.schemaHash ?? null,
      JSON.stringify(trace.toolCalls), trace.retrievalUsed ? 1 : 0, trace.verificationStatus,
      trace.repairAttempted ? 1 : 0, trace.confidence ?? null, trace.latencyMs ?? null, trace.inputTokens ?? null, trace.outputTokens ?? null, trace.estimatedCost ?? null,
      trace.transportSuccess === undefined ? null : trace.transportSuccess ? 1 : 0,
      trace.schemaSuccess === undefined ? null : trace.schemaSuccess ? 1 : 0,
      trace.semanticValidationSuccess === undefined ? null : trace.semanticValidationSuccess ? 1 : 0,
      JSON.stringify(trace.inputArtifactHashes), trace.error ?? null).run();
}

export async function recordDeploymentProvenance(db: D1Database, values: { deploymentSha: string; configRevision: string; pipelineVersion: string }): Promise<void> {
  const createdAt = new Date().toISOString();
  await db.batch([
    db.prepare("INSERT OR IGNORE INTO ai_config_revision (revision_id, parent_revision_id, git_sha, author, reason, created_at) VALUES (?, NULL, ?, 'git', 'Versioned runtime configuration', ?)")
      .bind(values.configRevision, values.deploymentSha, createdAt),
    db.prepare("INSERT OR IGNORE INTO ai_deployment (deployment_sha, config_revision, pipeline_version, rollout_percent, state, created_at) VALUES (?, ?, ?, 0, 'BENCHMARK', ?)")
      .bind(values.deploymentSha, values.configRevision, values.pipelineVersion, createdAt)
  ]);
}

function persistedEvidenceValue(evidence: Evidence): unknown {
  if (evidence.source === "student" || evidence.source === "paper" || evidence.source === "teacher" || evidence.source === "axon_db") {
    return { redacted: true, artifactHash: evidence.provenance.artifactHash ?? null };
  }
  return evidence.value;
}

export async function writeEvidenceGraph(db: D1Database, traceId: string, evidence: readonly Evidence[], claims: readonly Claim[]): Promise<void> {
  const statements: D1PreparedStatement[] = [];
  const evidenceIds = new Set(evidence.map((item) => item.id));
  for (const item of evidence) {
    const storedId = `${traceId}:${item.id}`;
    statements.push(db.prepare("INSERT OR REPLACE INTO evidence (id, trace_id, information_class, source, authority, value_json, provenance_json, verification, confidence) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)")
      .bind(storedId, traceId, item.informationClass, item.source, item.authority, JSON.stringify(persistedEvidenceValue(item)), JSON.stringify(item.provenance), item.verification, item.confidence ?? null));
  }
  for (const claim of claims) {
    const storedClaimId = `${traceId}:${claim.id}`;
    statements.push(db.prepare("INSERT OR REPLACE INTO claim (id, trace_id, text, type, risk, verification_status) VALUES (?, ?, ?, ?, ?, ?)")
      .bind(storedClaimId, traceId, claim.text, claim.type, claim.risk, claim.verificationStatus));
    for (const evidenceId of claim.evidenceIds.filter((id) => evidenceIds.has(id))) statements.push(db.prepare("INSERT OR IGNORE INTO claim_evidence (claim_id, evidence_id) VALUES (?, ?)").bind(storedClaimId, `${traceId}:${evidenceId}`));
  }
  if (statements.length > 0) await db.batch(statements);
}

export async function writeTutorAudit(db: D1Database, trace: TraceRecord, evidence: readonly Evidence[] = [], claims: readonly Claim[] = []): Promise<void> {
  await writeTrace(db, trace);
  await writeEvidenceGraph(db, trace.traceId, evidence, claims);
}

export async function recordConceptTaxonomy(db: D1Database, concepts: readonly Concept[], revision: string): Promise<void> {
  if (concepts.length === 0) return;
  await db.batch(concepts.map((concept) => db.prepare("INSERT OR IGNORE INTO concept_taxonomy (id, subject, parent_id, aliases_json, curricula_json, revision) VALUES (?, ?, ?, ?, ?, ?)")
    .bind(concept.id, concept.subject, concept.parent ?? null, JSON.stringify(concept.aliases), JSON.stringify(concept.curricula), revision)));
}

export async function recordRuntimeArtifacts(db: D1Database, prompts: readonly CompiledPrompt[], deploymentSha: string, configRevision: string): Promise<void> {
  const createdAt = new Date().toISOString();
  const statements: D1PreparedStatement[] = [];
  for (const prompt of prompts) {
    statements.push(db.prepare("INSERT OR IGNORE INTO schema_artifact (schema_id, schema_hash, schema_json, created_at) VALUES (?, ?, ?, ?)")
      .bind(prompt.schemaId, prompt.schemaHash, JSON.stringify(prompt.schema), createdAt));
    statements.push(db.prepare("INSERT OR IGNORE INTO prompt_artifact (prompt_id, prompt_hash, schema_id, schema_hash, deployment_sha, created_at) VALUES (?, ?, ?, ?, ?, ?)")
      .bind(prompt.id, prompt.promptHash, prompt.schemaId, prompt.schemaHash, deploymentSha, createdAt));
  }
  for (const [capability, promptId] of Object.entries(RUNTIME_CONFIG_V3.routes)) {
    statements.push(db.prepare(`INSERT OR IGNORE INTO ai_route
      (id, capability, risk, privacy_policy, provider, model, thinking_level, prompt_id, timeout_ms, enabled, config_revision)
      VALUES (?, ?, 'R1-R4', ?, 'gemini-zdr', ?, ?, ?, ?, 1, ?)`)
      .bind(`${configRevision}:${capability}`, capability,
        capability === "document_reading" ? "STUDENT_DOCUMENT_STRICT" : "STUDENT_CHAT_STRICT",
        RUNTIME_CONFIG_V3.primaryModel,
        capability === "classification" ? RUNTIME_CONFIG_V3.thinking.classification : capability === "verification" ? RUNTIME_CONFIG_V3.thinking.verification : capability === "adjudication" ? RUNTIME_CONFIG_V3.thinking.adjudication : RUNTIME_CONFIG_V3.thinking.standard,
        promptId, capability === "adjudication" ? RUNTIME_CONFIG_V3.timeouts.criticalMs : RUNTIME_CONFIG_V3.timeouts.defaultMs, configRevision));
  }
  if (statements.length > 0) await db.batch(statements);
}

export async function recordProviderObservation(db: D1Database, provider: string, model: string, observation: ProviderObservation): Promise<void> {
  const createdAt = new Date().toISOString();
  const prior = await db.prepare("SELECT state FROM provider_health WHERE provider = ? AND model = ?").bind(provider, model).first<{ state: string }>();
  await db.prepare(`INSERT INTO provider_observation
    (id, provider, model, success, timeout, rate_limited, server_error, schema_failure, semantic_failure, latency_ms, created_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`)
    .bind(crypto.randomUUID(), provider, model, observation.success ? 1 : 0, observation.timeout ? 1 : 0,
      observation.rateLimited ? 1 : 0, observation.serverError ? 1 : 0, observation.schemaFailure ? 1 : 0,
      observation.semanticFailure ? 1 : 0, observation.latencyMs, createdAt).run();
  const recent = await db.prepare(`SELECT timeout, rate_limited, server_error, schema_failure, semantic_failure, latency_ms
    FROM provider_observation WHERE provider = ? AND model = ? ORDER BY created_at DESC LIMIT 50`)
    .bind(provider, model).all<{ timeout: number; rate_limited: number; server_error: number; schema_failure: number; semantic_failure: number; latency_ms: number }>();
  const count = recent.results.length;
  const average = (field: "timeout" | "rate_limited" | "server_error" | "schema_failure" | "semantic_failure"): number => count ? recent.results.reduce((sum, item) => sum + item[field], 0) / count : 0;
  const rates = { timeout: average("timeout"), rateLimit: average("rate_limited"), serverError: average("server_error"), schema: average("schema_failure"), semantic: average("semantic_failure") };
  const failureRate = Math.max(rates.timeout, rates.rateLimit, rates.serverError, rates.schema, rates.semantic);
  const state = prior?.state === "OPEN" ? (observation.success ? "CLOSED" : "OPEN") : count >= 5 && failureRate >= 0.5 ? "OPEN" : "CLOSED";
  const sortedLatency = recent.results.map((item) => item.latency_ms).sort((left, right) => left - right);
  const p95Latency = sortedLatency[Math.max(0, Math.ceil(sortedLatency.length * 0.95) - 1)] ?? 0;
  await db.prepare(`INSERT OR REPLACE INTO provider_health
    (provider, model, state, timeout_rate, rate_limit_rate, server_error_rate, schema_failure_rate, semantic_failure_rate, p95_latency_ms, updated_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`)
    .bind(provider, model, state, rates.timeout, rates.rateLimit, rates.serverError, rates.schema, rates.semantic, p95Latency, createdAt).run();
}

export async function providerIsAvailable(db: D1Database, provider: string, model: string): Promise<boolean> {
  const row = await db.prepare("SELECT state, updated_at FROM provider_health WHERE provider = ? AND model = ?").bind(provider, model).first<{ state: string; updated_at: string }>();
  if (!row || row.state !== "OPEN") return true;
  return Date.now() - Date.parse(row.updated_at) >= 30_000;
}
