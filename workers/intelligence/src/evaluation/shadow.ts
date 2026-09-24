import type { TutorRequest, TutorResponse } from "../schemas";
import type { AIProvider, ModelRequest, ModelResponse } from "../providers/types";
import type { RetrievalService } from "../intelligence/retrieval/types";
import { TutorOrchestrator } from "../intelligence/tutor/orchestrator";

class ModelOverrideProvider implements AIProvider {
  readonly id: string;
  constructor(readonly base: AIProvider, readonly model: string) { this.id = base.id; }
  generate(request: ModelRequest): Promise<ModelResponse> { return this.base.generate({ ...request, model: this.model }); }
}

async function hash(value: string): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(value));
  return [...new Uint8Array(digest)].map((byte) => byte.toString(16).padStart(2, "0")).join("");
}

export async function runShadowTutor(input: {
  db: D1Database; liveTraceId: string; request: TutorRequest; provider: AIProvider; model: string;
  candidateConfigRevision: string; retrieval?: RetrievalService;
}): Promise<TutorResponse> {
  const orchestrator = new TutorOrchestrator({ provider: new ModelOverrideProvider(input.provider, input.model), ...(input.retrieval ? { retrieval: input.retrieval } : {}) });
  const response = await orchestrator.respond(input.request);
  const outputHash = await hash(JSON.stringify(response));
  await input.db.prepare(`INSERT INTO shadow_result
    (id, trace_id, candidate_config_revision, candidate_model, verification_status, metrics_json, output_hash, created_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)`)
    .bind(crypto.randomUUID(), input.liveTraceId, input.candidateConfigRevision, input.model,
      response.verification.passed ? "verified" : response.status,
      JSON.stringify({ passed: response.verification.passed, repaired: response.verification.repaired, failureCount: response.verification.failures.length, citationCount: response.citations.length }),
      outputHash, new Date().toISOString()).run();
  return response;
}
