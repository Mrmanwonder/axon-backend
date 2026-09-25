import { runAcademicTools } from "../../academic/tools";
import { resolveStableKnowledge } from "../../academic/stable-knowledge";
import { assembleContext } from "../context";
import { detectContradictions, detectEvidenceConflicts, verifyClaims } from "../claims/verifier";
import { renderReasoning } from "../pedagogy/renderer";
import { ModelRouter, classifyRisk } from "../routing/router";
import { NoCompliantProviderError } from "../routing/privacy";
import { parseSchema, ReasoningResultSchema, VerificationResultSchema, type Claim, type Evidence, type ReasoningResult, type TutorRequest, type TutorResponse } from "../../schemas";
import { promptRegistry } from "../../prompts";
import { ProviderError, type AIProvider, type ModelResponse } from "../../providers/types";
import { minimizePublicRetrievalQuery } from "../security/privacy";
import type { RetrievalService } from "../retrieval/types";
import { decideTools, detectIntent, promptIdFor, resolveTutorDepth } from "./routing";

export interface TutorRunTrace {
  traceId: string; intent: string; provider?: string; requestedModel?: string; servedModel?: string;
  thinkingLevel?: string; promptId?: string; promptHash?: string; schemaId?: string; schemaHash?: string;
  toolCalls: string[]; retrievalUsed: boolean; verificationStatus: string; repairAttempted: boolean; latencyMs?: number; error?: string;
  inputTokens?: number; outputTokens?: number; transportSuccess?: boolean; schemaSuccess?: boolean; semanticValidationSuccess?: boolean;
  verificationFailures?: string[]; groundingUsed?: boolean; answerStatus?: TutorResponse["status"];
  evidence?: Evidence[]; claims?: Claim[];
}
export interface TutorDependencies { provider: AIProvider; retrieval?: RetrievalService; stableKnowledge?: (message: string) => Promise<Evidence[]>; providerAvailable?: () => Promise<boolean>; trace?: (trace: TutorRunTrace) => void }

async function toolEvidence(request: TutorRequest, decisions: ReturnType<typeof decideTools>): Promise<Evidence[]> {
  const requested = new Set<string>();
  if (decisions.calculator) requested.add("axon.calculator.v1");
  if (decisions.symbolicMath) requested.add("axon.math.v1");
  if (decisions.units) requested.add("axon.units.v1");
  if (decisions.chemistry) requested.add("axon.chem.v1");
  try { return await runAcademicTools({ request: request.message, evidence: request.evidence ?? [] }, requested); }
  catch { return []; }
}

export function normalizeInboundEvidence(evidence: readonly Evidence[]): Evidence[] {
  return evidence.map((item) => {
    if (item.source !== "tool" && item.source !== "retrieval" && item.source !== "official_source" && item.source !== "stable_knowledge") return structuredClone(item);
    return { ...structuredClone(item), authority: "low", verification: "unverified", confidence: 0 };
  });
}

export class TutorOrchestrator {
  readonly #router = new ModelRouter();
  constructor(readonly dependencies: TutorDependencies) {}

  async respond(request: TutorRequest): Promise<TutorResponse> {
    const traceId = crypto.randomUUID();
    const intent = detectIntent(request);
    const decisions = decideTools(request, intent);
    const stableEvidence = this.dependencies.stableKnowledge ? await this.dependencies.stableKnowledge(request.message) : resolveStableKnowledge(request.message);
    const evidence = [...normalizeInboundEvidence(request.evidence ?? []), ...stableEvidence, ...await toolEvidence(request, decisions)];
    const retrievedEvidenceIds = new Set<string>();
    if (decisions.retrieval) {
      if (!this.dependencies.retrieval) {
        this.emit({ traceId, intent, toolCalls: [], retrievalUsed: false, groundingUsed: false, verificationStatus: "controlled_failure", verificationFailures: ["RETRIEVAL_FAILURE"], repairAttempted: false, answerStatus: "controlled_failure", error: "RETRIEVAL_FAILURE" });
        return this.failure(traceId, "Current information requires retrieval, but no compliant retrieval service is configured.");
      }
      try {
        const retrieved = await this.dependencies.retrieval.retrieve({ query: minimizePublicRetrievalQuery(request.message), purpose: intent === "source_question" ? "official_rule" : "current_fact", maxSources: 5 });
        for (const item of retrieved) retrievedEvidenceIds.add(item.id);
        evidence.push(...retrieved);
      } catch (error) {
        this.emit({ traceId, intent, toolCalls: ["retrieval"], retrievalUsed: true, groundingUsed: false, verificationStatus: "controlled_failure", verificationFailures: ["RETRIEVAL_FAILURE"], repairAttempted: false, answerStatus: "controlled_failure", error: "RETRIEVAL_FAILURE" });
        return this.failure(traceId, "I couldn’t retrieve reliable current sources, so I won’t guess. Please try again later.", [error instanceof Error ? error.message : "RETRIEVAL_FAILURE"]);
      }
    }
    const context = assembleContext({ intent, ...(request.subject ? { subject: request.subject } : {}), ...(request.topic ? { topic: request.topic } : {}), ...(request.paperId ? { paperId: request.paperId } : {}), evidence });
    if (decisions.paperEvidence && !context.some((item) => item.source === "paper" || item.source === "teacher")) {
      this.emit({ traceId, intent, toolCalls: [], retrievalUsed: false, groundingUsed: false, verificationStatus: "insufficient_evidence", verificationFailures: [], repairAttempted: false, answerStatus: "insufficient_evidence" });
      return { traceId, status: "insufficient_evidence", answer: "I can’t determine that without the relevant paper, student response, or teacher marking. Share the marked work and I’ll separate what was written, what was marked, and what the evidence can explain.", citations: [], verification: { passed: true, repaired: false, failures: [] } };
    }
    const risk = classifyRisk(intent);
    let route: ReturnType<ModelRouter["route"]>;
    try {
      const providerAvailable = await this.dependencies.providerAvailable?.() ?? true;
      route = this.#router.route({ capability: "tutoring", risk, privacyPolicy: "STUDENT_CHAT_STRICT", latencyBudgetMs: risk === "R4" ? 12_000 : 8_000, difficulty: risk === "R4" ? "complex" : risk === "R3" ? "standard" : "simple", multimodal: false }, providerAvailable ? new Set([this.dependencies.provider.id]) : new Set());
    } catch (error) {
      if (!(error instanceof NoCompliantProviderError)) throw error;
      this.emit({ traceId, intent, toolCalls: this.toolNames(decisions), retrievalUsed: decisions.retrieval, groundingUsed: retrievedEvidenceIds.size > 0, verificationStatus: "controlled_failure", verificationFailures: ["NO_COMPLIANT_PROVIDER"], repairAttempted: false, answerStatus: "controlled_failure", error: "NO_COMPLIANT_PROVIDER" });
      return this.failure(traceId, "No privacy-compliant model endpoint is currently available. Your student data was not sent.");
    }
    const compiled = await promptRegistry.compile(promptIdFor(intent));
    let modelResponse: ModelResponse;
    try {
      modelResponse = await this.dependencies.provider.generate({ model: route.model, system: compiled.system, task: compiled.task, user: request.message, schema: compiled.schema, thinkingLevel: route.thinkingLevel, evidence: context, timeoutMs: route.timeoutMs });
    } catch (error) {
      const code = error instanceof ProviderError ? error.code : "MODEL_FAILURE";
      this.emit({ traceId, intent, provider: route.provider, requestedModel: route.model, thinkingLevel: route.thinkingLevel, promptId: compiled.id, promptHash: compiled.promptHash, schemaId: compiled.schemaId, schemaHash: compiled.schemaHash, toolCalls: this.toolNames(decisions), retrievalUsed: decisions.retrieval, groundingUsed: retrievedEvidenceIds.size > 0, verificationStatus: "controlled_failure", verificationFailures: [code], repairAttempted: false, answerStatus: "controlled_failure", error: code, evidence: context });
      return this.failure(traceId, "The reasoning service is temporarily unavailable. No unverified answer was shown.", [code]);
    }
    let totalLatencyMs = modelResponse.latencyMs;
    let inputTokens = modelResponse.usage.inputTokens ?? 0;
    let outputTokens = modelResponse.usage.outputTokens ?? 0;
    let reasoning: ReasoningResult = { status: "insufficient_evidence", intent, claims: [], conceptIds: [], teachingStrategy: "direct", uncertaintyReason: "Invalid structured model output." };
    let failures: string[] = [];
    try { reasoning = { ...parseSchema(ReasoningResultSchema, modelResponse.output), intent }; }
    catch { failures.push("INVALID_SCHEMA: reasoning output failed canonical validation"); }
    failures.push(...this.verify(reasoning, context, decisions));
    if (failures.length === 0 && this.requiresModelVerifier(risk)) {
      const verification = await this.modelVerify(request, reasoning, context, route);
      totalLatencyMs += verification.latencyMs;
      inputTokens += verification.inputTokens;
      outputTokens += verification.outputTokens;
      failures.push(...verification.failures);
    }
    let repaired = false;
    if (failures.length > 0) {
      repaired = true;
      const repair = await promptRegistry.compile("tutor.repair.v1");
      try {
        modelResponse = await this.dependencies.provider.generate({ model: route.model, system: repair.system, task: `${repair.task}\n\nVERIFICATION FAILURES:\n${failures.join("\n")}`, user: request.message, schema: repair.schema, thinkingLevel: route.thinkingLevel, evidence: context, timeoutMs: route.timeoutMs });
      } catch (error) {
        const code = error instanceof ProviderError ? error.code : "MODEL_FAILURE";
        this.emit({ traceId, intent, provider: route.provider, requestedModel: modelResponse.requestedModel, servedModel: modelResponse.servedModel, thinkingLevel: route.thinkingLevel, promptId: compiled.id, promptHash: compiled.promptHash, schemaId: compiled.schemaId, schemaHash: compiled.schemaHash, toolCalls: this.toolNames(decisions), retrievalUsed: decisions.retrieval, groundingUsed: retrievedEvidenceIds.size > 0, verificationStatus: "failed", verificationFailures: [...failures, code], repairAttempted: true, answerStatus: "controlled_failure", latencyMs: totalLatencyMs, error: code, evidence: context, claims: reasoning.claims });
        return this.failure(traceId, "The answer could not be repaired safely, so it was withheld.", [code], true);
      }
      totalLatencyMs += modelResponse.latencyMs;
      inputTokens += modelResponse.usage.inputTokens ?? 0;
      outputTokens += modelResponse.usage.outputTokens ?? 0;
      try {
        reasoning = { ...parseSchema(ReasoningResultSchema, modelResponse.output), intent };
        failures = this.verify(reasoning, context, decisions);
      } catch {
        failures = ["INVALID_SCHEMA: repaired output failed canonical validation"];
      }
      if (failures.length === 0 && this.requiresModelVerifier(risk)) {
        const verification = await this.modelVerify(request, reasoning, context, route);
        totalLatencyMs += verification.latencyMs;
        inputTokens += verification.inputTokens;
        outputTokens += verification.outputTokens;
        failures.push(...verification.failures);
      }
    }
    if (failures.length > 0) {
      this.emit({ traceId, intent, provider: route.provider, requestedModel: modelResponse.requestedModel, servedModel: modelResponse.servedModel, thinkingLevel: route.thinkingLevel, promptId: compiled.id, promptHash: compiled.promptHash, schemaId: compiled.schemaId, schemaHash: compiled.schemaHash, toolCalls: this.toolNames(decisions), retrievalUsed: decisions.retrieval, groundingUsed: retrievedEvidenceIds.size > 0, verificationStatus: "failed", verificationFailures: failures, repairAttempted: repaired, answerStatus: "controlled_failure", latencyMs: totalLatencyMs, inputTokens, outputTokens, transportSuccess: true, schemaSuccess: !failures.some((failure) => failure.includes("INVALID_SCHEMA")), semanticValidationSuccess: false, error: failures.join("; "), evidence: context, claims: reasoning.claims });
      return this.failure(traceId, "I don't have enough reliable information to answer that confidently.", failures, repaired);
    }
    const verifiedClaims = verifyClaims(reasoning.claims, context).claims;
    const verifiedReasoning: ReasoningResult = { ...reasoning, claims: verifiedClaims };
    const citations = context.filter((item) => retrievedEvidenceIds.has(item.id) && item.provenance.url).map((item) => ({ title: typeof item.value === "object" && item.value !== null && "title" in item.value ? String(item.value.title) : item.provenance.url ?? "Source", url: item.provenance.url ?? "" }));
    this.emit({ traceId, intent, provider: route.provider, requestedModel: modelResponse.requestedModel, servedModel: modelResponse.servedModel, thinkingLevel: route.thinkingLevel, promptId: compiled.id, promptHash: compiled.promptHash, schemaId: compiled.schemaId, schemaHash: compiled.schemaHash, toolCalls: this.toolNames(decisions), retrievalUsed: decisions.retrieval, groundingUsed: retrievedEvidenceIds.size > 0, verificationStatus: "verified", verificationFailures: [], repairAttempted: repaired, answerStatus: verifiedReasoning.status, latencyMs: totalLatencyMs, inputTokens, outputTokens, transportSuccess: true, schemaSuccess: true, semanticValidationSuccess: true, evidence: context, claims: verifiedClaims });
    return { traceId, status: verifiedReasoning.status, answer: renderReasoning(verifiedReasoning, resolveTutorDepth(request)), citations, verification: { passed: true, repaired, failures: [] } };
  }

  private verify(result: ReasoningResult, evidence: readonly Evidence[], decisions: ReturnType<typeof decideTools>): string[] {
    const report = verifyClaims(result.claims, evidence);
    const failures = [...report.failures, ...detectContradictions(result.claims, evidence), ...detectEvidenceConflicts(evidence)];
    if (decisions.retrieval && !evidence.some((item) => item.source === "retrieval" || item.source === "official_source")) failures.push("required retrieval evidence is absent");
    if (decisions.calculator && result.claims.some((claim) => claim.type === "calculation") && !evidence.some((item) => item.source === "tool" && item.verification === "verified")) failures.push("required calculation tool was bypassed");
    if (decisions.units && !evidence.some((item) => item.provenance.toolId === "axon.units.v1")) failures.push("required units tool was unavailable");
    if (decisions.chemistry && !evidence.some((item) => item.provenance.toolId === "axon.chem.v1")) failures.push("required chemistry tool was unavailable");
    failures.push(...hintPolicyFailures(result));
    return [...new Set(failures)];
  }

  private failure(traceId: string, answer: string, failures: string[] = [], repaired = false): TutorResponse {
    return { traceId, status: "controlled_failure", answer, citations: [], verification: { passed: false, repaired, failures } };
  }

  private toolNames(decisions: ReturnType<typeof decideTools>): string[] {
    return Object.entries(decisions).filter(([, enabled]) => enabled).map(([name]) => name);
  }

  private emit(trace: TutorRunTrace): void { this.dependencies.trace?.(trace); }

  private requiresModelVerifier(risk: string): boolean { return risk === "R2" || risk === "R3" || risk === "R4"; }

  private async modelVerify(request: TutorRequest, reasoning: ReasoningResult, evidence: readonly Evidence[], route: ReturnType<ModelRouter["route"]>): Promise<{ failures: string[]; latencyMs: number; inputTokens: number; outputTokens: number }> {
    const verifier = await promptRegistry.compile("tutor.verifier.v2");
    try {
      const response = await this.dependencies.provider.generate({
        model: route.model, system: verifier.system, task: verifier.task,
        user: `<ORIGINAL_REQUEST>\n${request.message}\n</ORIGINAL_REQUEST>\n<PROPOSED_REASONING>\n${JSON.stringify(reasoning)}\n</PROPOSED_REASONING>`,
        schema: verifier.schema, thinkingLevel: route.thinkingLevel, evidence, timeoutMs: route.timeoutMs
      });
      const result = parseSchema(VerificationResultSchema, response.output);
      return { failures: result.passed ? [] : result.failures.map((failure) => `${failure.code}${failure.claimId ? ` ${failure.claimId}` : ""}: ${failure.detail}`), latencyMs: response.latencyMs, inputTokens: response.usage.inputTokens ?? 0, outputTokens: response.usage.outputTokens ?? 0 };
    } catch (error) {
      const code = error instanceof ProviderError ? error.code : "VERIFICATION_FAILED";
      return { failures: [`${code}: verifier did not produce a trustworthy verdict`], latencyMs: 0, inputTokens: 0, outputTokens: 0 };
    }
  }
}

export function hintPolicyFailures(result: ReasoningResult): string[] {
  if (result.intent !== "hint") return [];
  const failures: string[] = [];
  if (result.teachingStrategy !== "socratic" && result.teachingStrategy !== "step_by_step") failures.push("hint response used a full-answer teaching strategy");
  if (result.claims.some((claim) => /\b(?:final answer|answer is|solution is|x\s*=\s*-?\d)/i.test(claim.text))) failures.push("hint response revealed a complete solution");
  return failures;
}
