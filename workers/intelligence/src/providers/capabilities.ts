import { Type } from "@sinclair/typebox";
import type { AIProvider } from "./types";
import type { RetrievalService } from "../intelligence/retrieval/types";
import { parseSchema } from "../schemas";
import { readBoundedJsonBody } from "../shared/bounded-json";

export type ProviderCapability = "structured_output" | "thinking" | "timeout_enforcement" | "zero_data_retention" | "image_input" | "pdf_input" | "function_calling" | "native_search" | "code_execution";
export interface CapabilityProbeResult { capability: ProviderCapability; passed: boolean; details: Record<string, unknown> }

export interface ReleaseCapabilityArtifact {
  formatVersion: "axon-capability-probe.v1";
  model: string;
  geminiPassed: boolean;
  tavilyPassed: boolean;
  visionPassed: boolean;
  deploymentSha: string;
  configRevision: string;
  observedAt: string;
}

export interface ReleaseCapabilityProbeResult {
  artifact: ReleaseCapabilityArtifact;
  tutor: CapabilityProbeResult[];
  tavily: CapabilityProbeResult;
  vision: CapabilityProbeResult;
}

const ProbeSchema = Type.Object({ probe: Type.Literal("ok") }, { additionalProperties: false });

interface TutorProviderProbe {
  results: CapabilityProbeResult[];
  requestedModel?: string;
  servedModel?: string;
}

export async function probeTutorProvider(provider: AIProvider, model: string): Promise<TutorProviderProbe> {
  const results: CapabilityProbeResult[] = [
    { capability: "zero_data_retention", passed: provider.id === "gemini-zdr", details: { providerId: provider.id, attestation: "configuration" } },
    { capability: "timeout_enforcement", passed: true, details: { mechanism: "AbortSignal.timeout via shared model client" } },
    { capability: "image_input", passed: false, details: { reason: "Tutor adapter intentionally accepts text only; documents use the strict vision route." } },
    { capability: "pdf_input", passed: false, details: { reason: "Tutor adapter intentionally accepts text only; documents use the strict vision route." } },
    { capability: "function_calling", passed: false, details: { reason: "Academic tools are executed deterministically before model generation." } },
    { capability: "native_search", passed: false, details: { reason: "Current facts use the authority-filtered Tavily adapter." } },
    { capability: "code_execution", passed: false, details: { reason: "Code execution is not allowed in the tutor route." } }
  ];
  if (provider.id !== "gemini-zdr") {
    results.push(
      { capability: "structured_output", passed: false, details: { reason: "Live probe blocked until ZDR is attested." } },
      { capability: "thinking", passed: false, details: { reason: "Live probe blocked until ZDR is attested." } }
    );
    return { results };
  }
  try {
    const response = await provider.generate({
      model, system: "Return only the requested probe object.", task: "Return {\"probe\":\"ok\"}.", user: "synthetic capability probe",
      schema: ProbeSchema, thinkingLevel: "minimal", evidence: [], timeoutMs: 5_000
    });
    parseSchema(ProbeSchema, response.output);
    const modelMatches = response.requestedModel === model && response.servedModel === model;
    const modelDetails = { expectedModel: model, requestedModel: response.requestedModel, servedModel: response.servedModel };
    if (modelMatches) {
      results.push(
        { capability: "structured_output", passed: true, details: modelDetails },
        { capability: "thinking", passed: true, details: { ...modelDetails, requestedLevel: "minimal" } }
      );
    } else {
      results.push(
        { capability: "structured_output", passed: false, details: { ...modelDetails, reason: "Requested or served model did not match the certified model." } },
        { capability: "thinking", passed: false, details: { ...modelDetails, requestedLevel: "minimal", reason: "Thinking was exercised on a non-certified model." } }
      );
    }
    return { results, requestedModel: response.requestedModel, servedModel: response.servedModel };
  } catch (error) {
    const reason = error instanceof Error ? error.message : "Unknown probe failure";
    results.push(
      { capability: "structured_output", passed: false, details: { reason } },
      { capability: "thinking", passed: false, details: { reason } }
    );
    return { results };
  }
}

export async function persistCapabilityProbe(db: D1Database, provider: string, model: string, results: readonly CapabilityProbeResult[], provenance: { deploymentSha: string; configRevision: string }): Promise<void> {
  const probedAt = new Date().toISOString();
  await db.batch(results.map((result) => db.prepare("INSERT INTO capability_probe (id, provider, model, capability, passed, details_json, probed_at, deployment_sha, config_revision) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)")
    .bind(crypto.randomUUID(), provider, model, result.capability, result.passed ? 1 : 0, JSON.stringify(result.details), probedAt, provenance.deploymentSha, provenance.configRevision)));
}

async function probeTavily(retrieval?: RetrievalService): Promise<CapabilityProbeResult> {
  if (!retrieval) return { capability: "native_search", passed: false, details: { reason: "Tavily is not configured." } };
  try {
    const evidence = await retrieval.retrieve({
      query: "AQA official GCSE assessment objectives",
      purpose: "official_rule",
      preferredDomains: ["aqa.org.uk"],
      maxSources: 1
    });
    const authoritative = evidence.find((item) => item.authority === "primary" && item.verification === "verified" && item.provenance.url?.startsWith("https://"));
    return authoritative
      ? { capability: "native_search", passed: true, details: { evidenceCount: evidence.length, authority: authoritative.authority } }
      : { capability: "native_search", passed: false, details: { reason: "No authoritative verified evidence was returned." } };
  } catch (error) {
    return { capability: "native_search", passed: false, details: { reason: error instanceof Error ? error.message : "Unknown Tavily probe failure" } };
  }
}

type ServiceFetcher = Pick<Fetcher, "fetch">;

async function probeVision(service: ServiceFetcher, privacyAttested: boolean): Promise<CapabilityProbeResult> {
  if (!privacyAttested) return { capability: "image_input", passed: false, details: { reason: "Vision privacy is not attested." } };
  try {
    const response = await service.fetch("https://axon-document-vision/v1/probe", {
      method: "POST",
      headers: { "x-axon-contract-version": "axon-document-vision.v1" }
    });
    const payload = await readBoundedJsonBody<{ status?: string; contractVersion?: string; version?: string; regionCount?: number; readGroupCount?: number; readerCount?: number }>(response.body, 100_000);
    const passed = response.ok && payload.status === "passed" && payload.contractVersion === "axon-document-vision.v1" &&
      (payload.regionCount ?? 0) > 0 && (payload.readGroupCount ?? 0) > 0 && (payload.readerCount ?? 0) >= 2;
    return {
      capability: "image_input",
      passed,
      details: passed
        ? { contractVersion: payload.contractVersion, version: payload.version, regionCount: payload.regionCount, readGroupCount: payload.readGroupCount, readerCount: payload.readerCount }
        : { reason: "Vision probe did not prove a live staged read by two distinct readers.", responseStatus: response.status, contractVersion: payload.contractVersion, regionCount: payload.regionCount, readGroupCount: payload.readGroupCount, readerCount: payload.readerCount }
    };
  } catch (error) {
    return { capability: "image_input", passed: false, details: { reason: error instanceof Error ? error.message : "Unknown vision probe failure" } };
  }
}

export async function probeReleaseCapabilities(input: {
  provider: AIProvider;
  model: string;
  retrieval?: RetrievalService;
  visionService: ServiceFetcher;
  visionPrivacyAttested: boolean;
  deploymentSha: string;
  configRevision: string;
  now?: () => Date;
}): Promise<ReleaseCapabilityProbeResult> {
  const [tutorProbe, tavily, vision] = await Promise.all([
    probeTutorProvider(input.provider, input.model),
    probeTavily(input.retrieval),
    probeVision(input.visionService, input.visionPrivacyAttested)
  ]);
  const tutor = tutorProbe.results;
  const requiredTutorCapabilities = new Set<ProviderCapability>(["zero_data_retention", "timeout_enforcement", "structured_output", "thinking"]);
  const geminiPassed = [...requiredTutorCapabilities].every((capability) => tutor.some((result) => result.capability === capability && result.passed));
  return {
    artifact: {
      formatVersion: "axon-capability-probe.v1",
      model: tutorProbe.servedModel ?? tutorProbe.requestedModel ?? input.model,
      geminiPassed,
      tavilyPassed: tavily.passed,
      visionPassed: vision.passed,
      deploymentSha: input.deploymentSha,
      configRevision: input.configRevision,
      observedAt: (input.now ?? (() => new Date()))().toISOString()
    },
    tutor,
    tavily,
    vision
  };
}
