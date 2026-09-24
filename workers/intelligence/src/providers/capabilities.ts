import { Type } from "@sinclair/typebox";
import type { AIProvider } from "./types";
import { parseSchema } from "../schemas";

export type ProviderCapability = "structured_output" | "thinking" | "timeout_enforcement" | "zero_data_retention" | "image_input" | "pdf_input" | "function_calling" | "native_search" | "code_execution";
export interface CapabilityProbeResult { capability: ProviderCapability; passed: boolean; details: Record<string, unknown> }

const ProbeSchema = Type.Object({ probe: Type.Literal("ok") }, { additionalProperties: false });

export async function probeTutorProvider(provider: AIProvider, model: string): Promise<CapabilityProbeResult[]> {
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
    return results;
  }
  try {
    const response = await provider.generate({
      model, system: "Return only the requested probe object.", task: "Return {\"probe\":\"ok\"}.", user: "synthetic capability probe",
      schema: ProbeSchema, thinkingLevel: "minimal", evidence: [], timeoutMs: 5_000
    });
    parseSchema(ProbeSchema, response.output);
    results.push(
      { capability: "structured_output", passed: true, details: { servedModel: response.servedModel } },
      { capability: "thinking", passed: true, details: { requestedLevel: "minimal", servedModel: response.servedModel } }
    );
  } catch (error) {
    const reason = error instanceof Error ? error.message : "Unknown probe failure";
    results.push(
      { capability: "structured_output", passed: false, details: { reason } },
      { capability: "thinking", passed: false, details: { reason } }
    );
  }
  return results;
}

export async function persistCapabilityProbe(db: D1Database, provider: string, model: string, results: readonly CapabilityProbeResult[]): Promise<void> {
  const probedAt = new Date().toISOString();
  await db.batch(results.map((result) => db.prepare("INSERT INTO capability_probe (id, provider, model, capability, passed, details_json, probed_at) VALUES (?, ?, ?, ?, ?, ?, ?)")
    .bind(crypto.randomUUID(), provider, model, result.capability, result.passed ? 1 : 0, JSON.stringify(result.details), probedAt)));
}
