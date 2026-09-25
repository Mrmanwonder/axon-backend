import type { Capability, Difficulty, RiskLevel, ThinkingLevel } from "../intelligence/routing/router";

export const RUNTIME_CONFIG_V3 = Object.freeze({
  revisionId: "v3.default",
  pipelineVersion: "3.0.0",
  // The release contract names the requested model explicitly. Supabase still
  // owns the live route used by callModel(), but this expected value lets the
  // capability probe and circuit breaker fail closed if that route drifts.
  primaryModel: "gemini-3.5-flash-lite",
  thinking: {
    classification: "minimal", simple: "low", standard: "medium", complex: "high",
    verification: "medium", adjudication: "high"
  } satisfies Record<string, ThinkingLevel>,
  timeouts: { defaultMs: 8_000, criticalMs: 12_000 },
  verification: { maximumRepairCycles: 1, mandatoryGroundingRisks: ["R4"] as readonly RiskLevel[] },
  routes: {
    classification: "tutor.explain_concept.v2",
    document_reading: "tutor.paper_explanation.v3",
    tutoring: "tutor.explain_concept.v2",
    verification: "tutor.verifier.v2",
    adjudication: "tutor.paper_explanation.v3"
  } satisfies Record<Capability, string>
});

export function configuredThinking(capability: Capability, difficulty: Difficulty, risk: RiskLevel): ThinkingLevel {
  if (capability === "classification") return RUNTIME_CONFIG_V3.thinking.classification;
  if (capability === "adjudication") return RUNTIME_CONFIG_V3.thinking.adjudication;
  if (capability === "verification") return RUNTIME_CONFIG_V3.thinking.verification;
  if (risk === "R4" || difficulty === "complex") return RUNTIME_CONFIG_V3.thinking.complex;
  return difficulty === "standard" ? RUNTIME_CONFIG_V3.thinking.standard : RUNTIME_CONFIG_V3.thinking.simple;
}
