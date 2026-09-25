export type RolloutStage = "BENCHMARK" | "INTERNAL" | "ONE_PERCENT" | "FIVE_PERCENT" | "TWENTY_FIVE_PERCENT" | "FIFTY_PERCENT" | "FULL" | "HALTED";
const ORDER: readonly RolloutStage[] = ["BENCHMARK", "INTERNAL", "ONE_PERCENT", "FIVE_PERCENT", "TWENTY_FIVE_PERCENT", "FIFTY_PERCENT", "FULL"];
export interface CanarySignals { correctionRateDelta: number; markAttributionDelta: number; unsupportedClaimEscapeDelta: number; latencyP95Delta: number; providerFailureDelta: number; costDelta: number }

export function nextRolloutStage(current: RolloutStage, signals: CanarySignals): RolloutStage {
  if (current === "HALTED" || current === "FULL") return current;
  const halt = signals.correctionRateDelta > 0.01 || signals.markAttributionDelta < -0.005 || signals.unsupportedClaimEscapeDelta > 0 || signals.latencyP95Delta > 0.2 || signals.providerFailureDelta > 0.02 || signals.costDelta > 0.25;
  if (halt) return "HALTED";
  const index = ORDER.indexOf(current);
  return ORDER[index + 1] ?? "FULL";
}

export interface VersionedRuntimeConfig { revisionId: string; model: string; promptId: string; thinkingLevel: string; toolPolicy: string; verificationPolicy: string }
export function rollback(current: VersionedRuntimeConfig, prior: VersionedRuntimeConfig): VersionedRuntimeConfig {
  if (current.revisionId === prior.revisionId) throw new Error("Rollback target must be a prior immutable revision");
  return structuredClone(prior);
}
