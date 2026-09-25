export interface ModelRates { inputUsdPerMillion: number; outputUsdPerMillion: number }

export function configuredModelRates(input?: string, output?: string): ModelRates | undefined {
  const inputRate = Number(input);
  const outputRate = Number(output);
  if (input === undefined || output === undefined || !Number.isFinite(inputRate) || !Number.isFinite(outputRate) || inputRate < 0 || outputRate < 0) return undefined;
  return { inputUsdPerMillion: inputRate, outputUsdPerMillion: outputRate };
}

export function estimateModelCost(inputTokens: number, outputTokens: number, rates: ModelRates): number {
  if (!Number.isFinite(inputTokens) || !Number.isFinite(outputTokens) || inputTokens < 0 || outputTokens < 0) throw new Error("Token counts must be non-negative finite numbers");
  return inputTokens / 1_000_000 * rates.inputUsdPerMillion + outputTokens / 1_000_000 * rates.outputUsdPerMillion;
}
