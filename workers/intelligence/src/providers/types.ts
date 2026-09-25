import type { TSchema } from "@sinclair/typebox";
import type { Evidence } from "../schemas";
import type { ThinkingLevel } from "../intelligence/routing/router";

export interface ModelRequest {
  model: string;
  system: string;
  task: string;
  user: string;
  schema: TSchema;
  thinkingLevel: ThinkingLevel;
  evidence: readonly Evidence[];
  timeoutMs: number;
}

export interface ModelResponse {
  requestedModel: string;
  servedModel: string;
  output: unknown;
  usage: { inputTokens?: number; outputTokens?: number };
  latencyMs: number;
}

export interface AIProvider {
  readonly id: string;
  generate(request: ModelRequest): Promise<ModelResponse>;
}

export type ProviderFailureCode = "MODEL_TIMEOUT" | "MODEL_RATE_LIMIT" | "MODEL_SERVER_ERROR" | "INVALID_SCHEMA" | "MODEL_FAILURE";
export class ProviderError extends Error {
  constructor(readonly code: ProviderFailureCode, message: string) { super(message); this.name = "ProviderError"; }
}
