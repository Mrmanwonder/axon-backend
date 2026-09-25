import { NoCompliantProviderError, PRIVACY_POLICIES, type PrivacyPolicyId } from "./privacy";
import { configuredThinking, RUNTIME_CONFIG_V3 } from "../../config/runtime.v3";

export type Capability = "classification" | "document_reading" | "tutoring" | "verification" | "adjudication";
export type RiskLevel = "R0" | "R1" | "R2" | "R3" | "R4";
export type Difficulty = "simple" | "standard" | "complex";
export type ThinkingLevel = "minimal" | "low" | "medium" | "high";

export interface RouteRequest {
  capability: Capability;
  risk: RiskLevel;
  privacyPolicy: PrivacyPolicyId;
  latencyBudgetMs: number;
  difficulty: Difficulty;
  multimodal: boolean;
}

export interface ModelRoute {
  provider: string;
  model: string;
  thinkingLevel: ThinkingLevel;
  timeoutMs: number;
  fallbacks: readonly string[];
  promptId: string;
}

export class ModelRouter {
  route(request: RouteRequest, healthyProviders: ReadonlySet<string>): ModelRoute {
    const policy = PRIVACY_POLICIES[request.privacyPolicy];
    const provider = policy.allowedProviders.find((candidate) => healthyProviders.has(candidate));
    if (!provider) throw new NoCompliantProviderError(request.privacyPolicy);
    return {
      provider,
      model: RUNTIME_CONFIG_V3.primaryModel,
      thinkingLevel: configuredThinking(request.capability, request.difficulty, request.risk),
      timeoutMs: Math.min(request.latencyBudgetMs, request.risk === "R4" ? 12_000 : 8_000),
      fallbacks: [],
      promptId: RUNTIME_CONFIG_V3.routes[request.capability]
    };
  }
}

export function classifyRisk(intent: string): RiskLevel {
  if (intent === "casual") return "R0";
  if (intent === "direct_answer" || intent === "concept_explanation") return "R1";
  if (intent === "hint" || intent === "comparison") return "R2";
  if (intent === "problem_solving" || intent === "mistake_diagnosis" || intent === "work_check") return "R3";
  return "R4";
}
