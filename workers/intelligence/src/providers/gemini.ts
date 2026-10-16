import { callModel, ModelError, serviceClient } from "@mastery/shared";
import { parseSchema } from "../schemas";
import { fenceEvidence } from "../intelligence/security/fencing";
import { ProviderError, type AIProvider, type ModelRequest, type ModelResponse } from "./types";

// How the Gemini key in use is attested (GEMINI_PRIVACY_MODE):
//   "zdr"              — a written Google zero-data-retention arrangement covers the key.
//   "paid_no_training" — AI Studio paid tier: Google does not train on the data, and it keeps it
//                        for a bounded period that the Privacy Policy declares (owner decision,
//                        Axon.md §11). Never described as zero retention.
//   anything else      — unverified; student data is not sent.
export type GeminiPrivacyMode = "zdr" | "paid_no_training" | "unverified";

export function geminiPrivacyMode(value: unknown): GeminiPrivacyMode {
  const mode = String(value);
  return mode === "zdr" || mode === "paid_no_training" ? mode : "unverified";
}

const PROVIDER_IDS: Record<GeminiPrivacyMode, string> = {
  zdr: "gemini-zdr",
  paid_no_training: "gemini-paid",
  unverified: "gemini-unverified",
};

// Provider ids that may receive student data: Google does not train on it under either.
export const NO_TRAINING_PROVIDER_IDS: readonly string[] = [PROVIDER_IDS.zdr, PROVIDER_IDS.paid_no_training];

export class GeminiProvider implements AIProvider {
  readonly id: string;
  readonly privacyMode: GeminiPrivacyMode;
  constructor(readonly env: Env, privacyMode: GeminiPrivacyMode) {
    this.privacyMode = privacyMode;
    this.id = PROVIDER_IDS[privacyMode];
  }

  async generate(request: ModelRequest): Promise<ModelResponse> {
    try {
      const response = await callModel({
        env: this.env,
        sb: serviceClient(this.env),
        stage: "tutor",
        system: request.system,
        instruction: `${request.task}\n\n${fenceEvidence(request.evidence)}\n\n<STUDENT_REQUEST>\n${request.user}\n</STUDENT_REQUEST>`,
        schema: {
          name: typeof request.schema.$id === "string" ? request.schema.$id : "axon_tutor_response",
          schema: request.schema
        },
        validate: (value) => parseSchema(request.schema, value),
        timeoutMs: request.timeoutMs,
        thinkingLevel: request.thinkingLevel,
        expectedModel: request.model
      });
      return {
        requestedModel: response.requestedModel,
        servedModel: response.model,
        output: response.parsed,
        usage: {
          ...(response.inputTokens !== null ? { inputTokens: response.inputTokens } : {}),
          ...(response.outputTokens !== null ? { outputTokens: response.outputTokens } : {})
        },
        latencyMs: response.latencyMs
      };
    } catch (error) {
      if (error instanceof ProviderError) throw error;
      if (error instanceof ModelError) {
        const code = error.code === "timeout"
          ? "MODEL_TIMEOUT"
          : error.code === "rate_limited"
            ? "MODEL_RATE_LIMIT"
            : error.code === "provider_error"
              ? "MODEL_SERVER_ERROR"
              : error.code === "bad_shape" || error.code === "empty_response"
                ? "INVALID_SCHEMA"
                : "MODEL_FAILURE";
        throw new ProviderError(code, error.message);
      }
      throw new ProviderError("MODEL_FAILURE", error instanceof Error ? error.message : "Unknown Gemini failure");
    }
  }
}
