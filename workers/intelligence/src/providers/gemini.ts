import { callModel, ModelError, serviceClient } from "@mastery/shared";
import { parseSchema } from "../schemas";
import { fenceEvidence } from "../intelligence/security/fencing";
import { ProviderError, type AIProvider, type ModelRequest, type ModelResponse } from "./types";

export class GeminiProvider implements AIProvider {
  readonly id: string;
  constructor(readonly env: Env, privacyMode: "zdr" | "unverified") {
    this.id = privacyMode === "zdr" ? "gemini-zdr" : "gemini-unverified";
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
