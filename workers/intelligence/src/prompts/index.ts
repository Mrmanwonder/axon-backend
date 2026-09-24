import { ReasoningResultSchema, VerificationResultSchema } from "../schemas";
import { AXON_KERNEL_V3, TASK_CONTRACTS } from "./kernel.v3";
import { PromptRegistry } from "./registry";
import { RUNTIME_CONFIG_V3 } from "../config/runtime.v3";

export const promptRegistry = new PromptRegistry();

for (const [id, task] of Object.entries(TASK_CONTRACTS)) {
  const verifier = id === "tutor.verifier.v2";
  promptRegistry.register({ id, system: AXON_KERNEL_V3, task, schemaId: verifier ? "verification-result.v2" : "reasoning-result.v3", schema: verifier ? VerificationResultSchema : ReasoningResultSchema });
}

promptRegistry.validateRoutes(Object.keys(TASK_CONTRACTS));
promptRegistry.validateRoutes(Object.values(RUNTIME_CONFIG_V3.routes));

export { PromptRegistry } from "./registry";
export type { CompiledPrompt, PromptArtifact } from "./registry";
