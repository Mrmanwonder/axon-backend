import { mustRpc } from "./db.js";

/** The database checks the run's current phase under the same paper lock as retakes/commits. */
export async function pipelineWrite(sb: any, runId: string, stage: string, args: Record<string, unknown> = {}): Promise<boolean> {
  const result = await mustRpc(sb.rpc("pipeline_write", { p_run_id: runId, p_stage: stage, p_args: args }), "pipeline_write(" + stage + ")") as { applied?: boolean } | null;
  if (typeof result?.applied !== "boolean") throw new Error("Pipeline fence returned no durable result");
  return result.applied;
}
