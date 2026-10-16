import type { SupabaseClient } from "@supabase/supabase-js";
import { mustAffectRows, mustRpc } from "./db.js";

export type ExplanationSkipReason = "no_marks_lost" | "insufficient_evidence";

/** Preserve the reason for an honest absence before advancing the run.
    This never creates a loss event, changes a mark, or claims an explanation. */
export async function finishSkippedExplanation(
  sb: SupabaseClient, regionId: string, runId: string, reason: ExplanationSkipReason,
): Promise<void> {
  await mustAffectRows(
    sb.from("question_region").update({
      explain_status: "skipped",
      explain_failure_reason: reason,
      explain_status_at: new Date().toISOString(),
    }).eq("id", regionId).select("id"),
    "record explanation skip",
  );
  await mustRpc(sb.rpc("advance_after_explain", { p_run_id: runId }), "advance_after_explain");
}
