import type { SupabaseClient } from "@supabase/supabase-js";
import { mustAffectRows, mustData, mustRpc } from "./db.js";
import type { AlertSink, CostAlertRow } from "./cost_alerts.js";
import type { PurgeClaim, PurgeSink } from "./tutor_purge.js";
import type { BucketKind } from "./r2.js";

export interface DeletionClaim {
  id: string;
  bucket: BucketKind;
  key: string | null;
  prefix: string | null;
}

/**
 * All maintenance receipts are checked. A PostgREST error is not an empty queue
 * or a successful acknowledgement: callers must retain/retry the durable work.
 * This adapter never changes which objects are eligible for deletion.
 */
export function maintenanceStore(sb: SupabaseClient) {
  const alerts: AlertSink = {
    async pending(limit) {
      return await mustData(
        sb.rpc("pending_cost_alerts", { p_limit: limit }),
        "pending_cost_alerts",
      ) as CostAlertRow[];
    },
    async markDelivered(id, error) {
      await mustAffectRows(
        sb.from("cost_alert")
          .update(error
            ? { delivery_error: error }
            : { delivered_at: new Date().toISOString(), delivery_error: null })
          .eq("id", id)
          .select("id"),
        "cost_alert delivery receipt",
      );
    },
  };
  const tutorPurges: PurgeSink = {
    async claim(limit) {
      return await mustData(
        sb.rpc("claim_tutor_purges", { p_limit: limit }),
        "claim_tutor_purges",
      ) as PurgeClaim[];
    },
    async finish(id, error) {
      await mustRpc(
        sb.rpc("finish_tutor_purge", { p_id: id, p_error: error }),
        "finish_tutor_purge",
      );
    },
  };
  return {
    alerts,
    tutorPurges,
    async claimDeletions(limit: number): Promise<DeletionClaim[]> {
      return await mustData(
        sb.rpc("claim_deletions", { p_limit: limit }),
        "claim_deletions",
      ) as DeletionClaim[];
    },
    async finishDeletion(id: string, error: string | null = null): Promise<void> {
      await mustRpc(
        sb.rpc("finish_deletion", { p_id: id, p_error: error }),
        "finish_deletion",
      );
    },
  };
}
