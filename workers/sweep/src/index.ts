import { serviceClient } from "@mastery/shared/http.js";
import { deleteObject, deletePrefix, type BucketKind } from "@mastery/shared/r2.js";
import type { Env } from "@mastery/shared/env.js";
import { deliverCostAlerts, type CostAlertRow } from "@mastery/shared/cost_alerts.js";
import { runTutorPurges } from "@mastery/shared/tutor_purge.js";
import { queueTopicTagWork } from "@mastery/shared/topic_tag.js";

const KEYS_PER_TICK = 200;
const CLAIMS_PER_TICK = 20;

interface DeletionClaim {
  id: string;
  bucket: BucketKind;
  key: string | null;
  prefix: string | null;
}

export default {
  async scheduled(_controller: ScheduledController, env: Env) {
    const sb = serviceClient(env);

    // Stuck-run recovery executes in pg_cron; see
    // db/operations/enable-stuck-run-recovery.sql. The function is private,
    // so calling sb.rpc("sweep_stuck_runs") through the public API cannot work.

    // Spend alerts are detected in the database (pg_cron); this only delivers them. It runs first
    // so a failing deletion queue cannot hide a spend alert.
    try {
      await deliverCostAlerts(
        {
          async pending(limit) {
            const { data } = await sb.rpc("pending_cost_alerts", { p_limit: limit });
            return (data ?? []) as CostAlertRow[];
          },
          async markDelivered(id, error) {
            await sb
              .from("cost_alert")
              .update(error ? { delivery_error: error } : { delivered_at: new Date().toISOString(), delivery_error: null })
              .eq("id", id);
          },
        },
        env.ALERT_WEBHOOK_URL,
        async (url, body) => {
          const res = await fetch(url, { method: "POST", headers: { "content-type": "application/json" }, body });
          return { ok: res.ok, status: res.status };
        },
      );
    } catch (cause) {
      console.error("cost alert delivery failed", String(cause).slice(0, 200));
    }

    // Deletion parity for the Tutor's own provenance rows (AXO-126). Also before the R2 queue.
    try {
      await runTutorPurges(
        {
          async claim(limit) {
            const { data } = await sb.rpc("claim_tutor_purges", { p_limit: limit });
            return (data ?? []) as Array<{ id: number; paper_id: string }>;
          },
          async finish(id, error) {
            await sb.rpc("finish_tutor_purge", { p_id: id, p_error: error });
          },
        },
        env.INTELLIGENCE && env.AXON_ADMIN_TOKEN
          ? async (paperIds) => {
              const res = await env.INTELLIGENCE!.fetch("https://axon-intelligence.internal/v1/admin/purge", {
                method: "POST",
                headers: { "content-type": "application/json", authorization: `Bearer ${env.AXON_ADMIN_TOKEN}` },
                body: JSON.stringify({ paperIds }),
              });
              return { ok: res.ok, status: res.status };
            }
          : undefined,
      );
    } catch (cause) {
      console.error("tutor purge failed", String(cause).slice(0, 200));
    }

    // Syllabus topic tags for the heatmap: any committed question on a paper
    // with a subject and a loaded syllabus that has not been tagged against it.
    if (env.EXPLAIN_QUEUE) {
      try {
        const queued = await queueTopicTagWork({
          async claim(limit) {
            const { data, error } = await sb.rpc("claim_topic_tag_work", { p_limit: limit });
            if (error) throw new Error(error.message);
            return (data ?? []) as Array<{ region_id: string; document_id: string }>;
          },
          async send(messages) {
            await env.EXPLAIN_QUEUE!.sendBatch(messages.map((body) => ({ body })));
          },
        });
        if (queued) console.info("topic tags queued", queued);
      } catch (cause) {
        console.error("topic tag queueing failed", String(cause).slice(0, 200));
      }
    }

    const { data: claims, error: claimError } = await sb.rpc("claim_deletions", { p_limit: CLAIMS_PER_TICK });
    if (claimError) {
      console.error("claim_deletions failed", claimError.message);
      return;
    }

    for (const claim of (claims ?? []) as DeletionClaim[]) {
      try {
        if (claim.key) {
          await deleteObject(env, claim.bucket, claim.key);
          await sb.rpc("finish_deletion", { p_id: claim.id });
        } else if (claim.prefix) {
          const walk = await deletePrefix(env, claim.bucket, claim.prefix, { maxKeys: KEYS_PER_TICK });
          if (walk.done) {
            await sb.rpc("finish_deletion", { p_id: claim.id });
          } else {
            await sb.rpc("finish_deletion", { p_id: claim.id, p_error: `${walk.deleted} deleted, more to go` });
          }
        }
      } catch (cause) {
        await sb.rpc("finish_deletion", { p_id: claim.id, p_error: String(cause).slice(0, 500) });
      }
    }
  },
} satisfies ExportedHandler<Env>;
