import { serviceClient } from "@mastery/shared/http.js";
import { deleteObject, deletePrefix } from "@mastery/shared/r2.js";
import type { Env } from "@mastery/shared/env.js";
import { deliverCostAlerts } from "@mastery/shared/cost_alerts.js";
import { runTutorPurges } from "@mastery/shared/tutor_purge.js";
import { maintenanceStore } from "@mastery/shared/maintenance_store.js";
import { queueTopicTagWork } from "@mastery/shared/topic_tag.js";
import { chunkedSendBatch } from "@mastery/shared/chunked_send.js";

const KEYS_PER_TICK = 200;
const CLAIMS_PER_TICK = 20;


export default {
  async scheduled(_controller: ScheduledController, env: Env) {
    const sb = serviceClient(env);
    const maintenance = maintenanceStore(sb);

    // Stuck-run recovery executes in pg_cron; see
    // db/operations/enable-stuck-run-recovery.sql. The function is private,
    // so calling sb.rpc("sweep_stuck_runs") through the public API cannot work.

    // Spend alerts are detected in the database (pg_cron); this only delivers them. It runs first
    // so a failing deletion queue cannot hide a spend alert.
    try {
      await deliverCostAlerts(
        maintenance.alerts,
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
        maintenance.tutorPurges,
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
            // ⚡ Bolt: Use chunkedSendBatch for concurrent dispatch to avoid N+1 bottleneck and honor batch limits
            await chunkedSendBatch(env.EXPLAIN_QUEUE!, messages, (body) => ({ body }));
          },
        });
        if (queued) console.info("topic tags queued", queued);
      } catch (cause) {
        console.error("topic tag queueing failed", String(cause).slice(0, 200));
      }
    }

    const claims = await maintenance.claimDeletions(CLAIMS_PER_TICK);

    for (const claim of claims) {
      try {
        if (claim.key) {
          await deleteObject(env, claim.bucket, claim.key);
          await maintenance.finishDeletion(claim.id);
        } else if (claim.prefix) {
          const walk = await deletePrefix(env, claim.bucket, claim.prefix, { maxKeys: KEYS_PER_TICK });
          if (walk.done) {
            await maintenance.finishDeletion(claim.id);
          } else {
            await maintenance.finishDeletion(claim.id, `${walk.deleted} deleted, more to go`);
          }
        }
      } catch (cause) {
        await maintenance.finishDeletion(claim.id, String(cause).slice(0, 500));
      }
    }
  },
} satisfies ExportedHandler<Env>;
