import { serviceClient } from "@mastery/shared/http.js";
import { deleteObject, deletePrefix } from "@mastery/shared/r2.js";
import type { Env } from "@mastery/shared/env.js";
import { deliverCostAlerts } from "@mastery/shared/cost_alerts.js";
import { runTutorPurges } from "@mastery/shared/tutor_purge.js";
import { maintenanceStore } from "@mastery/shared/maintenance_store.js";
import { queueTopicTagWork } from "@mastery/shared/topic_tag.js";
import { chunkedSendBatch } from "@mastery/shared/chunked_send.js";
import { dispatchStageOutbox, type OutboxEvent } from "@mastery/shared/stage_outbox.js";
import { resendMailer, runParentDigests, type ClaimedDigest } from "@mastery/shared/parent_digest.js";

const KEYS_PER_TICK = 200;
const CLAIMS_PER_TICK = 20;
/** The one-minute cron only dispatches the stage outbox (AXO-225). Must match wrangler.toml. */
const OUTBOX_CRON = "* * * * *";
/** Sunday 12:30 UTC (18:00 IST): the weekly parent digests (AXO-263), and nothing else. Must match wrangler.toml. */
const DIGEST_CRON = "30 12 * * 0";

/**
 * AXO-263: queue, claim and send the weekly parent digests. The database decides who is owed one
 * and re-checks consent at claim time; this only renders and sends. Unconfigured, it queues and
 * claims nothing and says so in the log. Never logs an address, a name or a body.
 */
async function sendParentDigests(env: Env): Promise<void> {
  const sb = serviceClient(env);
  const mailer =
    env.RESEND_API_KEY && env.DIGEST_FROM_ADDRESS ? resendMailer(env.RESEND_API_KEY, env.DIGEST_FROM_ADDRESS) : null;
  const run = await runParentDigests(
    {
      async queue() {
        const { data, error } = await sb.rpc("queue_parent_digests");
        if (error) throw new Error(`queue_parent_digests: ${error.message}`);
        return Number(data ?? 0);
      },
      async claim(limit) {
        const { data, error } = await sb.rpc("claim_parent_digests", { p_limit: limit });
        if (error) throw new Error(`claim_parent_digests: ${error.message}`);
        return (data ?? []) as ClaimedDigest[];
      },
      async finish(id, sent, reason) {
        const { error } = await sb.rpc("finish_parent_digest", { p_id: id, p_sent: sent, p_error: reason });
        if (error) throw new Error(`finish_parent_digest: ${error.message}`);
      },
    },
    { mailer, appUrl: env.DIGEST_APP_URL ?? null, enabled: env.DIGEST_SEND_ENABLED === "true" },
  );
  if (!run.configured) console.error("parent digest: not configured, nothing queued or sent:", run.reason);
  else console.info("parent digest", { queued: run.queued, claimed: run.claimed, sent: run.sent, failed: run.failed });
}

/**
 * AXO-225: send the stage messages the database wrote when it re-drove a stalled
 * run (private.stage_outbox). Every minute, so a re-driven run starts again within
 * a minute of the pg_cron sweep that re-drove it.
 */
async function dispatchOutbox(env: Env): Promise<void> {
  if (!env.STRUCTURE_QUEUE || !env.CONTENT_QUEUE) {
    console.error("stage outbox: STRUCTURE_QUEUE or CONTENT_QUEUE is not bound; nothing dispatched");
    return;
  }
  const sb = serviceClient(env);
  const queues = { structure: env.STRUCTURE_QUEUE, content: env.CONTENT_QUEUE };
  const result = await dispatchStageOutbox({
    async claim(limit) {
      const { data, error } = await sb.rpc("claim_stage_outbox", { p_limit: limit });
      if (error) throw new Error(`claim_stage_outbox: ${error.message}`);
      return (data ?? []) as OutboxEvent[];
    },
    async finish(delivered, failed, reason) {
      const { error } = await sb.rpc("finish_stage_outbox", { p_delivered: delivered, p_failed: failed, p_error: reason });
      // Not fatal: an unfinished claim expires in two minutes and is sent again.
      if (error) console.error("finish_stage_outbox failed", error.message);
    },
    async send(stage, messages) {
      await chunkedSendBatch(queues[stage], messages, (body) => ({ body }));
    },
  });
  if (result.delivered || result.failed) console.info("stage outbox", result);
}


export default {
  async scheduled(controller: ScheduledController, env: Env) {
    if (controller.cron === OUTBOX_CRON) {
      try {
        await dispatchOutbox(env);
      } catch (cause) {
        console.error("stage outbox dispatch failed", String(cause).slice(0, 200));
      }
      return;
    }

    if (controller.cron === DIGEST_CRON) {
      try {
        await sendParentDigests(env);
      } catch (cause) {
        console.error("parent digest run failed", String(cause).slice(0, 200));
      }
      return;
    }

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
