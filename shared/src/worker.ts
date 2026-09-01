import type { SupabaseClient } from "@supabase/supabase-js";
import { serviceClient } from "./http.js";
import { ModelError } from "./openrouter.js";
import type { Env } from "./env.js";

export function isRetryable(error: unknown): boolean {
  if (error instanceof ModelError) return error.retryable;
  return false;
}

export async function failRun(sb: SupabaseClient, runId: string | null | undefined, reason: string): Promise<void> {
  if (!runId) return;
  await sb.rpc("run_advance", { p_run_id: runId, p_to: "failed", p_reason: reason });
}

// Generous: the in-process transient retry in callModel can itself take
// several seconds per attempt, and this budget has to cover it plus the rest
// of the handler. 100s was too tight once that retry was added — see
// AXON_FIX_BRIEF.md §3.5.
export const HANDLE_TIMEOUT_MS = 200_000;

export function withDeadline<T>(promiseOrValue: T | Promise<T>, ms: number, label: string): Promise<T> {
  let timer: ReturnType<typeof setTimeout>;
  const deadline = new Promise<never>((_, reject) => {
    timer = setTimeout(() => {
      const err = new Error(label + " exceeded " + ms + "ms budget — failing so the queue message does not hang forever");
      err.name = "HandlerTimeout";
      reject(err);
    }, ms);
  });
  return Promise.race([Promise.resolve(promiseOrValue), deadline]).finally(() => clearTimeout(timer));
}

export interface QueueContext<M> {
  env: Env;
  sb: SupabaseClient;
  msg: M;
  attempt: number;
  beat: () => Promise<void>;
}

interface RetryableMessage {
  run_id?: string;
  _retries?: number;
}

/**
 * Wraps a Cloudflare Queue consumer with: a Supabase service client, a
 * run-heartbeat before the handler runs, a handler deadline, and retry
 * logic for transient failures.
 *
 * On a retryable error, this re-enqueues manually via SELF_QUEUE with
 * exponential backoff (capped at 5 attempts) so the failure is visible in
 * `msg._retries` rather than relying on the queue's own opaque retry count.
 * A worker with no SELF_QUEUE binding falls through to the native
 * `message.retry()` — see AXON_FIX_BRIEF.md §4.D2 for where that still
 * applies (mastery-reconcile, until §9.2 adds the binding).
 */
export function consumeQueue<M extends RetryableMessage>(
  handle: (ctx: QueueContext<M>) => Promise<unknown>,
  onPermanent?: (ctx: QueueContext<M>, error: unknown) => Promise<unknown>
) {
  return async (batch: MessageBatch<M>, env: Env) => {
    const sb = serviceClient(env);
    await Promise.all(
      batch.messages.map(async (message) => {
        const msg = message.body;
        const retries = typeof msg._retries === "number" ? msg._retries : 0;
        const ctx: QueueContext<M> = {
          env,
          sb,
          msg,
          attempt: message.attempts,
          beat: async () => {
            if (msg.run_id) await sb.rpc("run_heartbeat", { p_run_id: msg.run_id });
          },
        };
        await ctx.beat();
        try {
          await withDeadline(handle(ctx), HANDLE_TIMEOUT_MS, "handler");
          message.ack();
        } catch (error) {
          if (isRetryable(error) && retries < 5) {
            console.warn("retrying (manual re-enqueue)", String(error), "attempt", retries + 1);
            const selfQueue = env.SELF_QUEUE;
            if (selfQueue) {
              try {
                await selfQueue.send({ ...msg, _retries: retries + 1 }, { delaySeconds: Math.min(60, 5 * Math.pow(2, retries)) });
                message.ack();
                return;
              } catch (sendErr) {
                console.error("manual re-enqueue failed, falling back to native retry", String(sendErr));
              }
            }
            console.warn("native retry fallback", String(error));
            message.retry();
            return;
          }
          console.error("failed permanently", String(error));
          try {
            await withDeadline(onPermanent?.(ctx, error), HANDLE_TIMEOUT_MS, "onPermanent");
          } finally {
            message.ack();
          }
        }
      })
    );
  };
}
