import type { SupabaseClient } from "@supabase/supabase-js";
import { serviceClient } from "./http.js";
import type { Env } from "./env.js";
import { isRetryable } from "./errors.js";
import { mustRpc } from "./db.js";

export { isRetryable };

/**
 * Move a run to `failed`, and say so if that write did not land.
 *
 * This used to be a bare `await sb.rpc(...)` whose result was dropped. The
 * Supabase client resolves rather than throws on a database error, so a failed
 * write here looked exactly like a successful one — and the caller went on to
 * acknowledge the queue message. The run stayed mid-pipeline with no message
 * left to drive it: work lost, silently, at the exact moment the system was
 * trying to record that something had gone wrong.
 *
 * It throws now, and `consumeQueue` below is careful about what that means.
 */
export async function failRun(
  sb: SupabaseClient,
  runId: string | null | undefined,
  reason: string,
  /** Stable machine code (see failure_codes.ts). `reason` is user copy; this is for queries and alerts. */
  failureReason?: string,
): Promise<void> {
  if (!runId) return;
  await mustRpc(
    sb.rpc("run_advance", {
      p_run_id: runId,
      p_to: "failed",
      p_reason: reason,
      ...(failureReason ? { p_failure_reason: failureReason } : {}),
    }),
    "run_advance(failed)",
  );
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
  /**
   * True when this delivery is a retry of work that was handed out before:
   * the queue's own redelivery (`attempts > 1`) or this harness's manual
   * re-enqueue (`_retries > 0`). Stage fan-out uses it to tell "my earlier
   * attempt advanced the run and its send may not have landed" apart from "a
   * concurrent peer advanced the run and is sending" (see `shouldFanOut`).
   */
  redelivered: boolean;
  beat: () => Promise<void>;
}

interface RetryableMessage {
  run_id?: string;
  _retries?: number;
}

/** How many manual re-enqueues a message gets before it is treated as permanent. */
export const MAX_MANUAL_RETRIES = 5;

/** The queue-message surface this harness actually uses. */
export interface AckableMessage<M> {
  body: M;
  attempts: number;
  ack: () => void;
  retry: () => void;
}

/**
 * The per-message control flow, separated from the batch loop and the client
 * construction so it can be tested directly.
 *
 * Everything interesting about this harness is the decision of when to ack,
 * and that decision is what the extraction exists to expose: `consumeQueue`
 * builds its Supabase client from `env` inside itself, which left the ack rule
 * reachable only through a real client. A rule that cannot be tested is a rule
 * that drifts.
 *
 * ── When a message may be acknowledged ────────────────────────────────────
 *
 * An ACK is a claim that the queue may forget this work. It is only true when
 * one of these is DURABLE — written, and confirmed written:
 *
 *   1. the work completed;
 *   2. retry work was created (the manual re-enqueue landed);
 *   3. a terminal failure state was recorded.
 *
 * "We called the function that records the terminal state" is not on that
 * list, and the previous version acknowledged on exactly that basis:
 *
 *     try { await onPermanent(ctx, error); } finally { message.ack(); }
 *
 * The `finally` fired whether or not `onPermanent` threw. Combined with a
 * `failRun` that ignored its own error, a Supabase outage produced: handler
 * fails → classified permanent → failRun silently does nothing → ACK. The run
 * sat in a non-terminal state forever and the message that could have moved it
 * was gone.
 *
 * So `onPermanent` failing now means NO ack: the message is retried, and the
 * next delivery gets another chance to record the terminal state. That can
 * re-run `onPermanent`, which is why it must be idempotent — moving a run to
 * `failed` twice is a no-op, and `run_advance` already refuses to resurrect a
 * terminal state.
 *
 * On a retryable error the message is re-enqueued via SELF_QUEUE with
 * exponential backoff, so the attempt count is visible in `msg._retries`
 * rather than in the queue's own opaque counter. A worker with no SELF_QUEUE
 * binding falls through to the native `message.retry()`.
 */
export async function processQueueMessage<M extends RetryableMessage>(
  message: AckableMessage<M>,
  sb: SupabaseClient,
  env: Env,
  handle: (ctx: QueueContext<M>) => Promise<unknown>,
  onPermanent?: (ctx: QueueContext<M>, error: unknown) => Promise<unknown>,
): Promise<void> {
  const msg = message.body;
  const retries = typeof msg._retries === "number" ? msg._retries : 0;
  const ctx: QueueContext<M> = {
    env,
    sb,
    msg,
    attempt: message.attempts,
    redelivered: message.attempts > 1 || retries > 0,
    beat: async () => {
      if (!msg.run_id) return;
      // Checked, because a silent heartbeat failure is what makes the
      // stale-run sweep wrong: the sweeper decides a run is abandoned from a
      // timestamp nobody managed to update. Treated like any other database
      // failure — retry rather than run on blind.
      await mustRpc(sb.rpc("run_heartbeat", { p_run_id: msg.run_id }), "run_heartbeat");
    },
  };

  try {
    // Inside the try now. The heartbeat is a database call like any other, and
    // an unhandled rejection here used to escape the per-message handler
    // entirely rather than going through retry.
    await ctx.beat();
    await withDeadline(handle(ctx), HANDLE_TIMEOUT_MS, "handler");
    message.ack();
  } catch (error) {
    if (isRetryable(error) && retries < MAX_MANUAL_RETRIES) {
      console.warn("retrying (manual re-enqueue)", String(error), "attempt", retries + 1);
      const selfQueue = env.SELF_QUEUE;
      if (selfQueue) {
        try {
          await selfQueue.send(
            { ...msg, _retries: retries + 1 },
            { delaySeconds: Math.min(60, 5 * Math.pow(2, retries)) },
          );
          // The replacement message is durable, so forgetting this one loses
          // nothing.
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
    if (!onPermanent) {
      // Nothing to record, so there is nothing that can fail to record.
      message.ack();
      return;
    }
    try {
      await withDeadline(onPermanent(ctx, error), HANDLE_TIMEOUT_MS, "onPermanent");
      message.ack();
    } catch (terminalError) {
      // The terminal state was NOT durably written. Acknowledging here would
      // lose the only thing that could still move this run.
      console.error(
        "could not record permanent failure — retrying rather than acknowledging",
        String(terminalError),
      );
      message.retry();
    }
  }
}

/**
 * Run `fn` over `items` with at most `limit` in flight at once, settling every
 * item (one failure never stops the others). Returns the settled results in
 * input order. No dependency: a dozen lines do what p-limit would.
 */
export async function settleWithConcurrency<T, R>(
  items: readonly T[],
  limit: number,
  fn: (item: T, index: number) => Promise<R>,
): Promise<PromiseSettledResult<R>[]> {
  const results: PromiseSettledResult<R>[] = new Array(items.length);
  const width = Math.max(1, Math.min(Math.floor(limit) || 1, items.length));
  let next = 0;
  const lane = async () => {
    while (next < items.length) {
      const i = next++;
      try {
        results[i] = { status: "fulfilled", value: await fn(items[i], i) };
      } catch (reason) {
        results[i] = { status: "rejected", reason };
      }
    }
  };
  await Promise.allSettled(Array.from({ length: width }, lane));
  return results;
}

export interface ConsumeOptions {
  /**
   * How many messages of one batch are handled at once inside a single
   * consumer invocation (AXO-211). Bounded because every handler holds its
   * images (base64, about 1.4 MB a page) and its request body in the same
   * 128 MB isolate, and because each one is a concurrent Gemini call. See
   * `queue_tuning.ts` for the per-stage values and the wall-time budget they
   * are checked against.
   */
  concurrency?: number;
}

/**
 * The queue() handler for a stage.
 *
 * A batch is processed CONCURRENTLY, up to `options.concurrency` at once, and
 * every message is acknowledged or retried on its own (`processQueueMessage`
 * decides). The batch is never acked or retried as a whole: one slow or
 * failing page must not hold back, or re-run, the pages that finished.
 */
export function consumeQueue<M extends RetryableMessage>(
  handle: (ctx: QueueContext<M>) => Promise<unknown>,
  onPermanent?: (ctx: QueueContext<M>, error: unknown) => Promise<unknown>,
  options: ConsumeOptions = {},
) {
  return async (batch: MessageBatch<M>, env: Env) => {
    const sb = serviceClient(env);
    await consumeBatch(batch.messages as readonly AckableMessage<M>[], sb, env, handle, onPermanent, options);
  };
}

/** The batch loop, separated from client construction so it can be tested. */
export async function consumeBatch<M extends RetryableMessage>(
  messages: readonly AckableMessage<M>[],
  sb: SupabaseClient,
  env: Env,
  handle: (ctx: QueueContext<M>) => Promise<unknown>,
  onPermanent?: (ctx: QueueContext<M>, error: unknown) => Promise<unknown>,
  options: ConsumeOptions = {},
): Promise<void> {
  const limit = options.concurrency ?? messages.length;
  const settled = await settleWithConcurrency(messages, limit, (message) =>
    processQueueMessage(message, sb, env, handle, onPermanent),
  );
  settled.forEach((result, i) => {
    if (result.status === "fulfilled") return;
    // processQueueMessage catches everything it knows how to classify. Reaching
    // here means the harness itself threw (an ack or retry call, say). Never
    // leave the message undecided: an explicit retry is the safe answer, and a
    // message already acked ignores it.
    console.error("queue harness error — retrying message", String(result.reason));
    try { messages[i].retry(); } catch (retryError) {
      console.error("could not mark message for retry", String(retryError));
    }
  });
}

/**
 * Should this caller send the next stage's work, given what an
 * `advance_after_*` function returned?
 *
 * Every `advance_after_*` function takes the run lock, so the run's STATUS
 * moves exactly once no matter how many messages finish together; exactly one
 * caller gets `advanced: true`. But once the run has moved on, the same
 * functions also hand the pending work list back to any later caller
 * (`advanced: false` with `enqueue_*`), so that a caller whose own downstream
 * send failed can resend after a retry. With a batch processed concurrently,
 * the later caller is usually a PEER that finished a moment after the one
 * that advanced — and resending would queue every region (and every content
 * model call) twice.
 *
 * So: send on `advanced: true`; send a re-issued list only on a redelivery,
 * which is exactly the case the re-issue exists for. The advancing caller's
 * failed send throws, its message is retried, and the retry is a redelivery.
 */
export function shouldFanOut(advance: { advanced?: boolean } | null | undefined, redelivered: boolean): boolean {
  if (!advance) return false;
  return advance.advanced === true || redelivered;
}
