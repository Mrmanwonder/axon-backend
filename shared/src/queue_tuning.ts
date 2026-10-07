/**
 * Queue consumer tuning per pipeline stage (AXO-211).
 *
 * Measured on a 13-page paper (run 24d72442, 6 Oct 2026): with
 * `max_batch_size = 1` every stage ran one message per invocation and relied
 * on Cloudflare's consumer autoscaler for parallelism. The autoscaler does not
 * ramp for a short burst, so structure ran its 13 pages strictly one after
 * another (about 97 s) and content climbed slowly from 1 to about 6 at once
 * (about 160 s for 24 regions).
 *
 * Now one invocation takes a whole batch and works through it concurrently
 * (`consumeQueue`, `settleWithConcurrency`), each message acked or retried on
 * its own. `maxBatchSize` must equal `max_batch_size` in the stage's
 * wrangler.toml; a test reads both and fails if they drift.
 *
 * Limits these numbers rely on (Workers Paid, standard usage model):
 *   - Queue consumer wall time: 15 minutes per invocation. Each message is
 *     bounded by HANDLE_TIMEOUT_MS (200 s) in `processQueueMessage`, so a batch
 *     takes at most ceil(maxBatchSize / concurrency) * 200 s of handler time;
 *     that is kept at or under 10 minutes (`WALL_BUDGET_MS`), leaving room for
 *     the permanent-failure path and queue sends. A batch that does overrun
 *     loses nothing: explicit acks already made stand, and anything unacked is
 *     redelivered.
 *   - CPU time: the default 30 s per invocation. Handlers are I/O bound: the
 *     CPU work per message is a native base64 of one or two images, a JSON
 *     encode/decode and pure planning code, tens of milliseconds, so even a
 *     25-message batch stays far below the default. No `limits.cpu_ms` is set.
 *   - Memory: 128 MB per isolate. A model call holds its images as base64 data
 *     URLs (pages are about 0.8 to 1 MB, so about 1.4 MB each in base64) plus
 *     the JSON request body, roughly 3 to 4 MB per call in flight.
 *     `concurrency` bounds that, together with the byte cap on the image cache
 *     in r2.ts.
 *   - Gemini rate limits: calls in flight per stage are at most
 *     `concurrency * max_concurrency` (wrangler.toml). A 429 is still retried
 *     in-process once with backoff (model-client.ts) and then through the
 *     queue with exponential delay (worker.ts), unchanged.
 */

export interface StageTuning {
  /** Must equal `max_batch_size` in the worker's wrangler.toml. */
  maxBatchSize: number;
  /** Messages handled at once inside one invocation. */
  concurrency: number;
}

/** Same value as HANDLE_TIMEOUT_MS in worker.ts; a test keeps them equal. */
export const PER_MESSAGE_BUDGET_MS = 200_000;
/** The share of the 15-minute consumer wall limit one batch may plan to use. */
export const WALL_BUDGET_MS = 10 * 60_000;

export const QUEUE_TUNING = {
  // A 13-page paper arrives as one batch (the contract allows 25 pages) and is
  // read 13 pages at once: one wave of structure calls instead of thirteen.
  structure: { maxBatchSize: 25, concurrency: 13 },
  // 24 regions in one batch, 12 calls in flight: two waves at most.
  content: { maxBatchSize: 24, concurrency: 12 },
  // One message per run. Batching helps only when several papers arrive
  // together, and costs nothing for one.
  triage: { maxBatchSize: 5, concurrency: 5 },
  reconcile: { maxBatchSize: 10, concurrency: 10 },
  adjudicate: { maxBatchSize: 5, concurrency: 5 },
  // One message per region after review, plus topic tags and scheme checks.
  // This worker holds no page images.
  explain: { maxBatchSize: 10, concurrency: 10 },
} as const satisfies Record<string, StageTuning>;

export type TunedStage = keyof typeof QUEUE_TUNING;

/** Worst-case handler time for one full batch at this tuning. */
export function worstCaseBatchMs(t: StageTuning): number {
  return Math.ceil(t.maxBatchSize / t.concurrency) * PER_MESSAGE_BUDGET_MS;
}
