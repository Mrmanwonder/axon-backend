import { readFileSync } from "node:fs";
import { expect, test } from "vitest";
import { QUEUE_TUNING, WALL_BUDGET_MS, worstCaseBatchMs } from "@mastery/shared/queue_tuning.js";

// AXO-224 regression: Cloudflare telemetry showed 50-subrequest invocations
// while 41 content messages exhausted retries. Do not restore the old 24-wide
// batch until an actual 12-page run and account limit have been verified.
test("content batches stay small and match live consumer configuration", () => {
  const toml = readFileSync(new URL("../wrangler.toml", import.meta.url), "utf8");
  const block = toml.split("[[queues.consumers]]").find(part => /queue\\s*=\\s*"content-queue"/.test(part));
  expect(block, "content queue stanza exists").toBeTruthy();
  const batchSize = Number(block!.match(/max_batch_size\\s*=\\s*(\\d+)/)?.[1]);
  const maxConcurrency = Number(block!.match(/max_concurrency\\s*=\\s*(\\d+)/)?.[1]);
  expect(batchSize).toBe(QUEUE_TUNING.content.maxBatchSize);
  expect(batchSize).toBeLessThanOrEqual(2);
  expect(QUEUE_TUNING.content.concurrency).toBeLessThanOrEqual(batchSize);
  expect(worstCaseBatchMs(QUEUE_TUNING.content)).toBeLessThanOrEqual(WALL_BUDGET_MS);
  expect(maxConcurrency * QUEUE_TUNING.content.concurrency).toBeLessThanOrEqual(12);
});
