import { readFileSync } from "node:fs";
import { expect, test } from "vitest";
import { QUEUE_TUNING, WALL_BUDGET_MS, worstCaseBatchMs } from "@mastery/shared/queue_tuning.js";

// AXO-224: 41 content-DLQ and 29 structure-DLQ messages, many exactly-50
// subrequest invocations. Prevent another 24/25-wide batch until an actual
// full-size run and the Cloudflare account's subrequest limit are verified.
for (const stage of ["content", "structure"] as const) {
  test(stage + " consumer batch configuration is bounded and matches tuning", () => {
    const path = stage === "content" ? "../wrangler.toml" : "../../structure/wrangler.toml";
    const toml = readFileSync(new URL(path, import.meta.url), "utf8");
    const stanza = toml.split("[[queues.consumers]]").slice(1)
      .find(part => part.includes('queue = "' + stage + '-queue"'));
    expect(stanza, stage + " consumer stanza exists").toBeTruthy();
    const batchSize = Number(stanza!.match(/max_batch_size\s*=\s*(\d+)/)?.[1]);
    const maxConcurrency = Number(stanza!.match(/max_concurrency\s*=\s*(\d+)/)?.[1]);
    const tuning = QUEUE_TUNING[stage];
    expect(batchSize).toBe(tuning.maxBatchSize);
    expect(batchSize).toBeLessThanOrEqual(2);
    expect(tuning.concurrency).toBeLessThanOrEqual(batchSize);
    expect(worstCaseBatchMs(tuning)).toBeLessThanOrEqual(WALL_BUDGET_MS);
    expect(maxConcurrency * tuning.concurrency).toBeLessThanOrEqual(12);
  });
}
