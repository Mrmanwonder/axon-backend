import { test } from "node:test";
import assert from "node:assert";
import { chunkedSendBatch } from "../chunked_send.js";

test("chunkedSendBatch", async (t) => {
  await t.test("chunks items into batches of 100", async () => {
    let callCount = 0;
    const calls: any[][] = [];
    const queue = {
      sendBatch: async (batch: any[]) => {
        callCount++;
        calls.push(batch);
      }
    };
    const items = Array.from({ length: 250 }, (_, i) => i);

    await chunkedSendBatch(queue, items, (id) => ({ body: { id } }));

    assert.strictEqual(callCount, 3);
    assert.strictEqual(calls[0].length, 100);
    assert.strictEqual(calls[1].length, 100);
    assert.strictEqual(calls[2].length, 50);
  });

  await t.test("limits concurrency", async () => {
    let inFlight = 0;
    let maxObservedInFlight = 0;
    let callCount = 0;

    const queue = {
      sendBatch: async (batch: any[]) => {
        callCount++;
        inFlight++;
        maxObservedInFlight = Math.max(maxObservedInFlight, inFlight);
        await new Promise((r) => setTimeout(r, 10));
        inFlight--;
      }
    };

    const items = Array.from({ length: 1000 }, (_, i) => i);
    await chunkedSendBatch(queue, items, (id) => ({ body: { id } }), 2);

    assert.strictEqual(callCount, 10);
    assert.strictEqual(maxObservedInFlight, 2);
  });

  await t.test("halts on first error and throws", async () => {
    let callCount = 0;
    const queue = {
      sendBatch: async (batch: any[]) => {
        callCount++;
        if (callCount === 2) {
          throw new Error("failed on second batch");
        }
        await new Promise((r) => setTimeout(r, 10));
      }
    };

    const items = Array.from({ length: 500 }, (_, i) => i);

    try {
      await chunkedSendBatch(queue, items, (id) => ({ body: { id } }), 1);
      assert.fail("should have thrown");
    } catch (error: any) {
      assert.strictEqual(error.message, "failed on second batch");
    }

    assert.strictEqual(callCount, 2);
  });
});
