/**
 * AXO-211: a queue batch is handled concurrently, every message is acked or
 * retried on its own, and a stage advances (and fans out) exactly once when
 * several messages finish together.
 */

import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

import {
  consumeBatch,
  settleWithConcurrency,
  shouldFanOut,
  HANDLE_TIMEOUT_MS,
  type AckableMessage,
  type QueueContext,
} from "../worker.js";
import { PermanentError, RetryableError, classifyDbError, isUniqueViolation } from "../errors.js";
import { QUEUE_TUNING, PER_MESSAGE_BUDGET_MS, WALL_BUDGET_MS, worstCaseBatchMs } from "../queue_tuning.js";
import { KeyedMutex } from "../keyed_mutex.js";
import { rememberImage, imageCacheChars } from "../r2.js";

type Body = { run_id: string; n: number; _retries?: number };

interface FakeMessage extends AckableMessage<Body> {
  acked: number;
  retried: number;
}

function message(n: number, attempts = 1, extra: Partial<Body> = {}): FakeMessage {
  const m: FakeMessage = {
    body: { run_id: "run", n, ...extra },
    attempts,
    acked: 0,
    retried: 0,
    ack: () => { m.acked++; },
    retry: () => { m.retried++; },
  };
  return m;
}

/** A Supabase stand-in that answers the harness's own RPCs (the heartbeat). */
const sb: any = { rpc: async () => ({ data: null, error: null }) };

function deferred() {
  let resolve!: () => void;
  const promise = new Promise<void>((r) => { resolve = r; });
  return { promise, resolve };
}

const tick = () => new Promise((r) => setTimeout(r, 0));

// ── concurrency ────────────────────────────────────────────────────────────

test("a batch is handled concurrently: every message starts before any finishes", async () => {
  const gate = deferred();
  const started: number[] = [];
  const messages = [1, 2, 3, 4, 5].map((n) => message(n));
  const run = consumeBatch<Body>(messages, sb, {} as any, async ({ msg }) => {
    started.push(msg.n);
    await gate.promise;
  }, undefined, { concurrency: 5 });
  for (let i = 0; i < 20 && started.length < 5; i++) await tick();
  assert.deepEqual(started.sort(), [1, 2, 3, 4, 5]);
  assert.equal(messages.filter((m) => m.acked).length, 0, "nothing is acked while work is in flight");
  gate.resolve();
  await run;
  assert.ok(messages.every((m) => m.acked === 1 && m.retried === 0));
});

test("in-invocation concurrency is bounded by the limit", async () => {
  let inFlight = 0;
  let peak = 0;
  const messages = Array.from({ length: 10 }, (_, i) => message(i));
  await consumeBatch<Body>(messages, sb, {} as any, async () => {
    inFlight++;
    peak = Math.max(peak, inFlight);
    await new Promise((r) => setTimeout(r, 5));
    inFlight--;
  }, undefined, { concurrency: 3 });
  assert.equal(peak, 3);
  assert.ok(messages.every((m) => m.acked === 1));
});

test("each message is acked or retried on its own: one failure neither blocks nor re-runs the rest", async () => {
  const ok1 = message(1);
  const transient = message(2);
  const permanent = message(3);
  const ok2 = message(4);
  const recorded: number[] = [];
  await consumeBatch<Body>([ok1, transient, permanent, ok2], sb, {} as any, async ({ msg }) => {
    if (msg.n === 2) throw new RetryableError("db_transient", "blip");
    if (msg.n === 3) throw new PermanentError("bad", "never");
  }, async ({ msg }) => { recorded.push(msg.n); }, { concurrency: 4 });

  assert.deepEqual([ok1.acked, ok1.retried], [1, 0]);
  assert.deepEqual([ok2.acked, ok2.retried], [1, 0]);
  // No SELF_QUEUE binding in this env: the native per-message retry.
  assert.deepEqual([transient.acked, transient.retried], [0, 1]);
  // Permanent: terminal state recorded, then acked.
  assert.deepEqual([permanent.acked, permanent.retried], [1, 0]);
  assert.deepEqual(recorded, [3]);
});

test("a retryable failure is re-enqueued per message with backoff when SELF_QUEUE exists", async () => {
  const sent: Array<{ body: Body; delay: number | undefined }> = [];
  const env: any = { SELF_QUEUE: { send: async (body: Body, o?: { delaySeconds?: number }) => { sent.push({ body, delay: o?.delaySeconds }); } } };
  const a = message(1);
  const b = message(2, 1, { _retries: 2 });
  const c = message(3);
  await consumeBatch<Body>([a, b, c], sb, env, async ({ msg }) => {
    if (msg.n !== 3) throw Object.assign(new Error("429"), { name: "ModelError", retryable: true });
  }, undefined, { concurrency: 3 });
  assert.equal(a.acked, 1);
  assert.equal(b.acked, 1);
  assert.equal(c.acked, 1);
  assert.deepEqual(sent.map((s) => [s.body.n, s.body._retries, s.delay]).sort(), [[1, 1, 5], [2, 3, 20]]);
});

test("a failed terminal write retries only that message", async () => {
  const a = message(1);
  const b = message(2);
  await consumeBatch<Body>([a, b], sb, {} as any, async ({ msg }) => {
    if (msg.n === 1) throw new PermanentError("bad", "never");
  }, async () => { throw new RetryableError("db_transient", "outage"); }, { concurrency: 2 });
  assert.deepEqual([a.acked, a.retried], [0, 1]);
  assert.deepEqual([b.acked, b.retried], [1, 0]);
});

test("redelivered is set from the queue's attempts or the manual retry count", async () => {
  const seen = new Map<number, boolean>();
  await consumeBatch<Body>([message(1), message(2, 2), message(3, 1, { _retries: 1 })], sb, {} as any,
    async ({ msg, redelivered }) => { seen.set(msg.n, redelivered); }, undefined, { concurrency: 3 });
  assert.deepEqual([seen.get(1), seen.get(2), seen.get(3)], [false, true, true]);
});

test("settleWithConcurrency settles every item in input order", async () => {
  const out = await settleWithConcurrency([3, 1, 2], 2, async (n) => {
    await new Promise((r) => setTimeout(r, n));
    if (n === 1) throw new Error("one");
    return n * 10;
  });
  assert.equal(out[0].status, "fulfilled");
  assert.equal((out[0] as PromiseFulfilledResult<number>).value, 30);
  assert.equal(out[1].status, "rejected");
  assert.equal((out[2] as PromiseFulfilledResult<number>).value, 20);
});

// ── advance exactly once ───────────────────────────────────────────────────

/**
 * The shape of `advance_after_structure` / `advance_after_crop` /
 * `advance_after_content` in production (read 7 Oct 2026): take the run lock,
 * return `advanced: false` while work is pending, move the status once and
 * return `advanced: true` with the work list, and afterwards hand the pending
 * list back (`advanced: false`) to any later caller.
 */
function fakeStage(items: number) {
  const state = { status: "structure", done: new Set<number>(), advances: 0 };
  const lock = new KeyedMutex();
  return {
    state,
    async finish(n: number) {
      await tick();
      state.done.add(n);
    },
    async advance() {
      return lock.run("run", async () => {
        await tick();
        if (state.done.size < items) return { advanced: false };
        if (state.status === "content") return { advanced: false, enqueue_content: ["r1", "r2"] };
        state.status = "content";
        state.advances++;
        return { advanced: true, enqueue_content: ["r1", "r2"] };
      });
    },
  };
}

test("several messages finishing together advance the stage once and fan out once", async () => {
  const stage = fakeStage(6);
  const sends: string[][] = [];
  const messages = Array.from({ length: 6 }, (_, i) => message(i));
  await consumeBatch<Body>(messages, sb, {} as any, async ({ msg, redelivered }) => {
    await stage.finish(msg.n);
    const advance = await stage.advance();
    if (shouldFanOut(advance, redelivered)) sends.push(advance.enqueue_content ?? []);
  }, undefined, { concurrency: 6 });
  assert.equal(stage.state.advances, 1);
  assert.equal(sends.length, 1, "next stage's work is queued exactly once");
  assert.ok(messages.every((m) => m.acked === 1));
});

test("without the redelivery gate the same race would queue the next stage several times", async () => {
  // This is what the re-issue branch does to concurrent peers when every
  // non-empty list is sent, and why `shouldFanOut` exists.
  const stage = fakeStage(6);
  let sends = 0;
  await consumeBatch<Body>(Array.from({ length: 6 }, (_, i) => message(i)), sb, {} as any, async ({ msg }) => {
    await stage.finish(msg.n);
    const advance = await stage.advance();
    if ((advance.enqueue_content ?? []).length) sends++;
  }, undefined, { concurrency: 6 });
  assert.equal(stage.state.advances, 1);
  assert.ok(sends > 1, `expected duplicate fan-out without the gate, saw ${sends}`);
});

test("the advancing message's failed send is resent on its redelivery, and nowhere else", async () => {
  const stage = fakeStage(3);
  let failNext = true;
  const sends: number[] = [];
  const handle = async ({ msg, redelivered }: QueueContext<Body>) => {
    await stage.finish(msg.n);
    const advance = await stage.advance();
    if (!shouldFanOut(advance, redelivered)) return;
    if (failNext) { failNext = false; throw new RetryableError("queue_send", "queue unavailable"); }
    sends.push(msg.n);
  };
  const first = [message(0), message(1), message(2)];
  await consumeBatch<Body>(first, sb, {} as any, handle, undefined, { concurrency: 3 });
  const failed = first.filter((m) => m.retried === 1);
  assert.equal(failed.length, 1, "only the message whose send failed is retried");
  assert.deepEqual(sends, []);
  // The queue redelivers it (attempts 2): the re-issued list is now honoured.
  const again = message(failed[0].body.n, 2);
  await consumeBatch<Body>([again], sb, {} as any, handle, undefined, { concurrency: 1 });
  assert.deepEqual(sends, [failed[0].body.n]);
  assert.equal(again.acked, 1);
  assert.equal(stage.state.advances, 1);
});

test("shouldFanOut: advanced always sends, a re-issued list only on redelivery", () => {
  assert.equal(shouldFanOut({ advanced: true }, false), true);
  assert.equal(shouldFanOut({ advanced: false }, false), false);
  assert.equal(shouldFanOut({ advanced: false }, true), true);
  assert.equal(shouldFanOut(null, true), false);
});

// ── helpers the structure stage relies on ─────────────────────────────────

test("a unique violation keeps its SQLSTATE so the structure stage can re-plan", () => {
  const e = classifyDbError({ code: "23505", message: "duplicate key" }, "pipeline_write");
  assert.ok(e instanceof PermanentError);
  assert.equal(isUniqueViolation(e), true);
  assert.equal(isUniqueViolation(classifyDbError({ code: "23503", message: "fk" }, "x")), false);
  assert.equal(isUniqueViolation(new Error("plain")), false);
});

test("KeyedMutex serialises one key and leaves other keys free", async () => {
  const m = new KeyedMutex();
  const order: string[] = [];
  let inA = 0;
  let peakA = 0;
  const a = (id: string) => m.run("a", async () => {
    inA++; peakA = Math.max(peakA, inA);
    order.push("start " + id);
    await new Promise((r) => setTimeout(r, 3));
    order.push("end " + id);
    inA--;
  });
  let bRan = false;
  await Promise.all([a("1"), a("2"), a("3"), m.run("b", async () => { bRan = true; })]);
  assert.equal(peakA, 1);
  assert.deepEqual(order, ["start 1", "end 1", "start 2", "end 2", "start 3", "end 3"]);
  assert.ok(bRan);
  assert.equal(m.size, 0);
  await assert.rejects(() => m.run("a", async () => { throw new Error("x"); }));
  await m.run("a", async () => {});
});

test("the image cache is bounded by bytes, oldest first", () => {
  const big = "x".repeat(1000);
  rememberImage("t/1", big, 2500);
  rememberImage("t/2", big, 2500);
  rememberImage("t/3", big, 2500);
  assert.ok(imageCacheChars() <= 2500);
  rememberImage("t/huge", "y".repeat(3000), 2500);
  assert.ok(imageCacheChars() <= 2500, "an entry larger than the budget is not cached");
});

// ── tuning matches the deployed consumers and fits the limits ─────────────

const here = dirname(fileURLToPath(import.meta.url));
const workers = join(here, "..", "..", "..", "workers");

test("the per-message budget in queue_tuning matches the harness", () => {
  assert.equal(PER_MESSAGE_BUDGET_MS, HANDLE_TIMEOUT_MS);
});

for (const [stage, tuning] of Object.entries(QUEUE_TUNING)) {
  test(`${stage}: wrangler max_batch_size matches QUEUE_TUNING and the batch fits the wall budget`, () => {
    const toml = readFileSync(join(workers, stage, "wrangler.toml"), "utf8");
    const consumer = toml.slice(toml.indexOf("[[queues.consumers]]"));
    const batch = Number(/max_batch_size\s*=\s*(\d+)/.exec(consumer)?.[1]);
    const timeout = Number(/max_batch_timeout\s*=\s*(\d+)/.exec(consumer)?.[1]);
    assert.equal(batch, tuning.maxBatchSize);
    assert.ok(timeout <= 1, "a short batch timeout, so a lone message is not held back");
    assert.ok(tuning.concurrency >= 1 && tuning.concurrency <= tuning.maxBatchSize);
    assert.ok(worstCaseBatchMs(tuning) <= WALL_BUDGET_MS,
      `${stage}: ${worstCaseBatchMs(tuning)} ms worst case exceeds ${WALL_BUDGET_MS} ms`);
  });
}
