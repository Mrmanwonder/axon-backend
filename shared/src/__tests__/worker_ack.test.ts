/**
 * When may a queue message be acknowledged?
 *
 * The re-audit's P0-D is a sequence, not a state, so the tests walk it:
 *
 *   1. a handler hits a transient Supabase error;
 *   2. the old classifier said "not a ModelError, therefore permanent";
 *   3. onPermanent ran failRun, which ignored its own {error};
 *   4. the ACK sat in a `finally`, so it fired anyway;
 *   5. the run stayed mid-pipeline with no message left to move it.
 *
 * Every step of that is a separate assertion below, because fixing any one of
 * them alone still loses the work.
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import { isRetryable, PermanentError, RetryableError, ConfigurationError, classifyDbError } from "../errors.js";
import { mustData, mustOk, mustMaybe, mustOne, mustAffectRows } from "../db.js";
import { processQueueMessage } from "../worker.js";

// ── the classifier ─────────────────────────────────────────────────────────

test("an unclassified error is retryable, because unknown is more often transient", () => {
  assert.equal(isRetryable(new Error("something nobody named")), true);
});

test("a permanent error is not retried", () => {
  assert.equal(isRetryable(new PermanentError("bad_input", "no")), false);
});

test("a configuration error is not retried — no amount of waiting fixes a missing binding", () => {
  assert.equal(isRetryable(new ConfigurationError("no_binding", "no")), false);
});

test("a retryable error is retried", () => {
  assert.equal(isRetryable(new RetryableError("db_transient", "later")), true);
});

test("a model error still carries its own judgement", () => {
  const nonRetryable = Object.assign(new Error("bad request"), { name: "ModelError", retryable: false });
  const retryable = Object.assign(new Error("429"), { name: "ModelError", retryable: true });
  assert.equal(isRetryable(nonRetryable), false);
  assert.equal(isRetryable(retryable), true);
});

test("a handler that ran out of budget has proved nothing, so it retries", () => {
  assert.equal(isRetryable(Object.assign(new Error("slow"), { name: "HandlerTimeout" })), true);
});

test("postgres classes are split by whether trying again could help", () => {
  assert.ok(classifyDbError({ code: "57014", message: "canceled" }, "x") instanceof RetryableError);
  assert.ok(classifyDbError({ code: "40001", message: "serialize" }, "x") instanceof RetryableError);
  assert.ok(classifyDbError({ code: "08006", message: "conn" }, "x") instanceof RetryableError);
  // A constraint violation will fail identically forever.
  assert.ok(classifyDbError({ code: "23505", message: "dup" }, "x") instanceof PermanentError);
  assert.ok(classifyDbError({ code: "22P02", message: "bad uuid" }, "x") instanceof PermanentError);
  // Grants and JWTs are deployment problems, alerted separately.
  assert.ok(classifyDbError({ code: "42501", message: "denied" }, "x") instanceof ConfigurationError);
  // No code at all is a fetch that never reached Postgres.
  assert.ok(classifyDbError({ message: "network" }, "x") instanceof RetryableError);
});

// ── the checked helpers ────────────────────────────────────────────────────

const ok = <T>(data: T) => Promise.resolve({ data, error: null });
const fails = (code: string) => Promise.resolve({ data: null, error: { code, message: "boom" } });

test("an empty result is not an error", async () => {
  assert.deepEqual(await mustData(ok([]), "read"), []);
  assert.equal(await mustMaybe(ok(null), "read"), null);
});

test("a failed read throws instead of returning empty", async () => {
  await assert.rejects(() => mustData(fails("57014"), "pages"), /pages.*57014/);
  await assert.rejects(() => mustOk(fails("57014"), "update"), /update/);
  await assert.rejects(() => mustMaybe(fails("57014"), "region"), /region/);
});

test("a row addressed by a queue message that is not there fails permanently", async () => {
  // Waiting cannot conjure it: the id came off the message.
  await assert.rejects(
    () => mustOne(ok(null), "run"),
    (e: Error) => e instanceof PermanentError,
  );
});

test("an update matching no rows is not a success", async () => {
  await assert.rejects(
    () => mustAffectRows(ok([]), "claim region"),
    (e: Error) => e instanceof PermanentError && /matched no rows/.test(e.message),
  );
  assert.deepEqual(await mustAffectRows(ok([{ id: "r1" }]), "claim region"), [{ id: "r1" }]);
});

// ── the harness ────────────────────────────────────────────────────────────

type Msg = { run_id?: string; _retries?: number };

function harness(opts: {
  handle: () => Promise<unknown>;
  onPermanent?: () => Promise<unknown>;
  selfQueue?: { send: () => Promise<void> };
  rpc?: (name: string) => { data: unknown; error: unknown };
  body?: Msg;
}) {
  const calls: string[] = [];
  const message = {
    body: opts.body ?? ({ run_id: "run-1" } as Msg),
    attempts: 1,
    ack: () => calls.push("ack"),
    retry: () => calls.push("retry"),
  };
  const env = {
    SUPABASE_URL: "http://localhost",
    SUPABASE_SERVICE_ROLE_KEY: "k",
    SELF_QUEUE: opts.selfQueue,
  } as never;

  // Stubbed rather than a real PostgREST client: these tests are about the
  // harness's control flow, not about the wire format.
  const sb = {
    rpc: async (name: string) => (opts.rpc ? opts.rpc(name) : { data: null, error: null }),
  } as never;

  return {
    calls,
    run: async () => {
      await processQueueMessage<Msg>(
        message,
        sb,
        env,
        async () => opts.handle(),
        opts.onPermanent ? async () => opts.onPermanent!() : undefined,
      );
      return calls;
    },
  };
}

const run = (h: ReturnType<typeof harness>) => h.run();

test("a successful handler acknowledges", async () => {
  const h = harness({ handle: async () => "done" });
  assert.deepEqual(await run(h), ["ack"]);
});

test("a transient database failure retries instead of being treated as permanent", async () => {
  // THE regression. Under the old classifier this was permanent, because it is
  // not a ModelError.
  let permanentRan = false;
  const h = harness({
    handle: async () => { throw new RetryableError("db_transient", "503"); },
    onPermanent: async () => { permanentRan = true; },
  });
  const calls = await run(h);
  assert.equal(permanentRan, false, "must not enter the permanent path");
  assert.deepEqual(calls, ["retry"]);
});

test("an unclassified handler error also retries", async () => {
  const h = harness({ handle: async () => { throw new Error("plain throw meaning retry"); } });
  assert.deepEqual(await run(h), ["retry"]);
});

test("a permanent error whose terminal write SUCCEEDS is acknowledged", async () => {
  const h = harness({
    handle: async () => { throw new PermanentError("bad", "malformed"); },
    onPermanent: async () => { /* recorded */ },
  });
  assert.deepEqual(await run(h), ["ack"]);
});

test("a permanent error whose terminal write FAILS is retried, not acknowledged", async () => {
  // The lost-work sequence. The old harness put ack() in a `finally`, so this
  // returned ["ack"] and the run was stranded with no message left.
  const h = harness({
    handle: async () => { throw new PermanentError("bad", "malformed"); },
    onPermanent: async () => { throw new RetryableError("db_transient", "503 while recording failure"); },
  });
  assert.deepEqual(await run(h), ["retry"]);
});

test("a permanent error with nothing to record is acknowledged", async () => {
  const h = harness({ handle: async () => { throw new PermanentError("bad", "x"); } });
  assert.deepEqual(await run(h), ["ack"]);
});

test("a retryable error acknowledges only once the replacement message is durable", async () => {
  let sent = 0;
  const h = harness({
    handle: async () => { throw new RetryableError("db_transient", "503"); },
    selfQueue: { send: async () => { sent += 1; } },
  });
  assert.deepEqual(await run(h), ["ack"]);
  assert.equal(sent, 1, "the replacement is what makes forgetting this one safe");
});

test("a failed re-enqueue falls back to native retry rather than acknowledging", async () => {
  const h = harness({
    handle: async () => { throw new RetryableError("db_transient", "503"); },
    selfQueue: { send: async () => { throw new Error("queue down"); } },
  });
  assert.deepEqual(await run(h), ["retry"]);
});

test("retries are bounded, and exhausting them routes to the permanent path", async () => {
  let permanentRan = false;
  const h = harness({
    body: { run_id: "run-1", _retries: 5 },
    handle: async () => { throw new RetryableError("db_transient", "503"); },
    onPermanent: async () => { permanentRan = true; },
  });
  const calls = await run(h);
  assert.equal(permanentRan, true);
  assert.deepEqual(calls, ["ack"]);
});

test("a heartbeat failure is a database failure, not something to run on blind", async () => {
  let handlerRan = false;
  const h = harness({
    handle: async () => { handlerRan = true; },
    rpc: (name) => name === "run_heartbeat"
      ? { data: null, error: { code: "57014", message: "canceled" } }
      : { data: null, error: null },
  });
  const calls = await run(h);
  assert.equal(handlerRan, false, "a run whose heartbeat is not landing must not be swept mid-work");
  assert.deepEqual(calls, ["retry"]);
});
