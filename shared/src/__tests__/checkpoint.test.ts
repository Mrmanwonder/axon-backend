import { test } from "node:test";
import assert from "node:assert/strict";
import { callModel } from "../model-client.js";
import { RetryableError } from "../errors.js";
import { digestHex } from "../checkpoint.js";

// AXO-225. The fake journal below follows the contract of public.work_* (proved
// against the real SQL in Axon-Site tests/db/work-journal.test.mjs). These tests
// prove callModel uses that contract correctly: when it pays, when it must not,
// and what it does when the journal itself misbehaves.

type Row = {
  work_id: string; status: "leased" | "output_saved" | "invalidated"; lease_token: string | null;
  expires: number; output: unknown; meta: unknown; key: string;
};

function fakeDb(stage: string, opts: { failSaves?: boolean } = {}) {
  const rows: Row[] = [];
  const calls: string[] = [];
  const route = {
    stage, primary_model: "gemini-test", fallbacks: [], temperature: 0, max_tokens: 512,
    prompt_version: "p.v1", thinking_level: "low", allow_training: false, enabled: true,
  };
  const keyOf = (a: Record<string, unknown>) =>
    [a.p_paper_id, a.p_stage, a.p_unit_key, a.p_input_digest, a.p_prompt_digest, a.p_model_id, a.p_prompt_version, a.p_schema_version].join("|");
  const sb = {
    from(table: string) {
      if (table === "model_route") {
        const q = { select: () => q, eq: () => q, maybeSingle: async () => ({ data: route, error: null }) };
        return q;
      }
      return { insert: async () => ({ error: null }) };
    },
    async rpc(name: string, a: Record<string, any>) {
      calls.push(name);
      if (name === "work_begin") {
        const key = keyOf(a);
        let row = rows.find((r) => r.key === key && r.status !== "invalidated");
        if (!row) { row = { work_id: `w${rows.length + 1}`, status: "leased", lease_token: null, expires: 0, output: null, meta: null, key }; rows.push(row); row.status = "leased"; row.expires = 0; }
        if (row.status === "output_saved") return { data: { action: "reuse", work_id: row.work_id, status: "output_saved", output: row.output, output_meta: row.meta, checksum: "x" }, error: null };
        if (row.status === "leased" && row.expires > Date.now()) return { data: { action: "busy", work_id: row.work_id, retry_after_seconds: 30 }, error: null };
        row.status = "leased"; row.lease_token = `t${calls.length}`; row.expires = Date.now() + 300_000;
        return { data: { action: "run", work_id: row.work_id, lease_token: row.lease_token, generation: 1, attempt: 1, unknown_outcomes: 0 }, error: null };
      }
      if (name === "work_save_output") {
        if (opts.failSaves) return { data: null, error: { code: "08006", message: "connection lost" } };
        const row = rows.find((r) => r.work_id === a.p_work_id)!;
        if (row.status === "output_saved") return { data: { result: "already_saved", output: row.output, output_meta: row.meta }, error: null };
        if (row.status !== "leased" || row.lease_token !== a.p_lease_token) return { data: { result: "fenced" }, error: null };
        row.status = "output_saved"; row.output = a.p_output; row.meta = a.p_output_meta; row.lease_token = null;
        return { data: { result: "saved" }, error: null };
      }
      if (name === "work_release") {
        const row = rows.find((r) => r.work_id === a.p_work_id)!;
        if (row.status === "leased" && row.lease_token === a.p_lease_token) { row.lease_token = null; row.expires = 0; }
        return { data: true, error: null };
      }
      if (name === "work_invalidate") {
        rows.find((r) => r.work_id === a.p_work_id)!.status = "invalidated";
        return { data: true, error: null };
      }
      if (name === "work_mark_applied") return { data: true, error: null };
      throw new Error("unexpected rpc " + name);
    },
  };
  return { sb, rows, calls };
}

const IDS = { paperId: "p1", studentId: "s1" };
const spec = async () => ({ unitKey: "content:p1:aa", imageDigest: await digestHex(["page-sha"]) });
const schema = { name: "t", schema: { type: "object" } };
const call = async (env: Record<string, string>, sb: unknown, stage: string, extra: Record<string, unknown> = {}) =>
  callModel({
    env, sb: sb as never, stage, system: "sys", instruction: "ins", schema,
    validate: (v: unknown) => v as { answer: string }, ...IDS, checkpoint: await spec(), ...extra,
  } as never);

function mockGemini(t: any, answer: unknown = { answer: "ok" }) {
  const counter = { n: 0 };
  t.mock.method(globalThis, "fetch", async () => {
    counter.n += 1;
    return Response.json({ model: "gemini-test", choices: [{ message: { content: JSON.stringify(answer) } }], usage: { prompt_tokens: 3, completion_tokens: 2, cost: 0.01 } });
  });
  return counter;
}

const ON = { GOOGLE_API_KEY: "k", WORK_JOURNAL: "1" };

test("the second run asks the journal first and makes no model call", async (t) => {
  const fx = fakeDb("ck-reuse");
  const gemini = mockGemini(t);
  const a = await call(ON, fx.sb, "ck-reuse");
  const b = await call(ON, fx.sb, "ck-reuse");
  assert.equal(gemini.n, 1);
  assert.equal(a.reused, undefined);
  assert.equal(b.reused, true);
  assert.deepEqual(b.parsed, { answer: "ok" });
  assert.equal(b.costUsd, 0);
  assert.equal(b.workId, a.workId);
});

test("the journal is off unless WORK_JOURNAL=1: no journal calls, every call is paid", async (t) => {
  const fx = fakeDb("ck-off");
  const gemini = mockGemini(t);
  await call({ GOOGLE_API_KEY: "k" }, fx.sb, "ck-off");
  await call({ GOOGLE_API_KEY: "k" }, fx.sb, "ck-off");
  assert.equal(gemini.n, 2);
  assert.deepEqual(fx.calls, []);
});

test("no checkpoint spec, or no student, means no journal", async (t) => {
  const fx = fakeDb("ck-nospec");
  const gemini = mockGemini(t);
  await call(ON, fx.sb, "ck-nospec", { checkpoint: undefined });
  await call(ON, fx.sb, "ck-nospec", { studentId: null });
  assert.equal(gemini.n, 2);
  assert.deepEqual(fx.calls, []);
});

test("a live lease held elsewhere throws a retryable error and pays nothing", async (t) => {
  const fx = fakeDb("ck-busy");
  let release!: () => void;
  const gate = new Promise<void>((r) => (release = r));
  let paid = 0;
  t.mock.method(globalThis, "fetch", async () => {
    paid += 1;
    await gate;
    return Response.json({ model: "gemini-test", choices: [{ message: { content: JSON.stringify({ answer: "ok" }) } }], usage: { prompt_tokens: 1, completion_tokens: 1 } });
  });
  const first = call(ON, fx.sb, "ck-busy"); // holds the lease, waiting on the provider
  await new Promise((r) => setTimeout(r, 20));
  await assert.rejects(call(ON, fx.sb, "ck-busy"), (e: unknown) => e instanceof RetryableError && e.code === "work_busy");
  release();
  await first;
  assert.equal(paid, 1); // only the lease holder paid
});

test("a failed model call releases the lease so the retry can claim at once", async (t) => {
  const fx = fakeDb("ck-fail");
  let n = 0;
  t.mock.method(globalThis, "fetch", async () => {
    n += 1;
    if (n <= 1) return new Response("boom", { status: 400 }); // not transient: no inline retry
    return Response.json({ model: "gemini-test", choices: [{ message: { content: JSON.stringify({ answer: "ok" }) } }], usage: { prompt_tokens: 1, completion_tokens: 1 } });
  });
  await assert.rejects(call(ON, fx.sb, "ck-fail"));
  assert.ok(fx.calls.includes("work_release"));
  const r = await call(ON, fx.sb, "ck-fail");
  assert.deepEqual(r.parsed, { answer: "ok" });
});

test("a saved answer the validator now refuses is invalidated and paid for again", async (t) => {
  const fx = fakeDb("ck-invalid");
  const gemini = mockGemini(t);
  await call(ON, fx.sb, "ck-invalid");
  let strict = false;
  const r = await call(ON, fx.sb, "ck-invalid", {
    validate: (v: any) => { if (!strict) { strict = true; throw new Error("schema moved on"); } return v; },
  });
  assert.equal(gemini.n, 2);
  assert.ok(fx.calls.includes("work_invalidate"));
  assert.deepEqual(r.parsed, { answer: "ok" });
});

test("if the journal cannot save, the paid answer is still returned", async (t) => {
  const fx = fakeDb("ck-nosave", { failSaves: true });
  const gemini = mockGemini(t);
  const r = await call(ON, fx.sb, "ck-nosave");
  assert.deepEqual(r.parsed, { answer: "ok" });
  assert.equal(gemini.n, 1);
  assert.equal(fx.calls.filter((c) => c === "work_save_output").length, 3);
});

test("different evidence or a different prompt is different work", async (t) => {
  const fx = fakeDb("ck-keys");
  const gemini = mockGemini(t);
  await call(ON, fx.sb, "ck-keys");
  await call(ON, fx.sb, "ck-keys", { checkpoint: { unitKey: "content:p1:aa", imageDigest: await digestHex(["other-page"]) } });
  await call(ON, fx.sb, "ck-keys", { instruction: "a teacher mark changed" });
  await call(ON, fx.sb, "ck-keys", { schema: { name: "t", schema: { type: "object", required: ["answer"] } } });
  assert.equal(gemini.n, 4);
  await call(ON, fx.sb, "ck-keys");
  assert.equal(gemini.n, 4);
});

test("the answer is saved as the model said it, before the caller sees it", async (t) => {
  const fx = fakeDb("ck-raw");
  mockGemini(t, { answer: "raw", extra: 1 });
  const r = await call(ON, fx.sb, "ck-raw", { validate: (v: any) => ({ answer: v.answer }) });
  assert.deepEqual(r.parsed, { answer: "raw" });
  assert.deepEqual(fx.rows[0].output, { answer: "raw", extra: 1 });
});
