import { test } from "node:test";
import assert from "node:assert/strict";
import { dispatchStageOutbox, messageFor, type OutboxEvent } from "../stage_outbox.js";

const ev = (id: string, stage: OutboxEvent["stage"], unit: string): OutboxEvent => ({ event_id: id, stage, run_id: "r1", unit_id: unit });

test("messages match what each stage worker reads", () => {
  assert.deepEqual(messageFor(ev("e", "structure", "p1")), { run_id: "r1", page_id: "p1" });
  assert.deepEqual(messageFor(ev("e", "content", "q1")), { run_id: "r1", region_id: "q1" });
});

test("nothing to send: no sends, no finish", async () => {
  let calls = 0;
  const out = await dispatchStageOutbox({
    claim: async () => [],
    finish: async () => { calls++; },
    send: async () => { calls++; },
  });
  assert.deepEqual(out, { delivered: 0, failed: 0 });
  assert.equal(calls, 0);
});

test("each stage is sent and marked delivered on its own", async () => {
  const sent: Array<[string, unknown[]]> = [];
  const finished: Array<[string[], string[], string | null]> = [];
  const out = await dispatchStageOutbox({
    claim: async () => [ev("a", "structure", "p1"), ev("b", "content", "q1"), ev("c", "content", "q2")],
    finish: async (d, f, e) => { finished.push([d, f, e]); },
    send: async (stage, msgs) => { sent.push([stage, msgs]); },
  });
  assert.deepEqual(out, { delivered: 3, failed: 0 });
  assert.deepEqual(sent, [
    ["structure", [{ run_id: "r1", page_id: "p1" }]],
    ["content", [{ run_id: "r1", region_id: "q1" }, { run_id: "r1", region_id: "q2" }]],
  ]);
  assert.deepEqual(finished, [[["a"], [], null], [["b", "c"], [], null]]);
});

test("a failed send backs off only that stage; the other stage is still delivered", async () => {
  const finished: Array<[string[], string[], string | null]> = [];
  const out = await dispatchStageOutbox({
    claim: async () => [ev("a", "structure", "p1"), ev("b", "content", "q1")],
    finish: async (d, f, e) => { finished.push([d, f, e]); },
    send: async (stage) => { if (stage === "structure") throw new Error("queue 503"); },
  });
  assert.deepEqual(out, { delivered: 1, failed: 1 });
  assert.deepEqual(finished[0], [[], ["a"], "Error: queue 503"]);
  assert.deepEqual(finished[1], [["b"], [], null]);
});
