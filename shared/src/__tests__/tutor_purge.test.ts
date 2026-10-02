import { test } from "node:test";
import assert from "node:assert/strict";
import { runTutorPurges, type PurgeSink } from "../tutor_purge.js";

function sink(claims: Array<{ id: number; paper_id: string }>) {
  const finished: Array<[number, string | null]> = [];
  let claimed = 0;
  const s: PurgeSink = {
    claim: async () => { claimed += 1; return claims; },
    finish: async (id, error) => { finished.push([id, error]); },
  };
  return { s, finished, claimedCount: () => claimed };
}

const two = [{ id: 1, paper_id: "p1" }, { id: 2, paper_id: "p2" }];

test("a purge the Tutor accepted marks every claimed row done", async () => {
  const { s, finished } = sink(two);
  let sent: string[] = [];
  const n = await runTutorPurges(s, async (ids) => { sent = ids; return { ok: true, status: 200 }; });
  assert.equal(n, 2);
  assert.deepEqual(sent, ["p1", "p2"]);
  assert.deepEqual(finished, [[1, null], [2, null]]);
});

test("a refusal records why on every row and delivers nothing", async () => {
  const { s, finished } = sink(two);
  const n = await runTutorPurges(s, async () => ({ ok: false, status: 500 }));
  assert.equal(n, 0);
  assert.deepEqual(finished, [[1, "tutor purge returned 500"], [2, "tutor purge returned 500"]]);
});

test("an unreachable Tutor records the error and leaves the rows for retry", async () => {
  const { s, finished } = sink(two);
  const n = await runTutorPurges(s, async () => { throw new Error("offline"); });
  assert.equal(n, 0);
  assert.match(String(finished[0][1]), /offline/);
});

test("with no Tutor configured nothing is claimed, so attempt counts are not burned", async () => {
  const { s, finished, claimedCount } = sink(two);
  assert.equal(await runTutorPurges(s, undefined), 0);
  assert.equal(claimedCount(), 0);
  assert.deepEqual(finished, []);
});

test("an empty queue calls nothing", async () => {
  const { s } = sink([]);
  let called = false;
  assert.equal(await runTutorPurges(s, async () => { called = true; return { ok: true, status: 200 }; }), 0);
  assert.equal(called, false);
});
