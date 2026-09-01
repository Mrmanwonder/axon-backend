import { test } from "node:test";
import assert from "node:assert/strict";
import { reconcile } from "../reconcile.js";

test("reconcile: marks that sum to the reported total are reconciled", () => {
  const result = reconcile(
    [
      { order_index: 0, label: "1", awarded: 3, available: 4, recognition: "high" },
      { order_index: 1, label: "2", awarded: 5, available: 6, recognition: "high" },
    ],
    8,
    10
  );
  assert.equal(result.reconciled, true);
  assert.equal(result.delta, 0);
  assert.equal(result.sum_awarded, 8);
});

test("reconcile: a mismatch is not reconciled and reports a delta", () => {
  const result = reconcile(
    [
      { order_index: 0, label: "1", awarded: 3, available: 4, recognition: "high" },
      { order_index: 1, label: "2", awarded: 5, available: 6, recognition: "medium" },
    ],
    9,
    10
  );
  assert.equal(result.reconciled, false);
  assert.equal(result.delta, -1);
});

test("reconcile: a question awarded more than its available marks is never reconciled, even if the total matches by coincidence", () => {
  const result = reconcile([{ order_index: 0, label: "1", awarded: 5, available: 4, recognition: "high" }], 5, null);
  assert.equal(result.checks.every_question_within_its_maximum, false);
  assert.equal(result.reconciled, false);
});

test("reconcile: with no reported total at all, reconciled is false rather than vacuously true", () => {
  const result = reconcile([{ order_index: 0, label: "1", awarded: 3, available: 4, recognition: "high" }], null, null);
  assert.equal(result.reconciled, false);
  assert.equal(result.delta, null);
});

test("reconcile: suspects rank the least-confident and most-incomplete questions first, but promote the region matching the exact delta", () => {
  const result = reconcile(
    [
      { order_index: 0, label: "1", awarded: 3, available: 4, recognition: "high" },
      { order_index: 1, label: "2", awarded: null, available: 6, recognition: "low" },
      { order_index: 2, label: "3", awarded: 2, available: 2, recognition: "high" },
    ],
    7,
    12
  );
  // delta = 3+0+2 - 7 = -2 -> |delta| = 2, matches region 2's awarded value exactly.
  assert.equal(result.suspects[0], 2);
});
