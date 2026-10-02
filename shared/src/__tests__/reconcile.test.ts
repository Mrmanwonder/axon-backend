import { test } from "node:test";
import assert from "node:assert/strict";
import { adjudicationTriggers, reconcile } from "../reconcile.js";

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

// Decision of 2026-10-02 (AXO-124): a paper with no printed total is "unchecked" (null), never
// "false". This replaces the earlier rule that stored false, which sent every such paper to the
// adjudicator for a problem it did not have.
test("reconcile: with no printed total at all, reconciled is null (unchecked), not false", () => {
  const result = reconcile([{ order_index: 0, label: "1", awarded: 3, available: 4, recognition: "high" }], null, null);
  assert.equal(result.reconciled, null);
  assert.equal(result.delta, null);
  assert.equal(result.added_up, true);
  assert.equal(result.partial, false);
  assert.equal(result.sum_awarded, 3);
});

test("reconcile: a printed total makes the total the paper's, not ours", () => {
  const result = reconcile([{ order_index: 0, label: "1", awarded: 3, available: 4, recognition: "high" }], 3, 4);
  assert.equal(result.added_up, false);
  assert.equal(result.reconciled, true);
});

test("reconcile: an unreadable mark makes the added-up total partial", () => {
  const result = reconcile(
    [
      { order_index: 0, label: "1", awarded: 3, available: 4, recognition: "high" },
      { order_index: 1, label: "2", awarded: null, available: 5, recognition: "low" },
    ],
    null,
    null
  );
  assert.equal(result.reconciled, null);
  assert.equal(result.partial, true);
  assert.equal(result.sum_awarded, 3);
});

test("reconcile: a mark above its maximum is false even with no printed total", () => {
  const result = reconcile([{ order_index: 0, label: "1", awarded: 5, available: 4, recognition: "high" }], null, null);
  assert.equal(result.reconciled, false);
});

test("adjudication: a missing total alone is never a trigger", () => {
  const regions = [
    { order_index: 0, label: "1", awarded: 3, available: 4, recognition: "high" as const },
    { order_index: 1, label: "2", awarded: 5, available: 6, recognition: "medium" as const },
  ];
  assert.deepEqual(adjudicationTriggers(reconcile(regions, null, null), regions, false), []);
});

test("adjudication: an unchecked paper is adjudicated on its own evidence", () => {
  const withUnreadable = [
    { order_index: 0, label: "1", awarded: 3, available: 4, recognition: "high" as const },
    { order_index: 1, label: "2", awarded: null, available: 5, recognition: "low" as const },
  ];
  assert.deepEqual(adjudicationTriggers(reconcile(withUnreadable, null, null), withUnreadable, false), ["low_confidence_marks"]);
  const clean = [{ order_index: 0, label: "1", awarded: 3, available: 4, recognition: "high" as const }];
  assert.deepEqual(adjudicationTriggers(reconcile(clean, null, null), clean, true), ["conflicting_reads"]);
});

test("adjudication: a mismatch or an over-maximum mark always triggers; a reconciled paper never does", () => {
  const regions = [{ order_index: 0, label: "1", awarded: 3, available: 4, recognition: "high" as const }];
  assert.deepEqual(adjudicationTriggers(reconcile(regions, 9, null), regions, false), ["total_mismatch"]);
  const over = [{ order_index: 0, label: "1", awarded: 5, available: 4, recognition: "high" as const }];
  assert.deepEqual(adjudicationTriggers(reconcile(over, null, null), over, false), ["mark_above_maximum"]);
  assert.deepEqual(adjudicationTriggers(reconcile(regions, 3, 4), regions, true), []);
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
