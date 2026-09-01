import { test } from "node:test";
import assert from "node:assert/strict";
import { assess, numberingSoundness } from "../confidence.js";

test("assess: all signals pass -> confident", () => {
  const { tier } = assess({
    recognition: "high",
    numberingSound: true,
    paperReconciled: true,
    awarded: 3,
    available: 4,
    layerFallback: false,
    unreadable: false,
  });
  assert.equal(tier, "confident");
});

test("assess: layerFallback vetoes confidence even when every other signal passes", () => {
  const { tier } = assess({
    recognition: "high",
    numberingSound: true,
    paperReconciled: true,
    awarded: 3,
    available: 4,
    layerFallback: true,
    unreadable: false,
  });
  assert.equal(tier, "unsure");
});

test("assess: unreadable always wins regardless of other signals", () => {
  const { tier } = assess({
    recognition: "high",
    numberingSound: true,
    paperReconciled: true,
    awarded: 3,
    available: 4,
    layerFallback: false,
    unreadable: true,
  });
  assert.equal(tier, "unreadable");
});

test("assess: low recognition never reaches confident", () => {
  const { tier } = assess({
    recognition: "low",
    numberingSound: true,
    paperReconciled: true,
    awarded: 3,
    available: 4,
    layerFallback: false,
    unreadable: false,
  });
  assert.equal(tier, "unsure");
});

test("numberingSoundness: a sequential run is all sound", () => {
  assert.deepEqual(numberingSoundness(["1", "2", "3"]), [true, true, true]);
});

test("numberingSoundness: a gap breaks soundness for the jump, not everything after", () => {
  // 1 -> 2 is sound, 2 -> 5 is not (skips 3, 4), 5 -> 6 is sound again.
  assert.deepEqual(numberingSoundness(["1", "2", "5", "6"]), [true, true, false, true]);
});

test("numberingSoundness: an unreadable label (null) is never sound, and breaks the chain for what follows it", () => {
  // "3" after a null looks back past the null to "1" as the last known
  // number, and 3 is neither equal to 1 nor one more than it.
  assert.deepEqual(numberingSoundness(["1", null, "3"]), [true, false, false]);
});
