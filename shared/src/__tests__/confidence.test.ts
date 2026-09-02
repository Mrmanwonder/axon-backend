import { test } from "node:test";
import assert from "node:assert/strict";
import { assess, numberingSoundness, downgradeRecognition } from "../confidence.js";

test("assess: all signals pass -> confident", () => {
  const { tier } = assess({
    recognition: "high",
    numberingSound: true,
    arithmeticOk: true,
    awarded: 3,
    available: 4,
    unreadable: false,
  });
  assert.equal(tier, "confident");
});

test("assess: arithmeticOk false keeps a region at unsure even when every other signal passes (§6.3 — but only for the region it applies to; a caller no longer has to fail every region on the paper for this)", () => {
  const { tier } = assess({
    recognition: "high",
    numberingSound: true,
    arithmeticOk: false,
    awarded: 3,
    available: 4,
    unreadable: false,
  });
  assert.equal(tier, "unsure");
});

test("assess: unreadable always wins regardless of other signals", () => {
  const { tier } = assess({
    recognition: "high",
    numberingSound: true,
    arithmeticOk: true,
    awarded: 3,
    available: 4,
    unreadable: true,
  });
  assert.equal(tier, "unreadable");
});

test("assess: low recognition never reaches confident", () => {
  const { tier } = assess({
    recognition: "low",
    numberingSound: true,
    arithmeticOk: true,
    awarded: 3,
    available: 4,
    unreadable: false,
  });
  assert.equal(tier, "unsure");
});

test("downgradeRecognition: steps high -> medium -> low, and low stays low", () => {
  assert.equal(downgradeRecognition("high"), "medium");
  assert.equal(downgradeRecognition("medium"), "low");
  assert.equal(downgradeRecognition("low"), "low");
  assert.equal(downgradeRecognition(null), null);
});

test("assess: a layer-fallback page downgrades recognition one step rather than vetoing the tier outright (§6.3) — high -> medium still reaches confident", () => {
  const { tier } = assess({
    recognition: downgradeRecognition("high"),
    numberingSound: true,
    arithmeticOk: true,
    awarded: 3,
    available: 4,
    unreadable: false,
  });
  assert.equal(tier, "confident");
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
