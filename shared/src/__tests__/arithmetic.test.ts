import { test } from "node:test";
import assert from "node:assert/strict";
import { checkAnswer, checkChain, rat, ratEq, ratStr } from "../arithmetic.js";

/**
 * Addendum C §6 — the regression suite, seeded from the live database.
 *
 * Every string here is a verbatim `question_region.student_answer` from project
 * dlgcqieyevoebefhcggi, read 2026-09-06. Each one currently has a model-judged
 * `arithmetic` signal that is wrong, non-deterministic, or both.
 *
 * Determinism is the property under test. A checker that is right 95% of the
 * time and different on each run is not 95% good, it is unusable, because you
 * cannot tell which 95%. So every case is asserted many times over.
 */

const RUNS = 25;
function stable<T>(f: () => T): T {
  const first = JSON.stringify(f());
  for (let i = 1; i < RUNS; i++) {
    assert.equal(JSON.stringify(f()), first, "verdict changed between identical runs");
  }
  return f();
}

// ── §6.1 ────────────────────────────────────────────────────────────────────
// Live: `arithmetic` read true, true, true, true, false across five rows
// holding this identical string. 4 + 1/2 is 9/2; 8 + 1/2 is 17/2.

const Q_A = "Working 4.5 = 4+1/2 = 8+1/2 = 9/2 = 0.1001 * 2^3 = 9/16 * 2^3 = (1/16 + 1/2) * 2^3 = 010010000000 0011";

test("§6.1 the false step is found, and found identically every time", () => {
  const v = stable(() => checkAnswer(Q_A));
  assert.equal(v.kind, "inconsistent");
  if (v.kind !== "inconsistent") return;
  assert.equal(v.leftValue, "9/2", "4 + 1/2");
  assert.equal(v.rightValue, "17/2", "8 + 1/2");
});

test("§6.1 the chain alone, without the prose prefix", () => {
  const v = checkChain("4.5 = 4+1/2 = 8+1/2 = 9/2");
  assert.equal(v.kind, "inconsistent");
  if (v.kind !== "inconsistent") return;
  assert.equal(v.at, 2, "the third segment is where it breaks");
});

// ── §6.2 ────────────────────────────────────────────────────────────────────
// Live: false in one row, true in another. It is arithmetically sound: the
// student carries 3/16 forward and multiplies. Strict transitivity would call
// that false; the continuation rule reads what was actually written.

test("§6.2 a correct running chain is consistent", () => {
  const v = stable(() => checkAnswer("Working 1/8 + 1/16 = 3/16 * 2^5 = 3/16 * 32 = 6"));
  assert.equal(v.kind, "consistent");
});

test("§6.2 continuation is earned by value, not assumed", () => {
  // 1/8 + 1/16 is 3/16, and the next segment contains 3/16 — carried forward.
  assert.equal(checkChain("1/8 + 1/16 = 3/16 * 2^5").kind, "consistent");
  // 5/16 appears nowhere in the previous segment's values, so this is not a
  // continuation and not an equality. It is simply wrong.
  assert.equal(checkChain("1/8 + 1/16 = 5/16 * 2^5").kind, "inconsistent");
});

// ── §6.3 ────────────────────────────────────────────────────────────────────
// Live: `3/16 | 2^5` was marked arithmetic true, four rows out of five. The `|`
// is not an operator and not a character the student wrote.

test("§6.3 an unparseable segment is unknown, never true", () => {
  const v = stable(() => checkAnswer("Working... 1/8 + 1/16 = 3/16 | 2^5\n3/16 * 32 = 6"));
  // The second line is checkable and sound, so the answer as a whole is not
  // unknown — but the malformed line must never have been read as true.
  assert.notEqual(v.kind, "inconsistent", "the sound line must not be called false");
  const line = checkChain("1/8 + 1/16 = 3/16 | 2^5");
  assert.equal(line.kind, "unknown", "a `|` makes the segment unreadable");
});

test("§6.3 unknown is a real verdict, distinct from false", () => {
  for (const junk of ["1 = 2 @ 3", "x + 1 = 2", "1 = ", "= 5", "1/0 = 2", "2^(1/2) = 1.414"]) {
    const v = checkChain(junk);
    assert.equal(v.kind, "unknown", `${junk} should be unknown, got ${v.kind}`);
  }
});

test("prose carries no arithmetic claim at all", () => {
  // Live rows: "Not Normalized" had `arithmetic` false on one row and true on
  // four others. It contains no arithmetic. The only honest verdict is unknown.
  for (const prose of [
    "Not Normalized",
    "Not Normalised",
    "Can be written in just 3 bits for each 0.1 for +ve 1.0 for -ve",
    "A user defined data type is used for creating custom data types as per the programs need",
  ]) {
    assert.equal(stable(() => checkAnswer(prose)).kind, "unknown", prose.slice(0, 40));
  }
});

// ── §6.6 ────────────────────────────────────────────────────────────────────
// The handwritten `8/2` was stored as `8+1`, which turned a correct step into a
// false one. Once the transcription is right, the chain must be consistent —
// which is what makes the inconsistency above a re-read trigger rather than a
// verdict on the student.

test("§6.6 the same chain is consistent once 8/2 is transcribed correctly", () => {
  const v = stable(() => checkChain("4.5 = 4+1/2 = 8/2+1/2 = 9/2"));
  assert.equal(v.kind, "consistent");
});

// ── exactness ───────────────────────────────────────────────────────────────

test("rationals are exact, so no float rounding decides a student's mark", () => {
  // 0.1 + 0.2 === 0.30000000000000004 in floating point. Not here.
  assert.equal(checkChain("0.1 + 0.2 = 0.3").kind, "consistent");
  assert.equal(checkChain("1/3 + 1/3 + 1/3 = 1").kind, "consistent");
  assert.equal(checkChain("1/3 = 0.333").kind, "inconsistent");
  assert.ok(ratEq(rat(2n, 4n), rat(1n, 2n)), "rationals normalise");
  assert.equal(ratStr(rat(9n, 2n)), "9/2");
});

test("a declared radix is honoured; an undeclared one is not invented", () => {
  // 100.1 in binary is 4 + 1/2. The paper's own notation says so.
  assert.equal(checkChain("100.1_2 = 4.5").kind, "consistent");
  assert.equal(checkChain("100.1_2 = 4.5_10").kind, "consistent");
  assert.equal(checkChain("1111_2 = 15").kind, "consistent");
  // A bare 0.1001 is decimal, because that is what the characters say. Reading
  // it as a binary mantissa would be inventing a convention the text does not
  // declare — which is the habit this whole module exists to end.
  assert.equal(checkChain("0.1001 = 9/16").kind, "inconsistent");
  assert.equal(checkChain("0.1001_2 = 9/16").kind, "consistent");
});

test("operators as students actually write them", () => {
  assert.equal(checkChain("3 × 4 = 12").kind, "consistent");
  assert.equal(checkChain("12 ÷ 4 = 3").kind, "consistent");
  assert.equal(checkChain("2^3 = 8").kind, "consistent");
  assert.equal(checkChain("-(3 - 5) = 2").kind, "consistent");
  assert.equal(checkChain("(1/16 + 1/2) * 2^3 = 4.5").kind, "consistent");
});

test("a runaway exponent cannot hang the pipeline", () => {
  assert.equal(checkChain("2^99999999 = 1").kind, "unknown");
});

test("multi-line answers: one bad line makes the answer inconsistent", () => {
  assert.equal(checkAnswer("1 + 1 = 2\n2 + 2 = 5").kind, "inconsistent");
  assert.equal(checkAnswer("1 + 1 = 2\n2 + 2 = 4").kind, "consistent");
  assert.equal(checkAnswer("some prose\nmore prose").kind, "unknown");
  assert.equal(checkAnswer(null).kind, "unknown");
  assert.equal(checkAnswer("").kind, "unknown");
});
