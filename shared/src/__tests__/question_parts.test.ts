import { test } from "node:test";
import assert from "node:assert/strict";
import { fullMarkPreviousContext } from "../question_parts.js";

const part = (
  label: string,
  orderIndex: number,
  marksAwarded: number | null,
  marksAvailable: number | null,
  studentAnswer = "answer",
) => ({
  label,
  key: null,
  questionText: `Question ${label}`,
  studentAnswer,
  marksAwarded,
  marksAvailable,
  orderIndex,
});

test("uses the immediately preceding full-mark letter part as setup", () => {
  const previous = part("a", 4, 3, 3, "Po(2.92)");
  const result = fullMarkPreviousContext("b", 5, [previous]);
  assert.equal(result?.label, "a");
  assert.equal(result?.studentAnswer, "Po(2.92)");
});

test("uses the previous roman subpart only inside the same parent label", () => {
  const previous = part("1(a)(i)", 4, 2, 2);
  assert.equal(fullMarkPreviousContext("1(a)(ii)", 5, [previous])?.label, "1(a)(i)");
  assert.equal(fullMarkPreviousContext("2(a)(ii)", 5, [previous]), null);
});

test("never carries a partly wrong earlier answer into the next part", () => {
  const previous = part("a", 4, 2, 3, "wrong setup");
  assert.equal(fullMarkPreviousContext("b", 5, [previous]), null);
});

test("does not jump over an intervening region", () => {
  const earlier = part("a", 3, 2, 2);
  const intervening = part("x", 4, 1, 1);
  assert.equal(fullMarkPreviousContext("b", 5, [earlier, intervening]), null);
});
