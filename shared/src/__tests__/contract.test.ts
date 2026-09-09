import { test } from "node:test";
import assert from "node:assert/strict";
import { takeBox } from "../contract.js";

test("takeBox: converts a normalised 0-1000 box to pixel coordinates", () => {
  const box = takeBox({ x: 100, y: 200, w: 50, h: 25 }, 1, 2000, 3000);
  assert.deepEqual(box, { page: 1, x: 200, y: 600, w: 100, h: 75 });
});

test("takeBox: rejects a box with a non-finite dimension rather than guessing", () => {
  assert.equal(takeBox({ x: 0, y: 0, w: NaN, h: 10 }, 1, 2000, 3000), null);
});

test("takeBox: rejects a zero-area box", () => {
  assert.equal(takeBox({ x: 0, y: 0, w: 0, h: 10 }, 1, 2000, 3000), null);
});

test("takeBox: rejects a box that runs off the 0-1000 grid", () => {
  assert.equal(takeBox({ x: 990, y: 0, w: 50, h: 10 }, 1, 2000, 3000), null);
});

test("takeBox: rejects null input outright", () => {
  assert.equal(takeBox(null, 1, 2000, 3000), null);
});

// Re-audit §12/INV-07: a coordinate must be inside the grid at BOTH ends.
// takeBox checked only the far edges, so a negative box became a negative
// pixel offset and the review UI would highlight off the page.
test("takeBox: a negative coordinate is not a box", () => {
  assert.equal(takeBox({ x: -1, y: 10, w: 100, h: 100 }, 1, 2400, 3200), null);
  assert.equal(takeBox({ x: 10, y: -50, w: 100, h: 100 }, 1, 2400, 3200), null);
  // Still accepts the legitimate origin.
  assert.ok(takeBox({ x: 0, y: 0, w: 100, h: 100 }, 1, 2400, 3200));
});
