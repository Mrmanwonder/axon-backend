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
