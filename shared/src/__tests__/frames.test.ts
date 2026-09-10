/**
 * Provenance means the highlighted pixels are the pixels the model read.
 *
 * Re-audit P0-H. The numbers in the first test are the audit's own worked
 * example, which is the point: a crop band low on the page, a box in the
 * middle of that crop, and the two answers — the right one and the one the
 * old code produced — are hundreds of pixels apart.
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import { mapModelBoxToPage, frameForIndex, type ModelFrame } from "../frames.js";
import { takeBox } from "../contract.js";

const PAGE_W = 2400;
const PAGE_H = 3200;

const pageFrame: ModelFrame = { kind: "page", pageNumber: 1, pageWidth: PAGE_W, pageHeight: PAGE_H };

/** A question band low on the page: full width, 600px tall, starting at y=1400. */
const cropFrame: ModelFrame = {
  kind: "crop",
  pageNumber: 1,
  pageWidth: PAGE_W,
  pageHeight: PAGE_H,
  band: { x: 0, y: 1400, w: 2400, h: 600 },
};

test("a box on a full page maps by the page's own dimensions", () => {
  const box = mapModelBoxToPage(pageFrame, { x: 100, y: 200, w: 300, h: 100 });
  assert.deepEqual(box, { page: 1, x: 240, y: 640, w: 720, h: 320 });
});

test("a box on a crop is scaled by the band and offset to where the band sits", () => {
  // The audit's worked example:
  //   x = 0    + 100/1000 * 2400 = 240
  //   y = 1400 + 200/1000 *  600 = 1520
  //   w =        300/1000 * 2400 = 720
  //   h =        100/1000 *  600 =  60
  const box = mapModelBoxToPage(cropFrame, { x: 100, y: 200, w: 300, h: 100 });
  assert.deepEqual(box, { page: 1, x: 240, y: 1520, w: 720, h: 60 });
});

test("the old full-page conversion of a crop box lands somewhere else entirely", () => {
  // What the worker actually did with a crop-local box.
  const wrong = takeBox({ x: 100, y: 200, w: 300, h: 100 }, 1, PAGE_W, PAGE_H)!;
  const right = mapModelBoxToPage(cropFrame, { x: 100, y: 200, w: 300, h: 100 })!;

  assert.equal(wrong.y, 640);
  assert.equal(right.y, 1520);
  assert.ok(
    Math.abs(wrong.y - right.y) > 800,
    "the highlight would sit most of a page away from the pixels the model read",
  );
  // X happens to agree here, which is why the bug is easy to miss: these crops
  // are full-width bands, so only the vertical axis is visibly wrong.
  assert.equal(wrong.x, right.x);
});

test("a cropmask maps identically to its crop — same band, different ink", () => {
  const mask: ModelFrame = { ...cropFrame, kind: "cropmask" };
  assert.deepEqual(
    mapModelBoxToPage(mask, { x: 100, y: 200, w: 300, h: 100 }),
    mapModelBoxToPage(cropFrame, { x: 100, y: 200, w: 300, h: 100 }),
  );
});

test("a crop covering the whole page agrees with the page frame", () => {
  // The equivalence that makes turning the crop stage on and off safe: the
  // same field must land in the same place either way.
  const wholePage: ModelFrame = {
    kind: "crop",
    pageNumber: 1,
    pageWidth: PAGE_W,
    pageHeight: PAGE_H,
    band: { x: 0, y: 0, w: PAGE_W, h: PAGE_H },
  };
  for (const raw of [
    { x: 0, y: 0, w: 1000, h: 1000 },
    { x: 100, y: 200, w: 300, h: 100 },
    { x: 750, y: 900, w: 250, h: 100 },
  ]) {
    assert.deepEqual(mapModelBoxToPage(wholePage, raw), mapModelBoxToPage(pageFrame, raw));
  }
});

// ── boxes the model could not have drawn ───────────────────────────────────

test("a negative coordinate is rejected rather than multiplied into one", () => {
  assert.equal(mapModelBoxToPage(pageFrame, { x: -10, y: 200, w: 300, h: 100 }), null);
  assert.equal(mapModelBoxToPage(pageFrame, { x: 100, y: -1, w: 300, h: 100 }), null);
  assert.equal(mapModelBoxToPage(cropFrame, { x: -10, y: 200, w: 300, h: 100 }), null);
});

test("a zero or negative size is not a box", () => {
  assert.equal(mapModelBoxToPage(pageFrame, { x: 10, y: 10, w: 0, h: 100 }), null);
  assert.equal(mapModelBoxToPage(pageFrame, { x: 10, y: 10, w: 100, h: -5 }), null);
});

test("a box off the far edge of the grid is rejected", () => {
  assert.equal(mapModelBoxToPage(pageFrame, { x: 900, y: 10, w: 200, h: 10 }), null);
  assert.equal(mapModelBoxToPage(pageFrame, { x: 1100, y: 10, w: 10, h: 10 }), null);
});

test("a small overshoot is tolerated, because models land a pixel over", () => {
  assert.ok(mapModelBoxToPage(pageFrame, { x: 0, y: 0, w: 1001, h: 1001 }));
});

test("anything that is not four finite numbers is not a box", () => {
  assert.equal(mapModelBoxToPage(pageFrame, null), null);
  assert.equal(mapModelBoxToPage(pageFrame, { x: "10", y: 10, w: 10, h: 10 }), null);
  assert.equal(mapModelBoxToPage(pageFrame, { x: NaN, y: 10, w: 10, h: 10 }), null);
  assert.equal(mapModelBoxToPage(pageFrame, { x: 10, y: 10 }), null);
});

// ── which image was that? ──────────────────────────────────────────────────

test("page_index picks its own frame", () => {
  const two: ModelFrame[] = [
    { kind: "page", pageNumber: 4, pageWidth: 100, pageHeight: 200 },
    { kind: "page", pageNumber: 5, pageWidth: 300, pageHeight: 400 },
  ];
  assert.equal(frameForIndex(two, 0)!.pageNumber, 4);
  assert.equal(frameForIndex(two, 1)!.pageNumber, 5);
  assert.equal(frameForIndex(two, undefined)!.pageNumber, 4, "absent means the first image");
});

test("an out-of-range page_index is rejected, not clamped to the last image", () => {
  // Clamping is how a field the model located on an image we did not send gets
  // recorded confidently on a different page.
  const two: ModelFrame[] = [
    { kind: "page", pageNumber: 4, pageWidth: 100, pageHeight: 200 },
    { kind: "page", pageNumber: 5, pageWidth: 300, pageHeight: 400 },
  ];
  assert.equal(frameForIndex(two, 2), null);
  assert.equal(frameForIndex(two, -1), null);
  assert.equal(frameForIndex(two, 1.5), null);
  assert.equal(frameForIndex([], 0), null);
});

test("each page in a multi-page question maps through its OWN dimensions", () => {
  // The second half of P0-H: the worker read every field's dimensions off the
  // first page, so on a booklet whose pages differ in size a field on page 5
  // was scaled by page 4's height.
  const frames: ModelFrame[] = [
    { kind: "page", pageNumber: 4, pageWidth: 2400, pageHeight: 3200 },
    { kind: "page", pageNumber: 5, pageWidth: 2400, pageHeight: 1600 },
  ];
  const raw = { x: 0, y: 500, w: 100, h: 100 };
  const onFour = mapModelBoxToPage(frameForIndex(frames, 0)!, raw)!;
  const onFive = mapModelBoxToPage(frameForIndex(frames, 1)!, raw)!;
  assert.equal(onFour.y, 1600);
  assert.equal(onFive.y, 800, "half the height means half the pixel offset");
  assert.notEqual(onFour.y, onFive.y);
});
