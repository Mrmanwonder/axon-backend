// The crop geometry, against the failure §8.5 calls out by name: "a crop that
// clips the mark is worse than no crop". Everything here is about what ends up
// inside the frame.

import { test } from "node:test";
import assert from "node:assert/strict";
import { bandForRegion, cutRegion, imageFormat, CROP_INSET_X, type PageSpan } from "../crop.js";
import { pageDimensions } from "../page.js";

const PAGE_W = 2400;
const PAGE_H = 3200;

const span = (page: number, box: { x: number; y: number; w: number; h: number }): PageSpan => ({ page, box });

test("a band spans the full page width, whatever the region's box did", () => {
  // The teacher's total for a question sits in the margin, outside the region
  // box, at a horizontal position nothing in the pipeline records —
  // `margin_band` is computed on the device and never submitted. Full width is
  // the only padding that cannot be too narrow.
  const band = bandForRegion([span(1, { x: 300, y: 800, w: 1500, h: 400 })], 1, PAGE_W, PAGE_H);
  assert.ok(band);
  assert.equal(band.x, 0);
  assert.equal(band.w, PAGE_W);
  assert.equal(CROP_INSET_X, 0, "the inset is what this test is asserting about; changing it changes the guarantee");
});

test("a band is padded above and below the region", () => {
  const box = { x: 300, y: 800, w: 1500, h: 400 };
  const band = bandForRegion([span(1, box)], 1, PAGE_W, PAGE_H)!;
  assert.ok(band.y < box.y, `band starts at ${band.y}, region at ${box.y} — no headroom for a mark written above the answer`);
  assert.ok(band.y + band.h > box.y + box.h, "band ends before the region does");
});

test("padding is generous even for a short region", () => {
  // A one-line question's box is a few dozen pixels tall. Padding by a share of
  // the box alone would give it almost nothing, and a marginal number written
  // slightly high would fall outside.
  const band = bandForRegion([span(1, { x: 300, y: 1000, w: 1500, h: 40 })], 1, PAGE_W, PAGE_H)!;
  assert.ok(band.y <= 1000 - PAGE_H * 0.02,
    `a 40px region got only ${1000 - band.y}px of headroom — a mark written above it would be cut off`);
});

test("padding is clamped to the page rather than running off it", () => {
  const top = bandForRegion([span(1, { x: 0, y: 0, w: PAGE_W, h: 300 })], 1, PAGE_W, PAGE_H)!;
  assert.equal(top.y, 0);
  const bottom = bandForRegion([span(1, { x: 0, y: PAGE_H - 300, w: PAGE_W, h: 300 })], 1, PAGE_W, PAGE_H)!;
  assert.ok(bottom.y + bottom.h <= PAGE_H);
});

test("a region spanning two pages is not cropped", () => {
  // Half an answer is worse than the whole paper: the model would read a
  // truncated answer confidently, having never been shown the rest of it.
  const spans = [span(1, { x: 0, y: 2400, w: 2400, h: 800 }), span(2, { x: 0, y: 0, w: 2400, h: 600 })];
  assert.equal(bandForRegion(spans, 1, PAGE_W, PAGE_H), null);
  assert.equal(bandForRegion(spans, 2, PAGE_W, PAGE_H), null);
});

test("a region on another page is not cropped from this one", () => {
  assert.equal(bandForRegion([span(2, { x: 0, y: 100, w: 2400, h: 400 })], 1, PAGE_W, PAGE_H), null);
});

test("a region covering nearly the whole page is not cropped", () => {
  // It is not a crop, it is a second copy of the page, and content already has
  // the page.
  assert.equal(bandForRegion([span(1, { x: 0, y: 0, w: PAGE_W, h: PAGE_H - 100 })], 1, PAGE_W, PAGE_H), null);
});

test("a missing or nonsensical box produces no band rather than a guessed one", () => {
  assert.equal(bandForRegion([{ page: 1 }], 1, PAGE_W, PAGE_H), null);
  assert.equal(bandForRegion([{ page: 1, box: null }], 1, PAGE_W, PAGE_H), null);
  assert.equal(bandForRegion([span(1, { x: 0, y: 0, w: 0, h: 100 })], 1, PAGE_W, PAGE_H), null);
  assert.equal(bandForRegion([span(1, { x: 0, y: 0, w: NaN, h: 100 })], 1, PAGE_W, PAGE_H), null);
  assert.equal(bandForRegion([], 1, PAGE_W, PAGE_H), null);
});

test("cutRegion copies exactly the pixels the band names", () => {
  // A 4x4 image where each pixel's red channel is its own index, so a shifted
  // or transposed copy is visible rather than plausible.
  const width = 4, height = 4;
  const data = new Uint8ClampedArray(width * height * 4);
  for (let i = 0; i < width * height; i++) {
    data[i * 4] = i;
    data[i * 4 + 3] = 255;
  }
  const cut = cutRegion({ data, width, height }, { x: 1, y: 1, w: 2, h: 2 });
  assert.equal(cut.width, 2);
  assert.equal(cut.height, 2);
  assert.deepEqual([cut.data[0], cut.data[4], cut.data[8], cut.data[12]], [5, 6, 9, 10]);
});

test("cutRegion clamps a band that runs past the image rather than reading past it", () => {
  const data = new Uint8ClampedArray(4 * 4 * 4).fill(255);
  const cut = cutRegion({ data, width: 4, height: 4 }, { x: 3, y: 3, w: 100, h: 100 });
  assert.equal(cut.width, 1);
  assert.equal(cut.height, 1);
});

test("format is sniffed from the bytes, not assumed", () => {
  assert.equal(imageFormat(new Uint8Array([0xff, 0xd8, 0xff, 0xe0, 0, 0, 0, 0, 0, 0, 0, 0])), "jpeg");
  assert.equal(imageFormat(new Uint8Array([0x89, 0x50, 0x4e, 0x47, 13, 10, 26, 10, 0, 0, 0, 0])), "png");
  assert.equal(imageFormat(new Uint8Array([0x52, 0x49, 0x46, 0x46, 0, 0, 0, 0, 0x57, 0x45, 0x42, 0x50])), "webp");
  assert.equal(imageFormat(new Uint8Array([0x52, 0x49, 0x46, 0x46, 0, 0, 0, 0, 0x57, 0x41, 0x56, 0x45])), null, "a RIFF WAVE is not a WebP");
  assert.equal(imageFormat(new Uint8Array([1, 2, 3])), null);
});

// ── the dimensions the bands are cut against ───────────────────────────────

test("recorded dimensions are used as they are", () => {
  const dims = pageDimensions({ conditioning_meta: { width: 2250, height: 3301 } });
  assert.deepEqual(dims, { width: 2250, height: 3301, source: "recorded" });
});

test("dimensions are derived from the source shape and the conditioned long edge", () => {
  // The pages written before conditioning recorded its own output size. The
  // source's aspect plus the long edge that came out determine the page exactly,
  // because conditioning scales both axes by one factor.
  const dims = pageDimensions({
    conditioning_meta: { source_size: { width: 3024, height: 4032 } },
    quality_signals: { long_edge: 2400 },
  });
  assert.equal(dims?.source, "derived");
  assert.equal(dims?.height, 2400);
  assert.equal(dims?.width, 1800);
});

test("dimensions that cannot be established come back null, never as a default", () => {
  // The whole reason this module exists: `meta.width ?? 2400` was not a
  // fallback, it was the only branch, and it mis-scaled every box on every page.
  assert.equal(pageDimensions({}), null);
  assert.equal(pageDimensions({ conditioning_meta: { source_size: { width: 3024, height: 4032 } } }), null);
  assert.equal(pageDimensions({ quality_signals: { long_edge: 2400 } }), null);
  assert.equal(pageDimensions({ conditioning_meta: { width: 0, height: 3200 } }), null);
});
