import { test } from "node:test";
import assert from "node:assert/strict";
import { planContentSources } from "../content_sources.js";
import { frameForIndex, mapModelBoxToPage, type ModelFrame } from "../frames.js";

test("an existing first-page crop cannot hide continuation working or its teacher mark", () => {
  const plan = planContentSources([{ page: 4 }, { page: 5 }], true);
  assert.equal(plan.kind, "pages");
  assert.deepEqual(plan.pageNumbers, [4, 5]);
  const frames: ModelFrame[] = plan.pageNumbers.map((pageNumber) => ({
    kind: "page", pageNumber, pageWidth: 2400,
    pageHeight: pageNumber === 4 ? 3200 : 1600,
  }));
  const continuation = frameForIndex(frames, 1);
  assert.ok(continuation);
  assert.deepEqual(mapModelBoxToPage(continuation, { x: 0, y: 500, w: 100, h: 100 }),
    { page: 5, x: 0, y: 800, w: 240, h: 160 });
});

test("one page can retain the crop and repeated bands do not invent a continuation", () => {
  assert.deepEqual(planContentSources([{ page: 2 }, { page: 2 }], true),
    { kind: "crop", pageNumbers: [2] });
});

test("a region without a crop retains all pages in first-seen order", () => {
  assert.deepEqual(planContentSources([{ page: 7 }, { page: 8 }, { page: 7 }], false),
    { kind: "pages", pageNumbers: [7, 8] });
});

test("an empty region never selects an unrelated crop", () => {
  assert.deepEqual(planContentSources([], true), { kind: "pages", pageNumbers: [] });
});
