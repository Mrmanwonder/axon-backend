/**
 * A teacher mark on a continuation page belongs to the question that continued.
 *
 * Re-audit P0-G. The structure worker stitches a page's first band onto the
 * previous question when the model says `continues_from_previous`, and then
 * `continue`d — so that question never entered the array mark attribution
 * looks at. Only the questions newly created on this page were candidates.
 *
 * The dangerous outcome is not the missing attribution. It is the near miss:
 * with the real question absent, the nearest-region fallback in
 * `assignToRegion` hands the mark to whatever else is on the page, and a mark
 * on the wrong question is plausible, confident and wrong — the exact shape
 * hard rule 1 exists to prevent.
 *
 * These tests drive `attribute` directly with the two candidate sets the
 * worker can build, because the bug is entirely in which regions it is given.
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import { attribute, type RawMark } from "../attribution.js";

const PAGE = 2;
const WIDTH = 2400;

/** Page 2 of a booklet: the tail of Q3 across the top, Q4 below it. */
const q3Tail = { page: PAGE, box: { x: 100, y: 0, w: 2000, h: 600 } };
const q4 = { page: PAGE, box: { x: 100, y: 700, w: 2000, h: 900 } };

/* The teacher's marginal numbers. A glyph inside the margin band classifies as
   `marginal_number` — the shape and the band do the work, since RawMark carries
   no class or value of its own. They are identified below by their y position:
   one beside the Q3 tail, one beside Q4. */
const MARGIN = { x0: 2150, x1: WIDTH };

const markBesideQ3: RawMark = {
  page: PAGE,
  box: { page: PAGE, x: 2200, y: 200, w: 90, h: 90 },
  shape: "glyph",
  metrics: {},
};

const markBesideQ4: RawMark = {
  page: PAGE,
  box: { page: PAGE, x: 2200, y: 1000, w: 90, h: 90 },
  shape: "glyph",
  metrics: {},
};

const run = (regions: Array<{ order_index: number; label: null; spans: unknown[] }>) =>
  attribute({
    regions: regions as never,
    marks: [markBesideQ3, markBesideQ4],
    marginBands: new Map([[PAGE, MARGIN]]),
    pageWidths: new Map([[PAGE, WIDTH]]),
  }).filter((m) => m.mark_class === "marginal_number");

/** The mark beside the Q3 tail (top of the page) and the one beside Q4. */
const besideQ3 = (out: ReturnType<typeof run>) => out.find((m) => m.box.y === 200);
const besideQ4 = (out: ReturnType<typeof run>) => out.find((m) => m.box.y === 1000);

test("the old candidate set sends the continuation's mark to the wrong question", () => {
  // What the worker built before: only regions created on this page. Q3 is
  // stitched and absent.
  const attributed = run([{ order_index: 0, label: null, spans: [q4] }]);

  const four = besideQ3(attributed);
  assert.ok(four, "the mark is still detected");
  // Q4 is the only candidate, so the nearest-region fallback gives it the 4
  // that belongs to Q3. This assertion documents the bug rather than endorsing
  // it: if a future change makes this null instead, that is an improvement and
  // this test should be updated deliberately, not silently.
  assert.equal(four!.region_index, 0, "with only Q4 present, Q3's mark lands on Q4");
});

test("including the stitched question puts each mark on its own question", () => {
  // What the worker builds now: the stitched prior first (a continuation is
  // always the top band), then the questions created on this page.
  const attributed = run([
    { order_index: 0, label: null, spans: [q3Tail] },
    { order_index: 1, label: null, spans: [q4] },
  ]);

  const four = besideQ3(attributed);
  const two = besideQ4(attributed);
  assert.equal(four!.region_index, 0, "the 4 belongs to the continued question");
  assert.equal(two!.region_index, 1, "and the 2 to the question that starts on this page");
});

test("the stitched question is offered only its band on THIS page", () => {
  // Its spans on earlier pages must not be handed to per-page attribution: a
  // margin mark on page 2 compared against a box on page 1 is a comparison
  // between two different sheets of paper. The worker passes `[span]` — this
  // page's band — and never the accumulated page_spans.
  const q3OnPageOne = { page: 1, box: { x: 100, y: 1800, w: 2000, h: 700 } };
  const attributed = run([
    { order_index: 0, label: null, spans: [q3OnPageOne, q3Tail] },
    { order_index: 1, label: null, spans: [q4] },
  ]);
  // Still correct here, because assignToRegion filters by page itself — the
  // point of the assertion is that carrying extra pages cannot move a mark.
  assert.equal(besideQ3(attributed)!.region_index, 0);
  assert.equal(besideQ4(attributed)!.region_index, 1);
});

test("a page with no questions at all attributes nothing rather than guessing", () => {
  const attributed = run([]);
  for (const m of attributed) {
    assert.equal(m.region_index, null, "no candidates means no attribution, not a nearest guess");
  }
});
