/**
 * AXO-116 regressions: the page-local half of the structure stage.
 *
 * Fixtures are synthetic. The label sequences mirror production run 89c8d7a9
 * (pages structured out of order; bare part labels `a`, `b` on several pages),
 * which lost 6 of 14 pages to a label-index collision and a misleading
 * "could not read this page" message.
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import {
  pageFurnitureLabel,
  planPage,
  regionsWrittenByPage,
  structureFailureReason,
  uniqueLabelKey,
} from "../structure_plan.js";
import { ModelError } from "../model-client.js";
import type { StructureRegion } from "../prompts/structure.v1.js";

const W = 2000;
const H = 2800;

function region(label: string | null, y: number, continues = false): StructureRegion {
  return {
    candidate_number: label,
    number_box: label ? { x: 40, y, w: 30, h: 20 } as never : null,
    box: { x: 30, y, w: 900, h: 150 } as never,
    continues_from_previous: continues,
    structure_confidence: "high",
  };
}

function plan(page: number, regions: StructureRegion[], taken: string[] = [], nextIndex = 0) {
  return planPage({
    regions,
    page,
    width: W,
    height: H,
    runId: "run",
    paperId: "paper",
    studentId: "student",
    nextIndex,
    takenLabels: new Set(taken),
  });
}

test("uniqueLabelKey: only numbered labels are unique within a run", () => {
  assert.equal(uniqueLabelKey("a"), null);
  assert.equal(uniqueLabelKey("(b)"), null);
  assert.equal(uniqueLabelKey("(ii)"), null);
  assert.equal(uniqueLabelKey(null), null);
  assert.equal(uniqueLabelKey("2. a)"), "2a");
  assert.equal(uniqueLabelKey("2a"), "2a");
  assert.equal(uniqueLabelKey("Q5"), "q5");
});

test("bare part labels repeated on a later page are kept, not rejected (run 89c8d7a9)", () => {
  // Page 8 was structured first and stored `c`, `d`; page 12 stored `a`, `b`.
  // Page 2 also prints `(a)`, `(b)` under its own question.
  const taken = ["a", "b", "c", "d"].map(uniqueLabelKey).filter((k): k is string => k !== null);
  const out = plan(2, [region("(a)", 100), region("(b)", 500)], taken, 4);
  assert.equal(out.length, 2);
  assert.deepEqual(out.map((r) => r.row.question_label), ["(a)", "(b)"]);
  assert.ok(out.every((r) => r.withheld_label === undefined));
  assert.deepEqual(out.map((r) => r.order_index), [4, 5]);
});

test("a numbered label already used in the run is withheld and sent to review; the region stays", () => {
  const out = plan(5, [region("4a", 100), region("4b", 500)], ["4a"]);
  assert.equal(out.length, 2);
  assert.equal(out[0].row.question_label, null);
  assert.equal(out[0].row.question_label_box, null);
  assert.equal(out[0].row.needs_review, true);
  assert.equal(out[0].withheld_label, "4a");
  assert.equal(out[1].row.question_label, "4b");
  assert.equal(out[1].row.needs_review, undefined);
});

test("a numbered label repeated within one page is withheld on its second appearance", () => {
  const out = plan(3, [region("3", 100), region("3.", 600)]);
  assert.equal(out[0].row.question_label, "3");
  assert.equal(out[1].row.question_label, null);
  assert.equal(out[1].row.needs_review, true);
});

test("a continuation band is written page-locally and flagged, never stitched here", () => {
  const out = plan(3, [region(null, 0, true), region("4", 700)], [], 7);
  assert.equal(out.length, 2);
  assert.equal(out[0].row.continues_from_previous, true);
  assert.equal(out[0].row.question_label, null);
  assert.deepEqual(out[0].row.page_spans, [out[0].span]);
  assert.equal(out[0].span.page, 3);
  assert.equal(out[1].row.continues_from_previous, false);
});

test("only the top band can be a continuation", () => {
  const out = plan(3, [region("4", 0), region(null, 700, true)]);
  assert.deepEqual(out.map((r) => r.row.continues_from_previous), [false, false]);
});

test("a continuation's own printed number is not taken as its label", () => {
  const out = plan(3, [region("2", 0, true)], ["2"]);
  assert.equal(out[0].row.question_label, null);
  assert.equal(out[0].withheld_label, undefined);
  assert.equal(out[0].row.needs_review, undefined);
});

test("regionsWrittenByPage selects exactly the regions whose first span is that page", () => {
  const rows = [
    { id: "p3-a", page_spans: [{ page: 3, box: {} }] },
    { id: "p1-merged", page_spans: [{ page: 1, box: {} }, { page: 3, box: {} }] },
    { id: "p4", page_spans: [{ page: 4, box: {} }] },
    { id: "broken", page_spans: null },
  ];
  assert.deepEqual(regionsWrittenByPage(rows, 3), ["p3-a"]);
});

test("only a model failure is reported as an unreadable page", () => {
  assert.match(structureFailureReason(new ModelError("bad_shape", "x")), /could not read this page/);
  const db = structureFailureReason(Object.assign(new Error("duplicate key"), { code: "23505" }));
  assert.doesNotMatch(db, /could not read/);
  assert.match(db, /Your page is kept/);
});

// Page furniture (structure.v2). Synthetic boxes on the model's 0-1000 grid: a
// Cambridge-style page prints a lone page number at the top centre.
const printed = (value: string, box = { x: 490, y: 20, w: 20, h: 18 }) => ({ value, box: box as never, page_index: 0 });
const labelled = (label: string, numberBox: { x: number; y: number; w: number; h: number }, y = 60): StructureRegion => ({
  candidate_number: label,
  number_box: numberBox as never,
  box: { x: 30, y, w: 900, h: 200 } as never,
  continues_from_previous: false,
  structure_confidence: "high",
});

test("pageFurnitureLabel: the printed page number at the top centre is not a question number", () => {
  assert.equal(pageFurnitureLabel(labelled("6", { x: 490, y: 20, w: 20, h: 18 }), printed("6")), true);
  // Footer band too.
  assert.equal(pageFurnitureLabel(labelled("6", { x: 488, y: 960, w: 24, h: 18 }), printed("6", { x: 488, y: 960, w: 24, h: 18 })), true);
});

test("pageFurnitureLabel: a genuine question 6 on printed page 6, at the left margin, is kept", () => {
  assert.equal(pageFurnitureLabel(labelled("6", { x: 60, y: 90, w: 20, h: 20 }), printed("6")), false);
  // Same number, but nowhere near the header or footer.
  assert.equal(pageFurnitureLabel(labelled("6", { x: 480, y: 500, w: 20, h: 20 }), printed("6")), false);
});

test("pageFurnitureLabel: the page number read into a continued part label is refused only when the boxes overlap", () => {
  // "6(b)" whose number box swallowed the page-number glyph.
  assert.equal(pageFurnitureLabel(labelled("6(b)", { x: 60, y: 15, w: 460, h: 60 }), printed("6")), true);
  // "6(b)" printed at the left margin, away from the page number: not refused here.
  assert.equal(pageFurnitureLabel(labelled("6(b)", { x: 60, y: 300, w: 40, h: 20 }), printed("6")), false);
});

test("pageFurnitureLabel: says nothing without a printed page number, or when the numbers differ", () => {
  assert.equal(pageFurnitureLabel(labelled("6", { x: 490, y: 20, w: 20, h: 18 }), null), false);
  assert.equal(pageFurnitureLabel(labelled("6", { x: 490, y: 20, w: 20, h: 18 }), undefined), false);
  assert.equal(pageFurnitureLabel(labelled("3", { x: 490, y: 20, w: 20, h: 18 }), printed("6")), false);
  assert.equal(pageFurnitureLabel(labelled("6", { x: 490, y: 20, w: 20, h: 18 }), printed("vi")), false);
  assert.equal(pageFurnitureLabel(labelled("(b)", { x: 60, y: 20, w: 30, h: 20 }), printed("6")), false);
});

test("planPage withholds a page-number label and sends the region to review, keeping the region", () => {
  const out = planPage({
    regions: [labelled("6", { x: 490, y: 20, w: 20, h: 18 }), labelled("4", { x: 60, y: 400, w: 20, h: 20 }, 400)],
    page: 6,
    width: W,
    height: H,
    runId: "run",
    paperId: "paper",
    studentId: "student",
    nextIndex: 0,
    takenLabels: new Set(),
    printedPageNumber: printed("6"),
  });
  assert.equal(out.length, 2);
  assert.equal(out[0]!.row.question_label, null);
  assert.equal(out[0]!.row.needs_review, true);
  assert.equal(out[0]!.withheld_label, "6");
  assert.equal(out[1]!.row.question_label, "4");
  assert.equal(out[1]!.row.needs_review, undefined);
});

test("planPage without a printed page number behaves as before", () => {
  const out = plan(6, [labelled("6", { x: 490, y: 20, w: 20, h: 18 })]);
  assert.equal(out[0]!.row.question_label, "6");
});
