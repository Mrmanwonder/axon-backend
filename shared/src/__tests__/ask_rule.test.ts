import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { askReasons, assess, paperShownUnmarked, plausible } from "../confidence.js";
import { placeRegions, placementVerdicts, placedLabelText, type PlacementInput } from "../placement.js";
import { judgeRegions, type RegionRow } from "../review_rule.js";
import { schemeQuestionMessages } from "../scheme_check.js";

/* Council D1 (7 Oct 2026): ask only what needs the student. Synthetic rows. */

// ── placement mirrors the site's AXO-122 walk ──────────────────────────────

const contract = JSON.parse(readFileSync(join(dirname(fileURLToPath(import.meta.url)), "fixtures", "question-count-contract.json"), "utf8")) as {
  cases: Array<{ name: string; regions: PlacementInput[]; expected: Record<string, number> }>;
};

for (const c of contract.cases) {
  test(`placement walk matches the site's counting contract: ${c.name}`, () => {
    const placed = placeRegions(c.regions);
    const counted = placed.filter((e) => e.counted);
    assert.deepEqual({
      questions_total: new Set(counted.filter((e) => e.q !== null).map((e) => e.q)).size,
      parts_total: counted.length,
      unassigned_parts: counted.filter((e) => e.unassigned).length,
      raw_region_count: c.regions.length,
    }, c.expected);
  });
}

const r = (label: string | null, page: number, y: number, order_index: number, evidence = true): PlacementInput =>
  ({ label, page, y, order_index, evidence });

test("numbering is judged in page order, not in the order pages finished structure", () => {
  // Stored order: page 2 finished first. Read in page order this is 1, 1(b), 2(a), (b).
  const v = placementVerdicts([r("2(a)", 2, 100, 0), r("(b)", 2, 600, 1), r("1(a)", 1, 100, 2), r("(b)", 1, 600, 3)]);
  assert.deepEqual(v.map((x) => x.structural), [true, true, true, true]);
  assert.deepEqual(v.map((x) => x.unplaceable), [false, false, false, false]);
  assert.deepEqual(v.map((x) => x.placedLabel), ["2(a)", "2(b)", "1(a)", "1(b)"]);
});

test("a bare part placed under its question does not fail numbering", () => {
  const v = placementVerdicts([r("3(a)", 1, 100, 0), r("(b)", 1, 400, 1), r("(c)", 2, 50, 2)]);
  assert.deepEqual(v.map((x) => x.structural), [true, true, true]);
  assert.equal(v[2].placedLabel, "3(c)");
});

test("an unplaceable part fails numbering and is unplaceable: (c)", () => {
  // A null label carrying a mark has no parent the walk can find.
  const v = placementVerdicts([r("1", 1, 100, 0), r(null, 1, 400, 1)]);
  assert.equal(v[1].structural, false);
  assert.equal(v[1].unplaceable, true);
  // A bare part already taken under its question is unassigned.
  const w = placementVerdicts([r("1(a)", 1, 100, 0), r("(a)", 1, 400, 1)]);
  assert.equal(w[1].unplaceable, true);
});

test("two numbered parts placed on the same question+part collide", () => {
  const v = placementVerdicts([r("2(a)", 1, 100, 0), r("2a", 1, 400, 1), r("2(b)", 1, 700, 2)]);
  assert.deepEqual(v.map((x) => x.unplaceable), [true, true, false]);
  assert.deepEqual(v.map((x) => x.structural), [false, false, true]);
});

test("a null label with nothing on it is not a part: numbering unknown, never asked", () => {
  const v = placementVerdicts([r("1", 1, 100, 0), r(null, 1, 400, 1, false)]);
  assert.equal(v[1].structural, "unknown");
  assert.equal(v[1].counted, false);
  assert.equal(v[1].unplaceable, false);
});

test("a question-number jump fails numbering but is not unplaceable", () => {
  const v = placementVerdicts([r("1", 1, 100, 0), r("4", 1, 400, 1)]);
  assert.deepEqual(v.map((x) => x.structural), [true, false]);
  assert.deepEqual(v.map((x) => x.unplaceable), [false, false]);
});

test("placed label text", () => {
  assert.equal(placedLabelText(3, "c"), "3(c)");
  assert.equal(placedLabelText(2, "d(i)"), "2(d)(i)");
  assert.equal(placedLabelText(2, "(ii)"), "2(ii)");
  assert.equal(placedLabelText(4, null), "4");
  assert.equal(placedLabelText(null, "c"), null);
});

// ── plausibility ───────────────────────────────────────────────────────────

test("plausibility: a null mark is false on a marked paper, unknown only on an unmarked one", () => {
  assert.equal(plausible(null, 4), false);
  assert.equal(plausible(2, null), false);
  assert.equal(plausible(null, 4, true), "unknown");
  assert.equal(plausible(null, null, true), "unknown");
});

test("plausibility: impossible values are false even on an unmarked paper", () => {
  assert.equal(plausible(5, 4, true), false);
  assert.equal(plausible(-1, 4, true), false);
  assert.equal(plausible(1, 0, true), false);
  assert.equal(plausible(1.25, 4, true), false);
  assert.equal(plausible(null, -2, true), false);
});

test("plausibility: no cap on marks available", () => {
  assert.equal(plausible(25, 40), true);
  assert.equal(plausible(0.5, 1), true);
});

test("paper shown unmarked only by triage ungraded_paper at high confidence", () => {
  assert.equal(paperShownUnmarked({ triage: { classification: "ungraded_paper", confidence: "high" } }), true);
  assert.equal(paperShownUnmarked({ triage: { classification: "ungraded_paper", confidence: "low" } }), false);
  assert.equal(paperShownUnmarked({ triage: { classification: "graded_exam", confidence: "high" } }), false);
  assert.equal(paperShownUnmarked(null), false);
});

test("assess: an unmarked paper's null mark does not by itself make a part unsure", () => {
  const base = { recognition: "high" as const, numberingSound: true, arithmeticOk: "unknown" as const, awarded: null, available: null, unreadable: false };
  assert.equal(assess({ ...base, paperUnmarked: true }).tier, "confident");
  assert.equal(assess(base).tier, "unsure");
});

// ── ask-rule ───────────────────────────────────────────────────────────────

const ask = (o: Partial<Parameters<typeof askReasons>[0]>) => askReasons({
  unreadable: false, recognition: "high", paperMarked: true, plausibility: true, unplaceable: false, counted: true, ...o,
});

test("ask-rule (a): unreadable, low or null recognition asks", () => {
  assert.deepEqual(ask({ unreadable: true }), ["unreadable"]);
  assert.deepEqual(ask({ recognition: "low" }), ["recognition"]);
  assert.deepEqual(ask({ recognition: null }), ["recognition"]);
  assert.deepEqual(ask({ recognition: "medium" }), []);
  assert.deepEqual(ask({ unreadable: true, counted: false }), ["unreadable"], "(a) applies even to a region that is not a part");
});

test("ask-rule (b): a missing or impossible mark asks on a marked paper only", () => {
  assert.deepEqual(ask({ plausibility: false }), ["teacher_mark"]);
  assert.deepEqual(ask({ plausibility: false, paperMarked: false }), []);
  assert.deepEqual(ask({ plausibility: "unknown", paperMarked: false }), []);
  assert.deepEqual(ask({ plausibility: false, counted: false }), []);
});

test("ask-rule (c): an unplaceable part asks", () => {
  assert.deepEqual(ask({ unplaceable: true }), ["unplaceable"]);
});

test("ask-rule: a confident part does not ask", () => {
  assert.deepEqual(ask({}), []);
});

// ── the worker's per-region decision ───────────────────────────────────────

const row = (o: Partial<RegionRow> & { page?: number; y?: number }): RegionRow => ({
  id: o.id ?? "x",
  order_index: o.order_index ?? 0,
  question_label: o.question_label ?? null,
  marks_awarded: o.marks_awarded === undefined ? 1 : o.marks_awarded,
  marks_available: o.marks_available === undefined ? 2 : o.marks_available,
  confidence_tier: o.confidence_tier ?? "unsure",
  confidence_signals: o.confidence_signals ?? { recognition_confidence: "high" },
  extract_status: o.extract_status ?? "done",
  page_spans: o.page_spans ?? [{ page: o.page ?? 1, box: { x: 0, y: o.y ?? 0, w: 10, h: 10 } }],
  student_answer: o.student_answer ?? "answer",
  question_text: o.question_text ?? null,
});

const judge = (rows: RegionRow[], opts: { paperUnmarked?: boolean; arithmetic?: Array<boolean | "unknown"> } = {}) =>
  judgeRegions(rows, { paperUnmarked: opts.paperUnmarked ?? false, fallbackPages: new Set(), arithmetic: opts.arithmetic ?? rows.map(() => "unknown") });

test("judge: arithmetic false alone leaves the part unsure but does not ask", () => {
  const [v] = judge([row({ id: "a", question_label: "1" })], { arithmetic: [false] });
  assert.equal(v.tier, "unsure");
  assert.equal(v.needs_review, false);
  assert.deepEqual(v.signals.ask, []);
});

test("judge: confident parts never ask", () => {
  const vs = judge([row({ id: "a", question_label: "1(a)", y: 0 }), row({ id: "b", question_label: "(b)", y: 50, order_index: 1 })], { arithmetic: [true, true] });
  assert.deepEqual(vs.map((v) => [v.tier, v.needs_review]), [["confident", false], ["confident", false]]);
  assert.equal(vs[1].signals.placed_label, "1(b)");
});

test("judge: a misread teacher mark on a marked paper asks; the same on an unmarked paper does not", () => {
  const marked = judge([row({ question_label: "1", marks_awarded: 3, marks_available: 2 })]);
  assert.deepEqual(marked[0].ask, ["teacher_mark"]);
  const missing = judge([row({ question_label: "1", marks_awarded: null })]);
  assert.deepEqual(missing[0].ask, ["teacher_mark"]);
  const unmarked = judge([row({ question_label: "1", marks_awarded: null, marks_available: null })], { paperUnmarked: true });
  assert.equal(unmarked[0].needs_review, false);
  assert.equal(unmarked[0].signals.plausibility, "unknown");
});

test("judge: unreadable and failed regions ask", () => {
  const vs = judge([
    row({ id: "u", question_label: "1", confidence_tier: "unreadable", marks_awarded: null, marks_available: null, student_answer: null }),
    row({ id: "f", question_label: "2", extract_status: "failed", y: 10, order_index: 1 }),
  ]);
  assert.deepEqual(vs.map((v) => v.tier), ["unreadable", "unreadable"]);
  assert.ok(vs.every((v) => v.needs_review && v.ask.includes("unreadable")));
});

test("judge: a part whose label clashes after placement asks", () => {
  const vs = judge([row({ id: "a", question_label: "2(a)", y: 0 }), row({ id: "b", question_label: "2a", y: 50, order_index: 1 })]);
  assert.ok(vs.every((v) => v.ask.includes("unplaceable")));
});

// ── scheme check uses the placed label ─────────────────────────────────────

test("scheme check finds a bare part's section by its placed label", () => {
  const sections = new Map([[3, "3(a) ... 3(c) scheme rows"]]);
  const rows = [
    { id: "a", order_index: 0, question_label: "3(a)", page_spans: [{ page: 1, box: { y: 0 } }], student_answer: "x" },
    { id: "c", order_index: 1, question_label: "(c)", page_spans: [{ page: 2, box: { y: 0 } }], student_answer: "y" },
    { id: "u", order_index: 2, question_label: "(d)", page_spans: [{ page: 2, box: { y: 90 } }], confidence_tier: "unreadable" },
  ];
  const messages = schemeQuestionMessages("run", rows, sections, "9231/11", null);
  assert.deepEqual(messages.map((m) => m.scheme_check_q.region_id), ["a", "c"]);
  assert.equal(messages[1].scheme_check_q.label, "3(c)");
  assert.equal(messages[1].scheme_check_q.section, "3(a) ... 3(c) scheme rows");
});
