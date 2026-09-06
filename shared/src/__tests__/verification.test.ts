import { test } from "node:test";
import assert from "node:assert/strict";
import { readSignals, tierFrom, mayReasonOverWorking, transcriptionIsAuthoritative, needsReread } from "../signals.js";
import { canonicalLabel, checkLabels, adjudicationBlocksCommit } from "../labels.js";

// ── signals: unknown is real, and no signal authorises another ───────────────

test("unknown survives the read instead of collapsing to false", () => {
  const s = readSignals({ arithmetic: null, structural: true, recognition: "false", plausibility: undefined });
  assert.equal(s.arithmetic, "unknown");
  assert.equal(s.structural, true);
  assert.equal(s.recognition, false);
  assert.equal(s.plausibility, "unknown");
});

/**
 * The live row this exists for: question `b`, five rows, every one carrying
 * `recognition: false` with `arithmetic: true`. Tidy arithmetic must not make
 * unreadable handwriting authoritative.
 */
test("a clean arithmetic reading cannot rescue unreadable handwriting", () => {
  const s = readSignals({ arithmetic: true, structural: true, recognition: false, plausibility: true });
  assert.equal(tierFrom(s), "unreadable");
  assert.equal(transcriptionIsAuthoritative(s), false);
  assert.equal(mayReasonOverWorking(s), false, "no corrected working may be written against this");
});

test("arithmetic unknown is normal for prose and does not make it unsure", () => {
  // "Not Normalized" has no arithmetic in it. Demanding a verdict would make
  // every essay answer unsure.
  const s = readSignals({ arithmetic: "unknown", structural: true, recognition: true, plausibility: true });
  assert.equal(tierFrom(s), "confident");
  // But it is still not permission to reason over a working chain.
  assert.equal(mayReasonOverWorking(s), false);
});

test("an unknown recognition never reaches confident", () => {
  const s = readSignals({ arithmetic: true, structural: true, recognition: "unknown", plausibility: true });
  assert.equal(tierFrom(s), "unsure");
  assert.equal(transcriptionIsAuthoritative(s), false);
});

test("a false arithmetic chain is a re-read, and lands unsure rather than unreadable", () => {
  const s = readSignals({ arithmetic: false, structural: true, recognition: true, plausibility: true });
  assert.equal(needsReread(s), true);
  assert.equal(tierFrom(s), "unsure");
});

test("all four true is the only route to confident", () => {
  assert.equal(tierFrom(readSignals({ arithmetic: true, structural: true, recognition: true, plausibility: true })), "confident");
  for (const k of ["structural", "plausibility"] as const) {
    const s = readSignals({ arithmetic: true, structural: true, recognition: true, plausibility: true, [k]: false });
    assert.equal(tierFrom(s), "unsure", `${k} false must not be confident`);
  }
});

// ── labels ──────────────────────────────────────────────────────────────────

test("the live label spellings collapse to one canonical part", () => {
  // Both of these are in production, on the same paper, for the same question.
  assert.equal(canonicalLabel("2a"), canonicalLabel("2. a)"));
  assert.equal(canonicalLabel("d(i)"), canonicalLabel("d) (i)"));
  assert.equal(canonicalLabel("(d)(ii)"), "d(ii)");
  assert.equal(canonicalLabel("b"), "b");
  assert.equal(canonicalLabel("1"), "1");
  assert.equal(canonicalLabel("(ii)"), "(ii)");
  assert.equal(canonicalLabel("Q1 continued"), null);
  assert.equal(canonicalLabel(null), null);
});

test("two regions claiming the same part fails the set", () => {
  const r = checkLabels(["c", "d(i)", "d(ii)", "e", "2a", "a", "2. a)"]);
  assert.equal(r.ok, false);
  const dup = r.problems.find((p) => p.kind === "duplicate");
  assert.ok(dup, "the 2a / 2. a) collision must be found");
  assert.deepEqual(dup!.at, [4, 6]);
});

test("a clean label set passes", () => {
  assert.equal(checkLabels(["c", "d(i)", "d(ii)", "e", "2a", "a", "b"]).ok, true);
});

test("one unreadable label is reported but does not fail the paper", () => {
  const r = checkLabels(["a", "b", "Q1 continued"]);
  assert.equal(r.ok, true, "nineteen good questions are not thrown away for one");
  assert.equal(r.problems.filter((p) => p.kind === "unreadable").length, 1);
});

// ── adjudication blocks the commit ──────────────────────────────────────────

/**
 * Verbatim from `confidence_signals.adjudication.evidence` on a committed row
 * carrying marks_awarded = 3.00. The system diagnosed its own labelling failure
 * and committed anyway.
 */
const LIVE_EVIDENCE =
  "The pipeline read 3/3 for question c, but looking at the first page, the mark for 1a is 3 " +
  "and 1b is 1. The pipeline seems to have misidentified the question labels or order. " +
  "The total on the paper is 21/40.";

test("the adjudication that shipped anyway now blocks the commit", () => {
  const r = adjudicationBlocksCommit({ evidence: LIVE_EVIDENCE });
  assert.equal(r.blocked, true);
  assert.match(r.reason!, /label/);
});

test("explicit verdicts block", () => {
  for (const verdict of ["reject", "needs_review", "blocked"]) {
    assert.equal(adjudicationBlocksCommit({ verdict }).blocked, true, verdict);
  }
  assert.equal(adjudicationBlocksCommit({ labels_misidentified: true }).blocked, true);
  assert.equal(adjudicationBlocksCommit({ structural_problem: true }).blocked, true);
});

test("an ordinary adjudication does not block", () => {
  assert.equal(adjudicationBlocksCommit({ verdict: "accept", evidence: "Totals agree with the front page." }).blocked, false);
  assert.equal(adjudicationBlocksCommit(null).blocked, false);
  assert.equal(adjudicationBlocksCommit({}).blocked, false);
  // Deliberately not a sentiment check: prose that merely sounds doubtful is
  // not a structural finding, and treating it as one would be a model judging a
  // model again.
  assert.equal(adjudicationBlocksCommit({ evidence: "The handwriting is difficult in places." }).blocked, false);
});
