import { test } from "node:test";
import assert from "node:assert/strict";
import { normalisePartKey, partReferences, resolveDependencies } from "../question_parts.js";
import { gateModelAnswer, subjectTerms } from "../grounding.js";

/**
 * The paper this file exists for.
 *
 * Run 291f5ae1 on the live project, 2026-09-05. Every string below is copied
 * from `question_region` and `region_explanation` as they actually stood, and
 * the two questions really were at adjacent order_index in the same run.
 */
const D_I = {
  label: "d(i)",
  key: normalisePartKey("d(i)"),
  questionText: "d) (i) State whether the floating-point number given in part (c) is normalised or not normalised.",
  studentAnswer: "Not Normalized",
  marksAwarded: 1,
  marksAvailable: 1,
  orderIndex: 1,
};
const D_II_QUESTION = "(ii) Justify your answer given in part (d)(i).";
const D_II_ANSWER = "Can be written in just 3 bits for each 0.1 for +ve 1.0 for -ve";

/** What the model actually returned as "the corrected working". */
const FABRICATION =
  "The answer in (d)(i) is correct because the signal-to-noise ratio remains above the " +
  "threshold required for accurate signal reconstruction, ensuring the bit error rate " +
  "stays within acceptable limits.";

test("normalisePartKey: the shapes a label actually arrives in", () => {
  assert.equal(normalisePartKey("d(i)"), "d(i)");
  assert.equal(normalisePartKey("d) (i)"), "d(i)");
  assert.equal(normalisePartKey("(d)(ii)"), "d(ii)");
  assert.equal(normalisePartKey("b"), "b");
  assert.equal(normalisePartKey("2a"), "a");
  assert.equal(normalisePartKey("2. a)"), "a");
  assert.equal(normalisePartKey("(ii)"), "(ii)");
  assert.equal(normalisePartKey(null), null);
  assert.equal(normalisePartKey("Q1 continued"), null);
});

test("partReferences: finds the part the production question pointed at", () => {
  assert.deepEqual(partReferences(D_II_QUESTION), ["d(i)"]);
});

test("partReferences: the other ways a Cambridge stem refers backwards", () => {
  assert.deepEqual(partReferences("Using the value found in part (b), calculate the mean."), ["b"]);
  assert.deepEqual(partReferences("Explain your answers to parts (a) and (b).").sort(), ["a", "b"]);
  assert.deepEqual(partReferences("State whether the number in part (c) is normalised."), ["c"]);
});

test("partReferences: prose containing the word 'part' is not a reference", () => {
  assert.deepEqual(partReferences("Describe one part of the fetch-execute cycle."), []);
  assert.deepEqual(partReferences("Name a part a compiler plays in translation."), []);
  assert.deepEqual(partReferences("Calculate the normalised representation of -4.5."), []);
});

test("resolveDependencies: d(ii) resolves to the d(i) sitting one row away", () => {
  const r = resolveDependencies(D_II_QUESTION, 2, [D_I] as any);
  assert.equal(r.dependent, true);
  assert.deepEqual(r.unresolved, []);
  assert.equal(r.resolved.length, 1);
  assert.equal(r.resolved[0].label, "d(i)");
  assert.equal(r.resolved[0].studentAnswer, "Not Normalized");
});

test("resolveDependencies: a part that is not in the run is reported, not guessed at", () => {
  const r = resolveDependencies(D_II_QUESTION, 2, [] as any);
  assert.equal(r.dependent, true);
  assert.deepEqual(r.unresolved, ["d(i)"]);
  assert.deepEqual(r.resolved, []);
});

test("resolveDependencies: only looks backwards", () => {
  // The same d(i) row, but sitting *after* this question. A forward match would
  // mean the labels are wrong, and feeding a later question to cover it up is
  // how one bad read becomes a confident explanation of the wrong thing.
  const r = resolveDependencies(D_II_QUESTION, 0, [{ ...D_I, orderIndex: 5 }] as any);
  assert.deepEqual(r.unresolved, ["d(i)"]);
});

test("resolveDependencies: a question that stands alone depends on nothing", () => {
  const r = resolveDependencies("Calculate the normalised representation of -4.5.", 3, [D_I] as any);
  assert.equal(r.dependent, false);
  assert.deepEqual(r.resolved, []);
  assert.deepEqual(r.unresolved, []);
});

test("resolveDependencies: 'part (d)' names every part of the d group", () => {
  const dii = { ...D_I, label: "d(ii)", key: "d(ii)", orderIndex: 2 };
  const r = resolveDependencies("Explain your answer to part (d).", 4, [D_I, dii] as any);
  assert.deepEqual(r.resolved.map((p) => p.label).sort(), ["d(i)", "d(ii)"]);
  assert.deepEqual(r.unresolved, []);
});

// ── The gate ───────────────────────────────────────────────────────────────

test("subjectTerms: exam scaffolding is not subject matter", () => {
  const t = subjectTerms("State your answer to the question above.");
  assert.equal(t.size, 0, `expected no subject terms, got ${[...t].join(",")}`);
});

test("subjectTerms: stems so bits and bit meet", () => {
  assert.ok(subjectTerms("three bits").has("bit"));
  assert.ok(subjectTerms("one bit").has("bit"));
});

test("gate: the production fabrication is withheld as off-topic", () => {
  const verdict = gateModelAnswer({
    modelAnswer: FABRICATION,
    questionText: D_II_QUESTION,
    studentAnswer: D_II_ANSWER,
    contextText: [D_I.questionText, D_I.studentAnswer],
    unresolvedDependencies: [],
  });
  assert.equal(verdict.modelAnswer, null, "signal-to-noise ratio is not this question");
  assert.equal(verdict.withheldReason, "off_topic");
});

test("gate: the corrected working this question should have had is kept", () => {
  const real =
    "The mantissa is 0.0101, which begins 0.0. A normalised positive number must begin 0.1, " +
    "so this floating-point number is not normalised.";
  const verdict = gateModelAnswer({
    modelAnswer: real,
    questionText: D_II_QUESTION,
    studentAnswer: D_II_ANSWER,
    contextText: [D_I.questionText, D_I.studentAnswer],
    unresolvedDependencies: [],
  });
  assert.equal(verdict.modelAnswer, real);
  assert.equal(verdict.withheldReason, null);
});

test("gate: an unresolved dependency withholds whatever came back", () => {
  const verdict = gateModelAnswer({
    modelAnswer: "Anything at all, however plausible it reads.",
    questionText: D_II_QUESTION,
    studentAnswer: D_II_ANSWER,
    contextText: [],
    unresolvedDependencies: ["d(i)"],
  });
  assert.equal(verdict.modelAnswer, null);
  assert.equal(verdict.withheldReason, "unresolved_dependency");
});

test("gate: the model declining to write one is not a withholding", () => {
  for (const empty of [null, "", "   "]) {
    const verdict = gateModelAnswer({
      modelAnswer: empty,
      questionText: D_II_QUESTION,
      studentAnswer: D_II_ANSWER,
      contextText: [],
      unresolvedDependencies: [],
    });
    assert.equal(verdict.modelAnswer, null);
    assert.equal(verdict.withheldReason, null, "an honest null carries no reason");
  }
});

test("gate: too little transcribed to judge topicality does not withhold on it", () => {
  // Hard rule 4 says an unreadable page says so. It does not say we invent a
  // verdict about a question we could not read — the dependency gate has
  // already passed, and there is nothing here to test the answer against.
  const verdict = gateModelAnswer({
    modelAnswer: "Some working about mantissas and exponents.",
    questionText: "(ii)",
    studentAnswer: null,
    contextText: [],
    unresolvedDependencies: [],
  });
  assert.equal(verdict.withheldReason, null);
  assert.ok(verdict.modelAnswer);
});
