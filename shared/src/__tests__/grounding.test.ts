import { test } from "node:test";
import assert from "node:assert/strict";
import { normalisePartKey, partReferences, resolveDependencies } from "../question_parts.js";
import { gateModelAnswer, subjectTerms } from "../grounding.js";
import { validate as validateContent } from "../prompts/content.v1.js";

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

// ── More ways a Cambridge stem refers backwards ─────────────────────────────

test("resolveDependencies: several referenced parts all come back", () => {
  const a = { label: "a", key: "a", questionText: "a) Convert 12 to binary.", studentAnswer: "1100", marksAwarded: 2, marksAvailable: 2, orderIndex: 0 };
  const b = { label: "b", key: "b", questionText: "b) Normalise it.", studentAnswer: "0.11 x 2^4", marksAwarded: 1, marksAvailable: 2, orderIndex: 1 };
  const r = resolveDependencies("Using your answers to parts (a) and (b), state the exponent.", 2, [a, b] as any);
  assert.deepEqual(r.resolved.map((p) => p.label), ["a", "b"]);
  assert.deepEqual(r.unresolved, []);
});

test("resolveDependencies: one referenced part present and one missing is not 'grounded'", () => {
  // The dangerous middle case. Half the setup is a worse position than none of
  // it, because the prompt looks populated.
  const a = { label: "a", key: "a", questionText: "a) Convert 12 to binary.", studentAnswer: "1100", marksAwarded: 2, marksAvailable: 2, orderIndex: 0 };
  const r = resolveDependencies("Using your answers to parts (a) and (b), state the exponent.", 2, [a] as any);
  assert.deepEqual(r.resolved.map((p) => p.label), ["a"]);
  assert.deepEqual(r.unresolved, ["b"]);
});

test("resolveDependencies: a part on an earlier page resolves the same way", () => {
  // order_index is run-wide, not page-wide, so a reference across a page break
  // needs nothing special — this pins that, because it is the case a
  // page-scoped implementation would silently get wrong.
  const c = { label: "c", key: "c", questionText: "c) Give the mantissa.", studentAnswer: "0.1001", marksAwarded: 1, marksAvailable: 1, orderIndex: 1 };
  const r = resolveDependencies("Justify your answer to part (c).", 6, [c] as any);
  assert.deepEqual(r.resolved.map((p) => p.label), ["c"]);
  assert.deepEqual(r.unresolved, []);
});

test("resolveDependencies: an ambiguous or malformed reference is unresolved, never guessed", () => {
  const d = { label: "d(i)", key: "d(i)", questionText: "d) (i) State whether it is normalised.", studentAnswer: "No", marksAwarded: 1, marksAvailable: 1, orderIndex: 1 };
  // Points at a part that is not on this paper at all.
  const missing = resolveDependencies("Justify your answer given in part (z)(iv).", 5, [d] as any);
  assert.deepEqual(missing.resolved, []);
  assert.equal(missing.unresolved.length, 1);
  // Points at nothing recognisable: not dependent, so nothing is withheld on it.
  const vague = resolveDependencies("Justify your previous answer.", 5, [d] as any);
  assert.equal(vague.dependent, false);
  assert.deepEqual(vague.unresolved, []);
});

/**
 * Replaying the shipped logic over every transcribed question in production
 * (42 questions across 6 runs, 2026-09-06) found 12 dependent questions and
 * resolved all 12, with no false dependents. Two distinct stems account for
 * them, and this is the second one — d(i) itself depends on part (c), which is
 * easy to miss because the eye goes to the "justify" in d(ii).
 */
test("resolveDependencies: d(i) depends on (c), the other real dependent stem", () => {
  const c = {
    label: "c", key: "c",
    questionText: "c) Calculate the denary value of the following binary floating number. Show your working.",
    studentAnswer: "Working... 1/8 + 1/16 = 3/16 | 2^5\n3/16 * 32 = 6",
    marksAwarded: 3, marksAvailable: 3, orderIndex: 0,
  };
  const r = resolveDependencies(
    "d) (i) State whether the floating-point number given in part (c) is normalised or not normalised.",
    1, [c] as any,
  );
  assert.equal(r.dependent, true);
  assert.deepEqual(r.resolved.map((p) => p.label), ["c"]);
  assert.deepEqual(r.unresolved, []);
});

/**
 * The same replay found no false dependents across those 42 questions. These
 * are the real non-dependent stems it had to leave alone — the regression that
 * would matter most, since a false dependency withholds a working that was fine.
 */
test("partReferences: the real non-dependent stems on this paper stay non-dependent", () => {
  for (const stem of [
    "a) Calculate the normalised floating-point representation of +4.5 in this system. Show your working.",
    "b) Calculate the normalised floating-point representation of -4.5 in this system. Show your working.",
    "c) Calculate the denary value of the following binary floating number. Show your working.",
    "e) The system changes so that it now allocates eight bits to both the mantissa and the exponent. Explain two effects this has on the numbers that can be represented.",
    "2. a) Describe the purpose of a user-defined data type.",
  ]) {
    assert.deepEqual(partReferences(stem), [], `false reference found in: ${stem}`);
  }
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
  assert.equal(verdict.status, "heuristic_off_topic");
  assert.equal(verdict.source, null);
});

test("gate: the corrected working this question should have had is kept, and attributed", () => {
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
  assert.equal(verdict.status, "complete");
  // Never verified_scheme: Cambridge scheme content is not ours to reproduce,
  // so what we show is our own method and says so.
  assert.equal(verdict.source, "axon_method");
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
  assert.equal(verdict.status, "missing_dependency");
});

test("gate: a missing dependency outranks an absent answer", () => {
  // Both are true; only one tells you a page is missing from the scan.
  const verdict = gateModelAnswer({
    modelAnswer: null,
    questionText: D_II_QUESTION,
    studentAnswer: null,
    contextText: [],
    unresolvedDependencies: ["d(i)"],
  });
  assert.equal(verdict.status, "missing_dependency");
});

test("gate: an unreadable stem is its own status", () => {
  const verdict = gateModelAnswer({
    modelAnswer: "Some working about mantissas.",
    questionText: null,
    studentAnswer: "0.1001",
    contextText: [],
    unresolvedDependencies: [],
  });
  assert.equal(verdict.modelAnswer, null);
  assert.equal(verdict.status, "missing_question_text");
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
    assert.equal(verdict.status, "complete", "an honest null is not a failed grounding");
    assert.equal(verdict.source, null);
  }
});

test("gate: too little transcribed to judge topicality does not withhold on it", () => {
  const verdict = gateModelAnswer({
    modelAnswer: "Some working about mantissas and exponents.",
    questionText: "(ii) Justify.",
    studentAnswer: null,
    contextText: [],
    unresolvedDependencies: [],
  });
  assert.equal(verdict.status, "complete");
  assert.ok(verdict.modelAnswer);
});

test("gate: a source is present exactly when an answer is", () => {
  // The invariant the database CHECK enforces, asserted here too so the two
  // cannot drift: no answer without a source, no source without an answer.
  const cases: Array<Parameters<typeof gateModelAnswer>[0]> = [
    { modelAnswer: FABRICATION, questionText: D_II_QUESTION, studentAnswer: D_II_ANSWER, contextText: [D_I.questionText, D_I.studentAnswer], unresolvedDependencies: [] },
    { modelAnswer: "x", questionText: D_II_QUESTION, studentAnswer: null, contextText: [], unresolvedDependencies: ["d(i)"] },
    { modelAnswer: null, questionText: D_II_QUESTION, studentAnswer: null, contextText: [], unresolvedDependencies: [] },
    { modelAnswer: "The mantissa begins 0.0 so it is not normalised, floating point.", questionText: D_II_QUESTION, studentAnswer: D_II_ANSWER, contextText: [D_I.questionText, D_I.studentAnswer], unresolvedDependencies: [] },
  ];
  for (const c of cases) {
    const v = gateModelAnswer(c);
    assert.equal(v.source !== null, v.modelAnswer !== null, "source and answer must agree");
    if (v.modelAnswer !== null) assert.equal(v.status, "complete", "an answer requires complete grounding");
  }
});

test("gate: a non-dependent question is never withheld for dependency", () => {
  const v = gateModelAnswer({
    modelAnswer: "4.5 is 100.1 in binary, normalised to 0.1001 x 2^3.",
    questionText: "b) Calculate the normalised floating-point representation of -4.5.",
    studentAnswer: "100.1 binary, 0.1001 x 2^3",
    contextText: [],
    unresolvedDependencies: [],
  });
  assert.equal(v.status, "complete");
  assert.ok(v.modelAnswer);
});

// ── Whole marks, at the model boundary ─────────────────────────────────────
//
// The database CHECK added in 20260906091000_whole_marks_only refuses a
// fractional mark outright. That is the guarantee; this is what keeps one
// misread digit from failing an entire run instead of the one question it
// concerns.

test("content: a fractional mark is dropped to null and sent to review", () => {
  const v = validateContent({
    unreadable: false,
    marks_awarded: { value: 0.5, box: { x: 1, y: 1, w: 1, h: 1, page_index: 0 } },
    marks_available: { value: 3, box: { x: 1, y: 1, w: 1, h: 1, page_index: 0 } },
    recognition_confidence: "high",
  });
  assert.equal(v.marks_awarded?.value, null, "0.5 is not a mark a CAIE teacher wrote");
  assert.equal(v.marks_available?.value, 3, "a sound allocation is left alone");
  assert.equal(v.recognition_confidence, "low", "the question must reach a person");
});

test("content: whole marks pass through untouched, at full confidence", () => {
  const v = validateContent({
    unreadable: false,
    marks_awarded: { value: 1, box: { x: 1, y: 1, w: 1, h: 1, page_index: 0 } },
    marks_available: { value: 3, box: { x: 1, y: 1, w: 1, h: 1, page_index: 0 } },
    recognition_confidence: "high",
  });
  assert.equal(v.marks_awarded?.value, 1);
  assert.equal(v.recognition_confidence, "high");
});

test("content: a fractional allocation is dropped too", () => {
  const v = validateContent({
    unreadable: false,
    marks_awarded: { value: 1, box: { x: 1, y: 1, w: 1, h: 1, page_index: 0 } },
    marks_available: { value: 2.5, box: { x: 1, y: 1, w: 1, h: 1, page_index: 0 } },
    recognition_confidence: "high",
  });
  assert.equal(v.marks_available?.value, null);
  assert.equal(v.recognition_confidence, "low");
});
