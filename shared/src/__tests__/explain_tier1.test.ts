import { test } from "node:test";
import assert from "node:assert/strict";
import { validate, SCHEMA } from "../prompts/explain_tier1.v1.js";

// The regression this file exists for: EXPLANATION_SCHEMA asks the model for
// `explanation`, and validate() read `v.body`. Nothing threw, nothing logged,
// and every explanation ever generated was discarded on the way to the column.
// The two names are asserted against each other here so they cannot drift apart
// again in silence.
test("validate: reads the prose out of the field the schema actually asks for", () => {
  const v = validate({
    can_explain: true,
    cause: "conceptual_gap",
    marks_lost: 2,
    explanation: "The two's complement inversion was skipped before the exponent was appended.",
    do_this_next: "Write the inverted mantissa on its own line before you append the exponent.",
    concepts: ["Floating-point representation"],
  });
  assert.equal(v.body, "The two's complement inversion was skipped before the exponent was appended.");
});

test("validate: the schema's property name and the validator agree", () => {
  const props = Object.keys((SCHEMA.schema as any).properties);
  assert.ok(props.includes("explanation"), "schema should ask for `explanation`");
  const v = validate({ cause: "incomplete", marks_lost: 1, explanation: "Half the method was missing.", concepts: [] });
  assert.equal(v.body, "Half the method was missing.", "validate must read the field the schema names");
});

test("validate: still accepts `body` as a fallback name", () => {
  assert.equal(validate({ cause: "incomplete", marks_lost: 1, body: "Only one reason given.", concepts: [] }).body, "Only one reason given.");
});

test("validate: blank and non-string prose is null, never an empty bubble", () => {
  assert.equal(validate({ cause: "incomplete", marks_lost: 1, explanation: "   ", concepts: [] }).body, null);
  assert.equal(validate({ cause: "incomplete", marks_lost: 1, explanation: 42, concepts: [] }).body, null);
  assert.equal(validate({ cause: "incomplete", marks_lost: 1, concepts: [] }).body, null);
});

// Hard rule: the cause enum is fixed. An eighth cause is dropped, not stored.
test("validate: an invented cause is refused, and takes marks_lost with it", () => {
  const v = validate({ cause: "vibes", marks_lost: 3, explanation: "x", concepts: [] });
  assert.equal(v.cause, null);
  assert.equal(v.marks_lost, null);
});

test("validate: concepts are capped at six and filtered to strings", () => {
  const v = validate({
    cause: "keyword_miss", marks_lost: 1, explanation: "x",
    concepts: ["a", "b", "c", "d", "e", "f", "g", 7, null],
  });
  assert.deepEqual(v.concepts, ["a", "b", "c", "d", "e", "f"]);
});

test("validate: nothing returned is an error, not a silent empty result", () => {
  assert.throws(() => validate(null));
  assert.throws(() => validate("nope"));
});

// ── Phase 2: command word, model answer, decomposed loss reasons ────────────

test("validate: a command word is canonicalised, not passed through as typed", () => {
  assert.equal(validate({ cause: "incomplete", marks_lost: 1, command_word: "explain", concepts: [] }).command_word, "Explain");
  assert.equal(validate({ cause: "incomplete", marks_lost: 1, command_word: "SHOW THAT", concepts: [] }).command_word, "Show that");
  assert.equal(validate({ cause: "incomplete", marks_lost: 1, command_word: " Justify, ", concepts: [] }).command_word, "Justify");
});

test("validate: a word outside the closed list renders nothing, and takes its note with it", () => {
  const v = validate({
    cause: "incomplete", marks_lost: 1, concepts: [],
    command_word: "Ponder", command_word_note: "Ponder wants you to muse.",
  });
  assert.equal(v.command_word, null);
  assert.equal(v.command_word_note, null, "a note without its word is a caption on a missing picture");
});

test("validate: a phrase that merely contains a command word is not one", () => {
  assert.equal(validate({ cause: "incomplete", marks_lost: 1, command_word: "state the explain", concepts: [] }).command_word, null);
});

test("validate: loss_reasons keep marks, cause and the anchoring note", () => {
  const v = validate({
    cause: "conceptual_gap", marks_lost: 2, concepts: [],
    loss_reasons: [
      { marks: 1, cause: "procedural_slip", note: "between your line 2 and line 3" },
      { marks: 1, cause: "conceptual_gap", note: "the exponent was never converted" },
    ],
  });
  assert.equal(v.loss_reasons.length, 2);
  assert.equal(v.loss_reasons[0].cause, "procedural_slip");
  assert.equal(v.loss_reasons[0].note, "between your line 2 and line 3");
});

// Hard rule 2: M/A/B/C is marking-scheme vocabulary, and Tier 1 has no scheme.
// The model is not asked for it and must not be able to smuggle it in.
test("validate: mark_type is null on tier 1 even when the model volunteers one", () => {
  const v = validate({
    cause: "conceptual_gap", marks_lost: 1, concepts: [],
    loss_reasons: [{ marks: 1, cause: "conceptual_gap", note: "x", mark_type: "M" }],
  });
  assert.equal(v.loss_reasons[0].mark_type, null);
});

test("validate: a loss reason with an invented cause or no marks is dropped, not stored", () => {
  const v = validate({
    cause: "incomplete", marks_lost: 2, concepts: [],
    loss_reasons: [
      { marks: 1, cause: "vibes", note: "x" },
      { marks: 0, cause: "incomplete", note: "y" },
      { marks: 1, cause: "incomplete", note: "z" },
    ],
  });
  assert.deepEqual(v.loss_reasons.map((r) => r.note), ["z"]);
});

test("validate: loss_reasons absent or malformed is an empty array, never a throw", () => {
  assert.deepEqual(validate({ cause: "incomplete", marks_lost: 1, concepts: [] }).loss_reasons, []);
  assert.deepEqual(validate({ cause: "incomplete", marks_lost: 1, concepts: [], loss_reasons: "no" }).loss_reasons, []);
  assert.deepEqual(validate({ cause: "incomplete", marks_lost: 1, concepts: [], loss_reasons: [null, 3] }).loss_reasons, []);
});

test("validate: model_answer is prose or null, never an empty bubble", () => {
  assert.equal(validate({ cause: "incomplete", marks_lost: 1, concepts: [], model_answer: "  " }).model_answer, null);
  assert.equal(
    validate({ cause: "incomplete", marks_lost: 1, concepts: [], model_answer: "0.1001 × 2^3\n1.01110000" }).model_answer,
    "0.1001 × 2^3\n1.01110000",
  );
});
