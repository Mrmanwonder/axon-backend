import { test } from "node:test";
import assert from "node:assert/strict";
import { validate, SCHEMA } from "../prompts/explain_tier1.v1.js";
import { COMMAND_WORDS, UNIVERSAL_MEANING, canonicalCommandWord, universalMeaning } from "../command_words.js";

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

// ── the sourced command-word list ──────────────────────────────────────────
// Generated from Cambridge's own command-word guidance. These assert the
// properties the generator has to preserve, not the whole table.

test("command words: the 22 universal words Cambridge publishes are all present", () => {
  for (const w of ["Analyse", "Assess", "Calculate", "Comment", "Compare", "Consider",
                   "Contrast", "Define", "Describe", "Develop", "Discuss", "Evaluate",
                   "Explain", "Give", "Identify", "Justify", "Outline", "Predict",
                   "Sketch", "State", "Suggest", "Summarise"]) {
    assert.ok(canonicalCommandWord(w), `${w} should be a recognised command word`);
    assert.ok(UNIVERSAL_MEANING[w as never], `${w} should carry Cambridge's own definition`);
  }
  assert.equal(Object.keys(UNIVERSAL_MEANING).length, 22);
});

test("command words: subject-specific words are recognised but carry no definition", () => {
  // Cambridge is explicit that a command word can mean something subject-specific,
  // so one subject's reading must never be shown to a student sitting another.
  for (const w of ["Show that", "Trace", "Advise", "Annotate", "Hence", "Prove"]) {
    assert.ok(canonicalCommandWord(w), `${w} should be recognised`);
    assert.equal(universalMeaning(canonicalCommandWord(w)), null, `${w} must not carry a universal definition`);
  }
});

test("command words: 'Mark allocation' is not a command word", () => {
  // It is a row in the source sheet, but it is guidance about marks, not a word
  // a question is built around. Tagging a question with it would be nonsense.
  assert.equal(canonicalCommandWord("Mark allocation"), null);
});

test("command words: the list is unique and free of stray whitespace", () => {
  assert.equal(new Set(COMMAND_WORDS).size, COMMAND_WORDS.length);
  for (const w of COMMAND_WORDS) assert.equal(w, w.trim());
});

test("validate: Cambridge's own definition beats the model's paraphrase", () => {
  const v = validate({
    cause: "incomplete", marks_lost: 1, concepts: [],
    command_word: "explain", command_word_note: "Explain means to write a lot.",
  });
  assert.equal(v.command_word, "Explain");
  assert.equal(v.command_word_note, UNIVERSAL_MEANING["Explain"]);
});

test("validate: the model's note is kept for a subject-specific word", () => {
  const v = validate({
    cause: "incomplete", marks_lost: 1, concepts: [],
    command_word: "Show that", command_word_note: "Every step to the given value must appear.",
  });
  assert.equal(v.command_word, "Show that");
  assert.equal(v.command_word_note, "Every step to the given value must appear.");
});

test("validate: a subject-specific word with no note renders the word alone", () => {
  const v = validate({ cause: "incomplete", marks_lost: 1, concepts: [], command_word: "Trace" });
  assert.equal(v.command_word, "Trace");
  assert.equal(v.command_word_note, null);
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

// ── error_type: Axon's own category, no scheme required ────────────────────

test("validate: error_type is carried through for each of the five values", () => {
  for (const t of ["method", "final_answer", "omitted_step", "presentation", "other"]) {
    const v = validate({
      cause: "incomplete", marks_lost: 1, concepts: [],
      loss_reasons: [{ marks: 1, cause: "incomplete", note: "x", error_type: t }],
    });
    assert.equal(v.loss_reasons[0].error_type, t);
  }
});

test("validate: an unknown or missing error_type becomes 'other', and the reason survives", () => {
  // The diagnosis is still real and still has marks against it — dropping the
  // whole reason over an unrecognised label would lose more than it protects.
  for (const t of [undefined, null, "M1", "accuracy", 7, {}]) {
    const v = validate({
      cause: "incomplete", marks_lost: 1, concepts: [],
      loss_reasons: [{ marks: 1, cause: "incomplete", note: "kept", error_type: t }],
    });
    assert.equal(v.loss_reasons.length, 1);
    assert.equal(v.loss_reasons[0].error_type, "other");
    assert.equal(v.loss_reasons[0].note, "kept");
  }
});

// Acceptance, addendum §"Do not do": the gate is the point. Tier 1 has no
// scheme to ground a code in, and Cambridge's notation is not ours to reproduce
// — Cambridge and Pearson refused third-party reproduction rights.
test("validate: mark_type is null however the model dresses it up", () => {
  for (const m of ["M", "A", "B", "C", "M1", "other", 1, true, "  M  "]) {
    const v = validate({
      cause: "incomplete", marks_lost: 2, concepts: [],
      loss_reasons: [{ marks: 1, cause: "incomplete", note: "x", error_type: "method", mark_type: m }],
    });
    assert.equal(v.loss_reasons[0].mark_type, null, `mark_type ${JSON.stringify(m)} must not survive tier 1`);
  }
});

test("the tier 1 prompt never asks for mark_type, and never mentions a code as available", () => {
  // Enforced by construction rather than convention: if the schema does not ask
  // for it, there is no field for a model to fill.
  const reason = (SCHEMA.schema as any).properties.loss_reasons.items;
  assert.ok(!("mark_type" in reason.properties), "loss_reasons items must not offer mark_type");
  assert.equal(reason.additionalProperties, false, "a model must not be able to add it either");
  assert.ok(reason.required.includes("error_type"), "error_type is what the model is asked for instead");
});

test("error_type values carry no single-letter or mark-scheme-shaped labels", () => {
  const values = (SCHEMA.schema as any).properties.loss_reasons.items.properties.error_type.enum;
  for (const v of values) {
    assert.ok(v.length > 1, `${v} must not be a single letter`);
    assert.ok(!/^[MABC]\d*$/i.test(v), `${v} must not look like mark-scheme notation`);
  }
});
