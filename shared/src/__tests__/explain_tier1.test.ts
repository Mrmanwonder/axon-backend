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
