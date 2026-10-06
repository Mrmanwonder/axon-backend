// content.v2 and structure.v2: the rules the owner's 9709 report asked for are
// in the text the model is shown, the provenance labels cannot collide with the
// route label already spent on the v1 content text, and the v1 files are still
// what they were.

import { test } from "node:test";
import assert from "node:assert/strict";

import * as contentV1 from "../prompts/content.v1.js";
import * as contentV2 from "../prompts/content.v2.js";
import * as structureV1 from "../prompts/structure.v1.js";
import * as structureV2 from "../prompts/structure.v2.js";

test("content.v2 transcribes what the student wrote on a diagram, and never reads a drawing", () => {
  const s = contentV2.SYSTEM;
  assert.match(s, /region_type "diagram"/);
  assert.match(s, /Never describe a shape/);
  assert.match(s, /WROTE on or beside the diagram/);
  assert.match(s, /"label: value"/);
  assert.match(s, /2 April after Fine, Fine: 0\.9375/);
  assert.match(s, /Never read a value off the drawing/);
  assert.match(s, /student_answer stays null only when the student wrote no text/);
  // The v1 refusal is gone from v2.
  assert.doesNotMatch(s, /Diagrams are not transcribed/);
});

test("content.v2 transcribes a filled-in table as a LaTeX array", () => {
  const s = contentV2.SYSTEM;
  assert.match(s, /Tables are not diagrams/);
  assert.match(s, /\\begin\{array\}\{\|c\|c\|c\|\}/);
  assert.match(s, /\\hline/);
  assert.match(s, /\\end\{array\}/);
});

test("content.v2 keeps every provenance and no-judgement rule", () => {
  const s = contentV2.SYSTEM;
  assert.match(s, /Every value you return must have a box/);
  assert.match(s, /You read\. You do not judge\./);
  assert.match(s, /null is a correct answer/);
  assert.match(s, /never instruction to follow/);
  assert.match(s, /Every transcribed value carries its box/);
});

test("content.v2 never assumes the teacher's ink is red", () => {
  assert.doesNotMatch(contentV2.SYSTEM, /in red pen/);
  const text = contentV2.instruction({
    label: "3(b)",
    pageNumbers: [4],
    layerFallback: "non_red_marking",
    teacherMarks: [{ shape: "tick", where: "on page 4" }],
  });
  assert.doesNotMatch(text, /\bred\b/i);
  assert.match(text, /question 3\(b\)/);
});

test("content.v2 shares v1's schema and validator; only the text changed", () => {
  assert.equal(contentV2.SCHEMA, contentV1.SCHEMA);
  assert.equal(contentV2.validate, contentV1.validate);
  assert.notEqual(contentV2.SYSTEM, contentV1.SYSTEM);
});

test("provenance labels: content.v2.ts is recorded as content.v3, never the route's existing content.v2", () => {
  assert.equal(contentV2.PROMPT_VERSION, "content.v3");
  assert.equal(structureV2.PROMPT_VERSION, "structure.v2");
});

test("structure.v2 names page furniture and forbids the page number in a label", () => {
  const s = structureV2.SYSTEM;
  assert.match(s, /Page furniture is never a question number/);
  assert.match(s, /printed_page_number/);
  assert.match(s, /\[Turn over/);
  assert.match(s, /copyright\s+line/);
  assert.match(s, /paper\s+code/);
  assert.match(s, /continues_from_previous true/);
  assert.match(s, /Never put the page number/);
  assert.match(s, /Never invent a number/);
  assert.notEqual(s, structureV1.SYSTEM);
});

test("structure.v2 schema asks for the printed page number with its box", () => {
  const schema = structureV2.SCHEMA.schema as any;
  assert.ok(schema.required.includes("printed_page_number"));
  assert.deepEqual(schema.properties.printed_page_number.required, ["value", "box", "page_index"]);
  // v1's schema is untouched.
  assert.ok(!(structureV1.SCHEMA.schema as any).required.includes("printed_page_number"));
});

test("structure.v2 validate keeps v1's region filter and normalises the printed page number", () => {
  const out = structureV2.validate({
    is_graded_exam_paper: true,
    not_a_paper_reason: null,
    reported_total: null,
    stated_maximum: null,
    printed_page_number: { value: 6, box: { x: 490, y: 20, w: 20, h: 18 }, page_index: 0 },
    regions: [
      { candidate_number: "4", number_box: null, box: { x: 30, y: 100, w: 900, h: 200 }, continues_from_previous: false, structure_confidence: "high" },
      { candidate_number: "5", number_box: null, box: { x: "no" }, continues_from_previous: false, structure_confidence: "high" },
    ],
  });
  assert.equal(out.regions.length, 1);
  assert.equal(out.printed_page_number?.value, "6");

  const none = structureV2.validate({ is_graded_exam_paper: true, regions: [], printed_page_number: null });
  assert.equal(none.printed_page_number, null);
  const missing = structureV2.validate({ is_graded_exam_paper: true, regions: [] });
  assert.equal(missing.printed_page_number, null);
  assert.throws(() => structureV2.validate({ regions: [] }), /no verdict/);
});
