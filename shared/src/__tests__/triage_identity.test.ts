import { test } from "node:test";
import assert from "node:assert/strict";
import { validate } from "../prompts/triage.v1.js";

test("triage normalizes a printed CBSE sample-paper route without inventing codes", () => {
  const result = validate({
    classification: "graded_exam",
    subject: "Physics",
    marked_page_count: 2,
    ink_colour: "red",
    confidence: "high",
    assessment_identity: {
      subject_code: "042",
      level: null,
      exam_year: 2027,
      session: "2026-27",
      paper_code: null,
      component_code: null,
      variant: null,
      zone: null,
      assessment_route: "Sample Question Paper",
      confidence: "high",
    },
  });

  assert.equal(result.assessment_identity?.assessment_route, "sample_paper");
  assert.equal(result.assessment_identity?.subject_code, "042");
  assert.equal(result.assessment_identity?.exam_year, 2027);
  assert.equal(result.assessment_identity?.session, "2026-27");
});

test("ordinary school-test metadata is not promoted to a known sample route", () => {
  const result = validate({
    classification: "graded_exam",
    subject: "Physics",
    marked_page_count: 1,
    ink_colour: "other",
    confidence: "low",
    assessment_identity: {
      subject_code: null,
      level: null,
      exam_year: null,
      session: null,
      paper_code: null,
      component_code: null,
      variant: null,
      zone: null,
      assessment_route: "Unit Test",
      confidence: "low",
    },
  });
  assert.equal(result.assessment_identity?.assessment_route, "Unit Test");
});
