import { test } from "node:test";
import assert from "node:assert/strict";
import {
  candidateIsResolvable,
  identityMatchesCandidate,
  normaliseQuestionLabel,
  type AssessmentCandidate,
} from "../assessment.js";

const candidate: AssessmentCandidate = {
  subject_code: "9702",
  level: null,
  exam_year: 2025,
  session: "May/June",
  paper_code: "42",
  component_code: "9702/42",
  variant: "2",
  zone: null,
  assessment_route: null,
  confidence: "high",
};

test("assessment candidates require visible exact discriminators", () => {
  assert.equal(candidateIsResolvable(candidate), true);
  assert.equal(candidateIsResolvable({ ...candidate, confidence: "low" }), false);
  assert.equal(candidateIsResolvable({ ...candidate, subject_code: null }), false);
  assert.equal(candidateIsResolvable({ ...candidate, exam_year: null }), false);
  assert.equal(candidateIsResolvable({ ...candidate, paper_code: null, component_code: null }), false);
});

test("identity matching compares every stored discriminator, not semantic similarity", () => {
  const identity = {
    level: null,
    exam_year: 2025,
    session: "May June",
    paper_code: "42",
    component_code: "9702-42",
    variant: "2",
    zone: null,
    assessment_route: null,
  };
  assert.equal(identityMatchesCandidate(identity, candidate, null), true);
  assert.equal(identityMatchesCandidate({ ...identity, variant: "3" }, candidate, null), false);
  assert.equal(identityMatchesCandidate({ ...identity, session: "Oct/Nov" }, candidate, null), false);
});

test("student-selected IB level can complete an otherwise exact candidate", () => {
  const ib = {
    level: "HL",
    exam_year: 2026,
    session: "May",
    paper_code: "P1",
    component_code: null,
    variant: null,
    zone: null,
    assessment_route: null,
  };
  const transcribed: AssessmentCandidate = {
    subject_code: "100452",
    level: null,
    exam_year: 2026,
    session: "May",
    paper_code: "P1",
    component_code: null,
    variant: null,
    zone: null,
    assessment_route: null,
    confidence: "high",
  };
  assert.equal(identityMatchesCandidate(ib, transcribed, "HL"), true);
  assert.equal(identityMatchesCandidate(ib, transcribed, "SL"), false);
});

test("question labels are normalized only typographically", () => {
  assert.equal(normaliseQuestionLabel(" 3 (b) (ii). "), "3(B)(II)");
  assert.equal(normaliseQuestionLabel(null), null);
  assert.notEqual(normaliseQuestionLabel("3(b)(ii)"), normaliseQuestionLabel("3(b)(iii)"));
});
