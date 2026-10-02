import { test } from "node:test";
import assert from "node:assert/strict";
import { adjudicationReasons, checkExplain } from "../eval/explain-score.js";
import * as judge from "../eval/judge.v1.js";
import { runExplainEvalCase, type EvalDeps } from "../eval/explain-run.js";
import type { ExplainResult } from "../prompts/explain_tier1.v2.js";

const MARKS = { marksAwarded: 2, marksAvailable: 3 };

// ── the hard-rule check ─────────────────────────────────────────────────────────────────────────

test("phrasings that dispute the teacher's mark are caught, in any of the forbidden forms", () => {
  const bad: Array<[string, string]> = [
    ["You should have got full marks here.", "should_have_got"],
    ["Arguably this answer deserves more credit.", "arguably"],
    ["A stricter reading of the question would give 3.", "stricter_reading"],
    ["The marking looks harsh for this working.", "judges_the_marking"],
    ["I think the teacher made a mistake.", "calls_marking_wrong"],
    ["You could have earned another mark by adding the unit.", "could_have_earned"],
    ["This is worth full marks.", "worth_more"],
    ["That is 3 out of 3 marks for the method.", "states_other_mark_total"],
    ["You should score 3 marks for this.", "claims_a_different_mark"],
  ];
  for (const [text, reason] of bad) assert.ok(adjudicationReasons([text], MARKS).includes(reason), `${reason}: ${text}`);
});

test("ordinary explanation text, including the teacher's real total, is not flagged", () => {
  const fine = [
    "Your answer is right. The mark went for the unit, which the question asked you to include.",
    "The teacher gave 2 out of 3 marks. The missing mark is for stating that the image is real.",
    "Write the formula on its own line before you substitute.",
    "Compare the strength of the forces, then say more energy is needed to overcome them.",
    "Check the lower limit before you evaluate; the value at x = 0 is subtracted.",
  ];
  for (const text of fine) assert.deepEqual(adjudicationReasons([text], MARKS), [], text);
});

test("checkExplain reports the arithmetic the pipeline would silently drop", () => {
  const parsed = {
    can_explain: true, cause: "incomplete", marks_lost: 1, body: "You stopped one step early.", do_this_next: "Write the unit on its own line.",
    concepts: [], command_word: null, command_word_note: null, model_answer: null,
    loss_reasons: [{ mark_type: null, error_type: "omitted_step", marks: 2, cause: "incomplete", note: null }],
  } as unknown as ExplainResult;
  const checks = checkExplain(parsed, MARKS);
  assert.equal(checks.loss_reasons_ok, false, "2 marks accounted for, only 1 lost");
  assert.equal(checks.teacher_mark_contradiction, false);
  assert.equal(checks.can_explain, true);
});

test("a decline is a result, not a contradiction", () => {
  const parsed = { can_explain: false, cause: null, body: null, do_this_next: null, concepts: [], command_word: null, command_word_note: null, model_answer: null, loss_reasons: [] } as unknown as ExplainResult;
  const checks = checkExplain(parsed, MARKS);
  assert.equal(checks.can_explain, false);
  assert.equal(checks.do_this_next_clears_floor, null);
  assert.equal(checks.teacher_mark_contradiction, false);
});

// ── the judge's contract ────────────────────────────────────────────────────────────────────────

test("the judge returns an integer 1-5 and only known flags", () => {
  assert.deepEqual(judge.validate({ faithfulness: 4, flags: ["generic_advice", "generic_advice"] }), { faithfulness: 4, flags: ["generic_advice"] });
  assert.throws(() => judge.validate({ faithfulness: 6, flags: [] }), /1 to 5/);
  assert.throws(() => judge.validate({ faithfulness: 3.5, flags: [] }), /1 to 5/);
  assert.throws(() => judge.validate({ faithfulness: 3, flags: ["made_up"] }), /unknown flags/);
});

test("the judge is told the teacher's mark is a fact and never to comment on it", () => {
  assert.match(judge.SYSTEM, /teacher's mark is a fact/);
  assert.match(judge.SYSTEM, /never comment on it/);
  const text = judge.instruction({
    question: "q", studentAnswer: "a", teacherRemark: null, marksAwarded: 1, marksAvailable: 2,
    explanation: { can_explain: true, cause: "incomplete", body: "b", do_this_next: null, model_answer: null },
  });
  assert.match(text, /Teacher's mark \(a fact\): 1 out of 2/);
  assert.match(text, /Teacher's remark: \(none\)/);
});

// ── the runner ──────────────────────────────────────────────────────────────────────────────────

function fakeDb(input: unknown, source = "synthetic") {
  const writes: Array<Record<string, unknown>> = [];
  const touched: string[] = [];
  const sb = {
    from(table: string) {
      touched.push(table);
      return {
        select: () => ({ eq: () => ({ maybeSingle: async () => ({ data: { id: "c1", source, input }, error: null }) }) }),
        upsert: async (row: Record<string, unknown>) => { writes.push(row); return { error: null }; },
      };
    },
  } as never;
  return { sb, writes, touched };
}

const EXPLANATION = {
  can_explain: true, cause: "incomplete", marks_lost: 1, body: "You have the method. The mark went for stating the image is real.",
  do_this_next: "Write 'real, inverted, magnified' on its own line after the calculation.", concepts: ["lens"], command_word: null,
  command_word_note: null, model_answer: null, loss_reasons: [{ mark_type: null, error_type: "omitted_step", marks: 1, cause: "incomplete", note: null }],
};
const INPUT = {
  label: "17", subject: "Physics", class_level: 12, marks_awarded: 2, marks_available: 3, question_text: "Find the image distance.",
  student_answer: "v = 60 cm", teacher_remark: "Nature of image not stated.", mark_shapes: ["cross"], prior_parts: [],
};
const candidate = { key: "a", model: "gemini-3.8-flash", thinking_level: "medium" as const };
const baseDeps = (judgeFlags: string[] = []): EvalDeps => ({
  callModel: (async (opts: { routeOverride?: { primary_model?: string } }) => opts.routeOverride?.primary_model === "gemini-3.1-pro-preview"
    ? { parsed: { faithfulness: 5, flags: judgeFlags }, latencyMs: 900, promptVersion: "eval_judge.v1" }
    : { parsed: EXPLANATION, latencyMs: 4200, promptVersion: "paper_feedback.v2" }) as never,
});

test("a case is run, scored, judged, and written as exactly one result row", async () => {
  const { sb, writes, touched } = fakeDb(INPUT);
  await runExplainEvalCase({ env: {} as never, sb, message: { eval_run_id: "r1", case_id: "c1", candidate } }, baseDeps());
  assert.equal(writes.length, 1);
  assert.equal(writes[0].status, "done");
  assert.equal(writes[0].schema_valid, true);
  assert.equal(writes[0].faithfulness, 5);
  assert.equal(writes[0].teacher_mark_contradiction, false);
  assert.equal(writes[0].latency_ms, 4200);
  assert.deepEqual([...new Set(touched)].sort(), ["eval_case", "eval_case_result"], "only the eval tables are touched");
});

test("a judge-raised contradiction counts alongside the rule check", async () => {
  const { sb, writes } = fakeDb(INPUT);
  await runExplainEvalCase({ env: {} as never, sb, message: { eval_run_id: "r1", case_id: "c1", candidate } }, baseDeps(["contradicts_teacher_mark"]));
  assert.equal(writes[0].teacher_mark_contradiction, true);
});

test("a control case with nothing lost is skipped without a model call", async () => {
  const { sb, writes } = fakeDb({ ...INPUT, marks_awarded: 3, marks_available: 3 });
  let called = false;
  await runExplainEvalCase({ env: {} as never, sb, message: { eval_run_id: "r1", case_id: "c1", candidate } }, { callModel: (async () => { called = true; }) as never });
  assert.equal(called, false);
  assert.equal(writes[0].status, "skipped");
  assert.equal(writes[0].skipped_reason, "no_marks_lost");
});

test("a production case is not run: it would read a student's stored text", async () => {
  const { sb, writes } = fakeDb(null, "production");
  let called = false;
  await runExplainEvalCase({ env: {} as never, sb, message: { eval_run_id: "r1", case_id: "c1", candidate } }, { callModel: (async () => { called = true; }) as never });
  assert.equal(called, false);
  assert.equal(writes[0].skipped_reason, "production_cases_not_enabled");
});

test("a model failure is recorded with its code and no invented score", async () => {
  const { sb, writes } = fakeDb(INPUT);
  const boom = Object.assign(new Error("shape"), { code: "bad_shape" });
  await runExplainEvalCase({ env: {} as never, sb, message: { eval_run_id: "r1", case_id: "c1", candidate } }, { callModel: (async () => { throw boom; }) as never });
  assert.equal(writes[0].status, "failed");
  assert.equal(writes[0].error_code, "bad_shape");
  assert.equal(writes[0].schema_valid, false);
  assert.equal(writes[0].faithfulness, undefined);
});

test("a judge that fails leaves the score empty and says so, never a made-up number", async () => {
  const { sb, writes } = fakeDb(INPUT);
  const deps: EvalDeps = {
    callModel: (async (opts: { routeOverride?: { primary_model?: string } }) => {
      if (opts.routeOverride?.primary_model === "gemini-3.1-pro-preview") throw new Error("judge down");
      return { parsed: EXPLANATION, latencyMs: 100, promptVersion: "paper_feedback.v2" };
    }) as never,
  };
  await runExplainEvalCase({ env: {} as never, sb, message: { eval_run_id: "r1", case_id: "c1", candidate } }, deps);
  assert.equal(writes[0].status, "done");
  assert.equal(writes[0].faithfulness, null);
  assert.ok((writes[0].judge_flags as string[]).includes("judge_failed"));
});
