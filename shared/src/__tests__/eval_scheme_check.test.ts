import { test } from "node:test";
import assert from "node:assert/strict";
import { scoreSchemeCheck, summariseSchemeCheck, passes } from "../eval/scheme-check-score.js";
import { runSchemeCheckEvalCase, type SchemeCheckEvalDeps } from "../eval/scheme-check-run.js";
import type { CheckResult } from "../prompts/scheme_check.v1.js";

const checked = (est: number, max = 4): CheckResult => ({
  canCheck: true, reason: null, estimatedMarks: est, maxMarks: max, confidence: "likely",
  whatWasRight: "Method was sound.", whatWasMissing: [], doThisNext: null,
});
const unchecked: CheckResult = {
  canCheck: false, reason: "Different question.", estimatedMarks: null, maxMarks: null, confidence: "unsure",
  whatWasRight: null, whatWasMissing: [], doThisNext: null,
};

test("an estimate inside the accepted range is a hit; above it is an over-estimate", () => {
  const e = { can_check: true, est: [2, 3] as [number, number], control: null };
  assert.equal(scoreSchemeCheck(checked(3), e).in_range, true);
  const over = scoreSchemeCheck(checked(4), e);
  assert.equal(over.in_range, false);
  assert.equal(over.over_estimate, true);
  assert.equal(over.abs_error, 1);
  const under = scoreSchemeCheck(checked(1), e);
  assert.equal(under.over_estimate, false);
  assert.equal(under.abs_error, 1);
});

test("refusing a checkable question is a miss, never a silent pass", () => {
  const s = scoreSchemeCheck(unchecked, { can_check: true, est: [4, 4], control: null });
  assert.equal(s.can_check_ok, false);
  assert.equal(s.in_range, false);
});

test("a mismatched scheme is right only when refused", () => {
  const e = { can_check: false, est: null, control: "mismatch" as const };
  assert.equal(scoreSchemeCheck(unchecked, e).can_check_ok, true);
  assert.equal(scoreSchemeCheck(checked(3), e).can_check_ok, false);
});

test("the summary fails the run when any control is wrong", () => {
  const rows = [
    { status: "done", score: scoreSchemeCheck(checked(0), { can_check: true, est: [0, 0], control: "blank" }) },
    { status: "done", score: scoreSchemeCheck(checked(3), { can_check: false, est: null, control: "mismatch" }) },
    { status: "done", score: scoreSchemeCheck(checked(4), { can_check: true, est: [4, 4], control: null }) },
  ];
  const s = summariseSchemeCheck(rows);
  assert.equal(s.blank_zero, "1/1");
  assert.equal(s.mismatch_refused, "0/1");
  assert.equal(s.controls_all_ok, false);
  assert.equal(passes(s, 2), false);
});

test("a failed call counts against schema validity", () => {
  const s = summariseSchemeCheck([
    { status: "done", score: scoreSchemeCheck(checked(4), { can_check: true, est: [4, 4], control: null }) },
    { status: "failed", score: null },
  ]);
  assert.equal(s.schema_valid, "1/2");
  assert.equal(s.schema_valid_rate, 0.5);
});

function fakeDb(input: unknown, opts: { stage?: string; source?: string } = {}) {
  const writes: Array<Record<string, unknown>> = [];
  const touched: string[] = [];
  const chain = (data: unknown) => {
    const q: Record<string, unknown> = {};
    for (const m of ["select", "eq"]) q[m] = () => q;
    q.maybeSingle = async () => ({ data, error: null });
    return q;
  };
  const sb = {
    from(table: string) {
      touched.push(table);
      if (table === "eval_case") return chain({ id: "case-1", stage: opts.stage ?? "scheme_check", source: opts.source ?? "synthetic", input });
      if (table === "eval_case_result") return { upsert: async (row: Record<string, unknown>) => { writes.push(row); return { error: null }; } };
      throw new Error(`unexpected table ${table}`);
    },
  };
  return { sb: sb as never, writes, touched };
}

const INPUT = {
  syllabus_code: "9709", paper: "9709/12 Synthetic", label: "1", marks_available: 3,
  question_text: "Q", student_answer: "A", scheme: "step | M1", conventions: null,
  expected: { can_check: true, est: [3, 3], control: null },
};
const message = { eval_run_id: "run-1", case_id: "case-1", candidate: { key: "k", model: "gemini-3.8-flash", thinking_level: "high" as const } };

test("the runner writes only its eval_case_result row, with no case text in it", async () => {
  const { sb, writes, touched } = fakeDb(INPUT);
  const deps: SchemeCheckEvalDeps = {
    callModel: (async (o: { validate: (v: unknown) => CheckResult; routeOverride?: { primary_model?: string } }) => {
      assert.equal(o.routeOverride?.primary_model, "gemini-3.8-flash");
      const parsed = o.validate({ can_check: true, reason: null, estimated_marks: 3, max_marks: 3, confidence: "likely", what_was_right: "All steps shown.", what_was_missing: [], do_this_next: null });
      return { parsed, promptVersion: "scheme_check.v1", latencyMs: 10, costUsd: 0.001, model: "gemini-3.8-flash" };
    }) as never,
  };
  const out = await runSchemeCheckEvalCase({ env: {} as never, sb, message }, deps);
  assert.equal(out.in_range, true);
  assert.deepEqual([...new Set(touched)], ["eval_case", "eval_case_result"]);
  assert.equal(writes.length, 1);
  assert.equal(writes[0].status, "done");
  const json = JSON.stringify(writes[0]);
  assert.ok(!json.includes("step | M1"), "scheme text never lands in the result row");
});

test("a validator rejection (scheme notation shown to the student) is a failed case", async () => {
  const { sb, writes } = fakeDb(INPUT);
  const deps: SchemeCheckEvalDeps = {
    callModel: (async (o: { validate: (v: unknown) => CheckResult }) => {
      try {
        o.validate({ can_check: true, reason: null, estimated_marks: 3, max_marks: 3, confidence: "likely", what_was_right: "You earned M1 for the method.", what_was_missing: [], do_this_next: null });
      } catch (cause) {
        throw Object.assign(new Error(String(cause)), { code: "bad_shape" });
      }
      throw new Error("validator should have rejected");
    }) as never,
  };
  const out = await runSchemeCheckEvalCase({ env: {} as never, sb, message }, deps);
  assert.equal(out.failed, "bad_shape");
  assert.equal(writes[0].status, "failed");
  assert.equal(writes[0].schema_valid, false);
});

test("a case from another stage is skipped, not run", async () => {
  const { sb, writes } = fakeDb(INPUT, { stage: "topic_tag" });
  const out = await runSchemeCheckEvalCase({ env: {} as never, sb, message }, { callModel: (async () => { throw new Error("must not call"); }) as never });
  assert.equal(out.skipped, true);
  assert.equal(writes[0].skipped_reason, "not_a_scheme_check_case");
});
