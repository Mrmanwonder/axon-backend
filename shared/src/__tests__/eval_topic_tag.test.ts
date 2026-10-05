import { test } from "node:test";
import assert from "node:assert/strict";
import { scoreTopicTag, summariseTopicTag } from "../eval/topic-tag-score.js";
import { runTopicTagEvalCase, type TopicTagEvalDeps } from "../eval/topic-tag-run.js";

const EXPECT = { primary: ["2.2.3", "3.2.3"], also_ok: ["2.2.1"], control: null };

test("a primary tag from the accepted set is a hit; a strong tag outside the set is wrong", () => {
  const s = scoreTopicTag({ canTag: true, tags: [
    { code: "3.2.3", strength: "strong", primary: true },
    { code: "2.2.1", strength: "strong", primary: false },
    { code: "1.1.3", strength: "strong", primary: false },
    { code: "1.6.1", strength: "partial", primary: false },
  ] }, EXPECT);
  assert.equal(s.primary_hit, true);
  assert.deepEqual(s.wrong_strong, ["1.1.3"], "a partial tag never reaches analytics, so it is not counted wrong");
  assert.equal(s.strong, 3);
});

test("a control is right only when nothing strong is placed", () => {
  const control = { primary: [], also_ok: [], control: "off_syllabus" as const };
  assert.equal(scoreTopicTag({ canTag: false, tags: [] }, control).control_ok, true);
  assert.equal(scoreTopicTag({ canTag: true, tags: [{ code: "1.1.1", strength: "partial", primary: true }] }, control).control_ok, true);
  const wrong = scoreTopicTag({ canTag: true, tags: [{ code: "1.1.1", strength: "strong", primary: true }] }, control);
  assert.equal(wrong.control_ok, false);
  assert.deepEqual(wrong.wrong_strong, ["1.1.1"]);
  assert.equal(wrong.primary_hit, null);
});

test("abstaining on a real question is a miss, never a silent pass", () => {
  const s = scoreTopicTag({ canTag: false, tags: [] }, EXPECT);
  assert.equal(s.primary_hit, false);
  assert.equal(s.abstained, true);
});

test("the summary counts failures against schema validity and primary hits", () => {
  const hit = scoreTopicTag({ canTag: true, tags: [{ code: "2.2.3", strength: "strong", primary: true }] }, EXPECT);
  const miss = scoreTopicTag({ canTag: true, tags: [{ code: "1.1.3", strength: "strong", primary: true }] }, EXPECT);
  const ctl = scoreTopicTag({ canTag: false, tags: [] }, { primary: [], also_ok: [], control: "no_context" });
  const sum = summariseTopicTag([
    { status: "done", score: hit }, { status: "done", score: miss }, { status: "failed", score: null }, { status: "done", score: ctl },
  ]);
  assert.equal(sum.schema_valid, "3/4");
  assert.equal(sum.primary_hit, "1/3");
  assert.equal(sum.wrong_strong_tags, 1);
  assert.equal(sum.strong_precision, 0.5);
  assert.equal(sum.controls_ok, "1/1");
});

function fakeDb(input: unknown, opts: { stage?: string; source?: string; doc?: boolean } = {}) {
  const writes: Array<Record<string, unknown>> = [];
  const touched: string[] = [];
  const chain = (data: unknown) => {
    const q: Record<string, unknown> = {};
    for (const m of ["select", "eq", "neq", "order", "limit"]) q[m] = () => q;
    q.maybeSingle = async () => ({ data, error: null });
    q.then = (resolve: (v: unknown) => void) => resolve({ data, error: null });
    return q;
  };
  const sb = {
    from(table: string) {
      touched.push(table);
      if (table === "eval_case") return chain({ id: "c1", stage: opts.stage ?? "topic_tag", source: opts.source ?? "synthetic", input });
      if (table === "syllabus_document") return chain(opts.doc === false ? null : { id: "d1", title: "Mathematics", syllabus_code: "9709", version_label: "2026-2027" });
      if (table === "syllabus_topic") return chain([
        { id: "t1", parent_id: null, code: "2.2", kind: "topic", title: "Logarithmic and exponential functions", objective_text: null },
        { id: "o1", parent_id: "t1", code: "2.2.3", kind: "objective", title: "use logarithms", objective_text: "use logarithms to solve equations" },
      ]);
      return { upsert: async (row: Record<string, unknown>) => { writes.push(row); return { error: null }; } };
    },
  } as never;
  return { sb, writes, touched };
}

const INPUT = { syllabus_code: "9709", label: "3", marks_available: 4, question_text: "Solve 5^x = 3.", student_answer: "x = 0.683", expected: EXPECT };
const candidate = { key: "a", model: "gemini-3.8-flash", thinking_level: "medium" as const };

test("a case is run against the published syllabus and written as one scored row", async () => {
  const { sb, writes, touched } = fakeDb(INPUT);
  let shown = "";
  const deps: TopicTagEvalDeps = {
    callModel: (async (o: { instruction: string; validate: (v: unknown) => unknown }) => {
      shown = o.instruction;
      return { parsed: o.validate({ can_tag: true, tags: [{ code: "2.2.3", strength: "strong", primary: true }, { code: "9.9.9", strength: "strong", primary: false }] }), latencyMs: 1500, promptVersion: "topic_tag.v1", costUsd: 0.001 };
    }) as never,
  };
  await runTopicTagEvalCase({ env: {} as never, sb, message: { eval_run_id: "r1", case_id: "c1", candidate } }, deps);
  assert.match(shown, /2\.2\.3 \[2\.2 Logarithmic and exponential functions\] use logarithms to solve equations/);
  assert.equal(writes.length, 1);
  assert.equal(writes[0].status, "done");
  const out = writes[0].output as { score: { primary_hit: boolean; wrong_strong: string[] }; tags: unknown[] };
  assert.equal(out.score.primary_hit, true);
  assert.equal(out.tags.length, 1, "a code outside the closed list is dropped before scoring");
  assert.deepEqual([...new Set(touched)].sort(), ["eval_case", "eval_case_result", "syllabus_document", "syllabus_topic"]);
});

test("an explain case or a production case is not run here", async () => {
  for (const opts of [{ stage: "explain" }, { source: "production" }]) {
    const { sb, writes } = fakeDb(INPUT, opts);
    let called = false;
    await runTopicTagEvalCase({ env: {} as never, sb, message: { eval_run_id: "r1", case_id: "c1", candidate } }, { callModel: (async () => { called = true; }) as never });
    assert.equal(called, false);
    assert.equal(writes[0].status, "skipped");
  }
});

test("a missing syllabus is a skip with its reason, and a model failure keeps its code", async () => {
  const missing = fakeDb(INPUT, { doc: false });
  await runTopicTagEvalCase({ env: {} as never, sb: missing.sb, message: { eval_run_id: "r1", case_id: "c1", candidate } }, { callModel: (async () => { throw new Error("no"); }) as never });
  assert.equal(missing.writes[0].skipped_reason, "syllabus_not_loaded");
  const failing = fakeDb(INPUT);
  const boom = Object.assign(new Error("shape"), { code: "bad_shape" });
  await runTopicTagEvalCase({ env: {} as never, sb: failing.sb, message: { eval_run_id: "r1", case_id: "c1", candidate } }, { callModel: (async () => { throw boom; }) as never });
  assert.equal(failing.writes[0].status, "failed");
  assert.equal(failing.writes[0].error_code, "bad_shape");
});
