import { beforeEach, expect, test, vi } from "vitest";
import { fakeDb } from "../../../shared/src/__tests__/support/fake-db.js";
import { SYSTEM as TIER1_SYSTEM } from "@mastery/shared/prompts/explain_tier1.v2.js";
import { SYSTEM as TIER2_SYSTEM } from "@mastery/shared/prompts/explain_tier2.v1.js";

const m = vi.hoisted(() => ({
  callModel: vi.fn(), resolveSchemeEvidence: vi.fn(),
  runExplainEvalCase: vi.fn(), runTopicTagEvalCase: vi.fn(), runSchemeCheckEvalCase: vi.fn(),
  tagRegion: vi.fn(), failTopicTag: vi.fn(),
  runSchemeCheck: vi.fn(), failSchemeCheck: vi.fn(), runSchemeCheckQuestion: vi.fn(), failSchemeCheckQuestion: vi.fn(),
}));
vi.mock("@mastery/shared/worker.js", async (orig) => ({
  ...await orig<any>(),
  consumeQueue: (handle: any, onPermanent: any) => Object.assign(handle, { onPermanent }),
}));
vi.mock("@mastery/shared/model-client.js", async (orig) => ({ ...await orig<any>(), callModel: m.callModel }));
vi.mock("@mastery/shared/assessment.js", () => ({ resolveSchemeEvidence: m.resolveSchemeEvidence }));
vi.mock("@mastery/shared/eval/explain-run.js", () => ({ runExplainEvalCase: m.runExplainEvalCase }));
vi.mock("@mastery/shared/eval/topic-tag-run.js", () => ({ runTopicTagEvalCase: m.runTopicTagEvalCase }));
vi.mock("@mastery/shared/eval/scheme-check-run.js", () => ({ runSchemeCheckEvalCase: m.runSchemeCheckEvalCase }));
vi.mock("@mastery/shared/topic_tag.js", () => ({ tagRegion: m.tagRegion, failTopicTag: m.failTopicTag }));
vi.mock("@mastery/shared/scheme_check.js", () => ({
  runSchemeCheck: m.runSchemeCheck, failSchemeCheck: m.failSchemeCheck,
  runSchemeCheckQuestion: m.runSchemeCheckQuestion, failSchemeCheckQuestion: m.failSchemeCheckQuestion,
}));

import worker from "../src/index.js";
const handle = worker.queue as any;

const region = (over: any = {}) => ({
  id: "q1", paper_id: "paper", student_id: "student", question_label: "1(a)", question_text: "Solve 2x + 3 = 11.",
  student_answer: "x = 7", teacher_remark: null, marks_awarded: 1, marks_available: 3, explain_status: "queued",
  student_confirmed_at: "2026-10-27T10:00:00Z", order_index: 0, ...over,
});
const answer = (over: any = {}) => ({
  can_explain: true, cause: "procedural_slip", body: "You subtracted 3 from one side only.",
  do_this_next: "Write the subtraction on both sides before you divide.", concepts: ["linear equations"],
  command_word: "solve", command_word_note: null, model_answer: null,
  loss_reasons: [{ marks: 2, error_type: "method", note: "unbalanced step" }], ...over,
});

function db(over: { region?: any; paper?: any; claimed?: boolean } = {}) {
  const r = over.region === undefined ? region() : over.region;
  return fakeDb({
    question_region: r ? [r] : null,
    extraction_run: { route_override: null },
    paper: over.paper ?? { subject: "Mathematics", tier: "tier_1" },
    student: { class_level: "IGCSE", programme_id: null, board: "cambridge" },
    teacher_mark: [{ mark_class: "cross", comment_text: null }],
  }, () => ({ advanced: true }), (q) =>
    // The claim update and finishSkippedExplanation both need a matched row.
    q.op === "update" ? (over.claimed === false && (q.payload as any)?.explain_status === "running" ? [] : [{ id: "q1" }]) : null);
}
const ctx = (f: ReturnType<typeof fakeDb>, msg: any = { run_id: "run", region_id: "q1" }) =>
  ({ env: {}, sb: f.sb, signal: new AbortController().signal, msg, attempt: 1, beat: async () => {} });
const upsert = (f: ReturnType<typeof fakeDb>) => f.writes.find((w) => w.table === "region_explanation")?.payload as any;
const statusUpdates = (f: ReturnType<typeof fakeDb>) => f.writes.filter((w) => w.table === "question_region" && w.op === "update").map((w) => (w.payload as any).explain_status);

beforeEach(() => {
  for (const fn of Object.values(m)) fn.mockReset();
  m.callModel.mockResolvedValue({ parsed: answer(), model: "gemini-3.8-flash", promptVersion: "explain_tier1.v2", webSources: [] });
  m.resolveSchemeEvidence.mockResolvedValue(null);
});

test("eval, topic tag and scheme check messages go to their own runners, not the pipeline", async () => {
  m.runExplainEvalCase.mockResolvedValue({ ok: "eval" });
  m.runTopicTagEvalCase.mockResolvedValue({ ok: "tt-eval" });
  m.runSchemeCheckEvalCase.mockResolvedValue({ ok: "sc-eval" });
  m.runSchemeCheck.mockResolvedValue({ ok: "sc" });
  m.runSchemeCheckQuestion.mockResolvedValue({ ok: "scq" });
  m.tagRegion.mockResolvedValue({ ok: "tag" });
  const f = db();
  expect(await handle(ctx(f, { eval: { x: 1 } }))).toEqual({ detail: { ok: "eval" } });
  expect(await handle(ctx(f, { eval_topic_tag: { x: 1 } }))).toEqual({ detail: { ok: "tt-eval" } });
  expect(await handle(ctx(f, { eval_scheme_check: { x: 1 } }))).toEqual({ detail: { ok: "sc-eval" } });
  expect(await handle(ctx(f, { scheme_check: { run_id: "run" } }))).toEqual({ detail: { ok: "sc" } });
  expect(await handle(ctx(f, { scheme_check_q: { run_id: "run" } }))).toEqual({ detail: { ok: "scq" } });
  expect(await handle(ctx(f, { topic_tag: { region_id: "q1", document_id: "d" } }))).toEqual({ detail: { ok: "tag" } });
  expect(m.tagRegion.mock.calls[0][0]).toMatchObject({ regionId: "q1", documentId: "d" });
  expect(m.callModel).not.toHaveBeenCalled();
});

test("a question already explained, or missing, is skipped", async () => {
  expect(await handle(ctx(db({ region: region({ explain_status: "done" }) })))).toEqual({ detail: { skipped: "already explained" } });
  expect(await handle(ctx(db({ region: null })))).toEqual({ detail: { skipped: "no such question" } });
});

test("nothing is explained before the student has confirmed the reading", async () => {
  const f = db({ region: region({ student_confirmed_at: null }) });
  expect(await handle(ctx(f))).toEqual({ detail: { skipped: "not confirmed" } });
  expect(statusUpdates(f)).toEqual(["pending"]);
  expect(m.callModel).not.toHaveBeenCalled();
});

test("a question with no marks lost is closed without a model call", async () => {
  const f = db({ region: region({ marks_awarded: 3 }) });
  expect(await handle(ctx(f))).toEqual({ detail: { skipped: "no marks lost" } });
  expect(f.writes.find((w) => (w.payload as any)?.explain_failure_reason)?.payload).toMatchObject({ explain_failure_reason: "no_marks_lost" });
  expect(m.callModel).not.toHaveBeenCalled();
});

test("a missing teacher mark is unknown, not zero: nothing is explained as lost (AXO-300)", async () => {
  for (const r of [region({ marks_awarded: null }), region({ marks_available: null })]) {
    const f = db({ region: r });
    expect(await handle(ctx(f))).toEqual({ detail: { skipped: "no teacher mark to explain" } });
    expect(f.writes.find((w) => (w.payload as any)?.explain_failure_reason)?.payload).toMatchObject({ explain_status: "skipped", explain_failure_reason: "insufficient_evidence" });
    expect(upsert(f)).toBeUndefined();
  }
  expect(m.callModel).not.toHaveBeenCalled();
});

test("a second delivery that loses the claim does not explain the question again", async () => {
  expect(await handle(ctx(db({ claimed: false })))).toEqual({ detail: { skipped: "already claimed by another delivery" } });
  expect(m.callModel).not.toHaveBeenCalled();
});

test("tier 1: explained from the teacher's marks, with web grounding, and the run advances", async () => {
  const f = db();
  const out = await handle(ctx(f));
  const call = m.callModel.mock.calls[0][0];
  expect(call.system).toBe(TIER1_SYSTEM);
  expect(call.webTools.searchContext).toContain("Mathematics");
  expect(call.webTools.searchContext).not.toContain("x = 7"); // the student's answer never reaches web search
  const row = upsert(f);
  expect(row).toMatchObject({ region_id: "q1", tier: "tier_1", cause: "procedural_slip", marks_lost: 2, scheme_source: null });
  expect(statusUpdates(f)).toEqual(["running", "done"]);
  expect(f.rpcNamed("advance_after_explain")).toEqual([{ p_run_id: "run" }]);
  expect(out.detail).toMatchObject({ explanation_tier: "tier_1", floor_cleared: true });
});

test("the explanation never carries a mark of its own", async () => {
  const f = db();
  await handle(ctx(f));
  const row = upsert(f);
  for (const key of ["marks_awarded", "marks_available", "total_awarded"]) expect(row).not.toHaveProperty(key);
  expect(f.writes.filter((w) => w.table === "question_region").every((w) => !("marks_awarded" in (w.payload as any)))).toBe(true);
});

test("tier 2: with a matched official scheme, the scheme prompt is used, no web search, and the source is cited", async () => {
  m.resolveSchemeEvidence.mockResolvedValue({
    canonicalQuestionId: "cq1", markingScheme: "M1 for subtracting 3 from both sides", sourceUrl: "https://cbseacademic.nic.in/ms.pdf",
    version: "2025", assessmentIdentityId: "ai1", schemeDocumentId: "sd1", retrievalMode: "exact",
  });
  const f = db({ paper: { subject: "Mathematics", tier: "tier_2" } });
  const out = await handle(ctx(f));
  const call = m.callModel.mock.calls[0][0];
  expect(call.system).toBe(TIER2_SYSTEM);
  expect(call.webTools).toBeUndefined();
  expect(upsert(f)).toMatchObject({ tier: "tier_2", scheme_source: "https://cbseacademic.nic.in/ms.pdf", scheme_version: "2025", canonical_question_id: "cq1" });
  expect(f.writes.find((w) => (w.payload as any)?.canonical_question_id === "cq1" && w.table === "question_region")).toBeTruthy();
  expect(out.detail.explanation_tier).toBe("tier_2");
});

test("a tier 2 paper whose question cannot be matched falls back to tier 1, never an approximated scheme", async () => {
  const f = db({ paper: { subject: "Mathematics", tier: "tier_2" } });
  await handle(ctx(f));
  expect(m.callModel.mock.calls[0][0].system).toBe(TIER1_SYSTEM);
  expect(upsert(f)).toMatchObject({ tier: "tier_1", scheme_source: null });
});

test("when the model cannot explain from the evidence, nothing is written but the skip", async () => {
  m.callModel.mockResolvedValue({ parsed: answer({ can_explain: false }), model: "m", promptVersion: "p", webSources: [] });
  const f = db();
  expect((await handle(ctx(f))).detail.skipped).toMatch(/could not explain/);
  expect(upsert(f)).toBeUndefined();
});

test("a 'do this next' that does not clear the quality floor is dropped, not shown", async () => {
  m.callModel.mockResolvedValue({ parsed: answer({ do_this_next: "Revise." }), model: "m", promptVersion: "p", webSources: [] });
  const f = db();
  const out = await handle(ctx(f));
  expect(upsert(f).do_this_next).toBeNull();
  expect(out.detail.floor_cleared).toBe(false);
});

test("loss reasons adding up to more than the marks lost are discarded", async () => {
  m.callModel.mockResolvedValue({ parsed: answer({ loss_reasons: [{ marks: 2, error_type: "method", note: "" }, { marks: 1, error_type: "final_answer", note: "" }] }), model: "m", promptVersion: "p", webSources: [] });
  const f = db();
  await handle(ctx(f));
  expect(upsert(f).loss_reasons).toEqual([]);
});

test("a permanent failure marks the question failed with a code and still advances the run", async () => {
  const f = db();
  await handle.onPermanent(ctx(f), new Error("model gave up"));
  const failed = f.writes.find((w) => w.table === "question_region")!.payload as any;
  expect(failed.explain_status).toBe("failed");
  expect(typeof failed.explain_failure_reason).toBe("string");
  expect(f.rpcNamed("advance_after_explain")).toEqual([{ p_run_id: "run" }]);
});

test("a permanent failure on side work goes to that work's own failure handler", async () => {
  const f = db();
  await handle.onPermanent(ctx(f, { topic_tag: { region_id: "q1", document_id: "d" } }), new Error("x"));
  expect(m.failTopicTag).toHaveBeenCalledTimes(1);
  await handle.onPermanent(ctx(f, { scheme_check: { run_id: "run" } }), new Error("x"));
  expect(m.failSchemeCheck).toHaveBeenCalledTimes(1);
  await handle.onPermanent(ctx(f, { eval: { x: 1 } }), new Error("x"));
  expect(f.writes).toEqual([]);
  expect(f.rpcCalls).toEqual([]);
});
