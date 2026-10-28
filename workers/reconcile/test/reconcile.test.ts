import { beforeEach, expect, test, vi } from "vitest";
import { applied, fakeDb, stale } from "../../../shared/src/__tests__/support/fake-db.js";

const m = vi.hoisted(() => ({ failRun: vi.fn() }));
vi.mock("@mastery/shared/worker.js", async (orig) => ({
  ...await orig<any>(),
  consumeQueue: (handle: any, onPermanent: any) => Object.assign(handle, { onPermanent }),
  failRun: m.failRun,
}));

import worker from "../src/index.js";
const handle = worker.queue as any;

const run = { id: "run", paper_id: "paper", student_id: "student", status: "content", tier_routing: null };
const region = (i: number, awarded: number | null, available = 5) => ({
  id: `r${i}`, order_index: i, question_label: `${i + 1}`, marks_awarded: awarded, marks_available: available,
  confidence_tier: "confident", confidence_signals: { recognition_confidence: "high" }, extract_status: "done",
  page_spans: [{ page: 1, box: { x: 0, y: i * 100, w: 100, h: 90 } }],
  student_answer: "x = 2", question_text: `Question ${i + 1}`, answer_block: null,
});

function db(over: { run?: any; paper?: any; regions?: any[]; start?: any; result?: any; pages?: any[] } = {}) {
  return fakeDb({
    extraction_run: over.run === undefined ? run : over.run,
    paper: over.paper ?? { reported_total: 8, stated_maximum: 10 },
    question_region: over.regions ?? [region(0, 5), region(1, 3)],
    paper_page: over.pages ?? [],
  }, (name, args) => {
    if (name !== "pipeline_write") return null;
    if (args.p_stage === "reconcile_start") return over.start ?? applied;
    return over.result ?? applied;
  });
}
const queue = () => ({ send: vi.fn(async () => {}) });
const ctx = (f: ReturnType<typeof fakeDb>, env: any = { ADJUDICATE_QUEUE: queue() }) =>
  ({ env, sb: f.sb, msg: { run_id: "run" }, attempt: 1, beat: async () => {} });
const result = (f: ReturnType<typeof fakeDb>) => f.rpcNamed("pipeline_write").find((a) => a.p_stage === "reconcile_result");

beforeEach(() => m.failRun.mockReset());

test("a run that is already finished, failed or waiting on review is left alone", async () => {
  for (const status of ["failed", "rejected", "committed", "needs_review", "ready"]) {
    const f = db({ run: { ...run, status } });
    expect(await handle(ctx(f))).toEqual({ detail: { skipped: status } });
    expect(f.rpcCalls).toEqual([]);
  }
});

test("a run stuck in adjudicating is re-dispatched, and a missing queue is an error rather than a silent stop", async () => {
  const env = { ADJUDICATE_QUEUE: queue() };
  expect(await handle(ctx(db({ run: { ...run, status: "adjudicating" } }), env))).toEqual({ detail: { resumed: "adjudication dispatch" } });
  expect(env.ADJUDICATE_QUEUE.send).toHaveBeenCalledWith({ run_id: "run" });
  await expect(handle(ctx(db({ run: { ...run, status: "adjudicating" } }), {}))).rejects.toThrow(/not configured/);
});

test("a reconciliation another worker already started is not repeated", async () => {
  const f = db({ start: stale });
  expect(await handle(ctx(f))).toEqual({ detail: { skipped: "stale reconciliation" } });
  expect(result(f)).toBeUndefined();
});

test("no questions at all fails the run visibly", async () => {
  expect(await handle(ctx(db({ regions: [] })))).toEqual({ detail: { failed: "no questions" } });
  expect(m.failRun).toHaveBeenCalledWith(expect.anything(), "run", expect.stringMatching(/could not find any questions/), "reconcile_no_questions");
});

test("marks that add up to the printed total go to review, not adjudication", async () => {
  const f = db();
  const env = { ADJUDICATE_QUEUE: queue() };
  const out = await handle(ctx(f, env));
  expect(out.detail).toMatchObject({ reconciled: true, unchecked: false, questions: 2 });
  const w = result(f);
  expect(w.p_args.to).toBe("needs_review");
  expect(w.p_args.paper_result).toMatchObject({ total_awarded: 8, reconciled: true, total_basis: "printed" });
  expect(w.p_args.confidence.map((c: any) => c.id)).toEqual(["r0", "r1"]);
  // Signals the region already had survive the reconcile verdict.
  expect(w.p_args.confidence[0].signals.recognition_confidence).toBe("high");
  expect(env.ADJUDICATE_QUEUE.send).not.toHaveBeenCalled();
});

test("marks that do not add up go to adjudication and the queue is told", async () => {
  const f = db({ paper: { reported_total: 9, stated_maximum: 10 } });
  const env = { ADJUDICATE_QUEUE: queue() };
  const out = await handle(ctx(f, env));
  expect(out.detail.reconciled).toBe(false);
  expect(out.detail.adjudicate.length).toBeGreaterThan(0);
  expect(result(f).p_args.to).toBe("adjudicating");
  expect(result(f).p_args.run_result.reconcile_delta).not.toBe(0);
  expect(env.ADJUDICATE_QUEUE.send).toHaveBeenCalledWith({ run_id: "run" });
});

test("a paper with no printed total is unchecked and labelled as added up, never as a mismatch", async () => {
  const f = db({ paper: { reported_total: null, stated_maximum: null } });
  const out = await handle(ctx(f));
  expect(out.detail).toMatchObject({ reconciled: null, unchecked: true });
  const w = result(f);
  expect(w.p_args.run_result).toMatchObject({ reconciled: null, status_reason_code: "no_printed_total" });
  expect(w.p_args.paper_result).toMatchObject({ total_awarded: 8, total_basis: "added_up" });
});

test("a result written after the run moved on is reported stale and does not dispatch", async () => {
  const env = { ADJUDICATE_QUEUE: queue() };
  const out = await handle(ctx(db({ paper: { reported_total: 9, stated_maximum: 10 }, result: stale }), env));
  expect(out).toEqual({ detail: { skipped: "stale reconciliation result" } });
  expect(env.ADJUDICATE_QUEUE.send).not.toHaveBeenCalled();
});

test("a permanent failure keeps the pages when they exist and says so", async () => {
  await handle.onPermanent(ctx(db({ pages: [{ id: "p1" }] })), new Error("db down"));
  expect(m.failRun).toHaveBeenCalledWith(expect.anything(), "run", expect.stringMatching(/your pages are kept/), expect.any(String));
  m.failRun.mockReset();
  await handle.onPermanent(ctx(db({ pages: [] })), new Error("db down"));
  expect(m.failRun).toHaveBeenCalledWith(expect.anything(), "run", expect.stringMatching(/could not find the pages/), "reconcile_pages_missing");
});
