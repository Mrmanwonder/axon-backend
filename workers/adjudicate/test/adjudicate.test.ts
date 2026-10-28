import { beforeEach, expect, test, vi } from "vitest";
import { applied, fakeDb, stale } from "../../../shared/src/__tests__/support/fake-db.js";

const m = vi.hoisted(() => ({ callModel: vi.fn(), imageRef: vi.fn() }));
vi.mock("@mastery/shared/worker.js", async (orig) => ({
  ...await orig<any>(),
  consumeQueue: (handle: any, onPermanent: any) => Object.assign(handle, { onPermanent }),
}));
vi.mock("@mastery/shared/model-client.js", async (orig) => ({ ...await orig<any>(), callModel: m.callModel }));
vi.mock("@mastery/shared/r2.js", () => ({ imageRef: m.imageRef }));

import worker from "../src/index.js";
const handle = worker.queue as any;

const run = { id: "run", paper_id: "paper", student_id: "student", status: "adjudicating", reconcile_delta: 2, route_override: null };
const region = (i: number, tier = "confident") => ({
  id: `r${i}`, order_index: i, question_label: `${i}`, marks_awarded: 2, marks_available: 3,
  confidence_tier: tier, confidence_signals: { seen: true }, crop_key: `crop/${i}`, page_spans: [],
});

function db(over: { run?: any; paper?: any; regions?: any[]; write?: any } = {}) {
  return fakeDb({
    extraction_run: over.run === undefined ? run : over.run,
    paper: over.paper ?? { reported_total: 20, total_awarded: 18 },
    question_region: over.regions ?? [region(0), region(1, "unsure")],
    paper_page: { r2_bucket: "derived", r2_key: "page/1" },
  }, (name) => (name === "pipeline_write" ? over.write ?? applied : null));
}
const ctx = (f: ReturnType<typeof fakeDb>) => ({ env: {}, sb: f.sb, signal: new AbortController().signal, msg: { run_id: "run" }, attempt: 1, beat: async () => {} });
const writes = (f: ReturnType<typeof fakeDb>) => f.rpcNamed("pipeline_write");

beforeEach(() => {
  m.callModel.mockReset();
  m.imageRef.mockReset().mockImplementation(async (_env: any, bucket: string, key: string) => ({ url: `${bucket}/${key}` }));
  m.callModel.mockResolvedValue({ parsed: { cause: "misread", checked: true, corrections: [] } });
});

test("a missing run and a run past adjudication are skipped without a write", async () => {
  const gone = db({ run: null });
  expect(await handle(ctx(gone))).toEqual({ detail: { skipped: "no such run" } });
  const moved = db({ run: { ...run, status: "explaining" } });
  expect(await handle(ctx(moved))).toEqual({ detail: { skipped: "explaining" } });
  expect(writes(gone)).toEqual([]);
  expect(writes(moved)).toEqual([]);
  expect(m.callModel).not.toHaveBeenCalled();
});

test("a run with no regions advances without asking the model", async () => {
  const f = db({ regions: [] });
  expect(await handle(ctx(f))).toEqual({ detail: { skipped: "nothing to adjudicate" } });
  expect(writes(f)).toEqual([{ p_run_id: "run", p_stage: "adjudicate", p_args: {} }]);
  expect(m.callModel).not.toHaveBeenCalled();
});

test("the model sees at most six crops, least confident first, plus the cover page", async () => {
  const regions = [0, 1, 2, 3, 4, 5, 6, 7].map((i) => region(i, i === 7 ? "unreadable" : i === 6 ? "unsure" : "confident"));
  await handle(ctx(db({ regions })));
  const { images } = m.callModel.mock.calls[0][0];
  expect(images).toHaveLength(7);
  expect(images[0].url).toBe("derived/crop/7");
  expect(images[1].url).toBe("derived/crop/6");
  expect(images.at(-1).url).toBe("derived/page/1");
});

test("each correction marks its question unsure and for review, keeping earlier signals", async () => {
  m.callModel.mockResolvedValue({ parsed: { cause: "misread", checked: true, corrections: [
    { order_index: 1, field: "marks_awarded", corrected_value: 1, evidence: "the 2 is a 1" },
    { order_index: 99, field: "marks_awarded", corrected_value: 0, evidence: "no such question" },
  ] } });
  const f = db();
  expect(await handle(ctx(f))).toEqual({ detail: { cause: "misread", flagged: 1 } });
  const [w] = writes(f);
  expect(w.p_stage).toBe("adjudicate");
  expect(w.p_args.confidence).toEqual([{
    id: "r1", tier: "unsure", needs_review: true,
    signals: { seen: true, adjudication: { field: "marks_awarded", suggests: 1, evidence: "the 2 is a 1" } },
  }]);
  expect(w.p_args.reason).toBe("The marks do not quite add up. We have put the questions to check first.");
  expect(w.p_args.adjudication.blocks_commit).toBeUndefined();
});

test("a finding that the question labels are wrong blocks the commit", async () => {
  m.callModel.mockResolvedValue({ parsed: { cause: "structure", checked: true, corrections: [
    { order_index: 0, field: "question_label", corrected_value: "1a", evidence: "The pipeline seems to have misidentified the question labels or order." },
  ] } });
  const f = db();
  await handle(ctx(f));
  const [w] = writes(f);
  expect(w.p_args.adjudication.blocks_commit).toBe(true);
  expect(w.p_args.adjudication.blocked_reason).toMatch(/question labels/);
});

test("a paper with no printed total is never told its marks do not add up", async () => {
  const f = db({ paper: { reported_total: null, total_awarded: 18 } });
  await handle(ctx(f));
  expect(writes(f)[0].p_args.reason).toBe("Some marks on this paper were hard to read. We have put those questions to check first.");
  expect(m.callModel.mock.calls[0][0].instruction).toBeTruthy();
});

test("a result for a run that moved on is reported stale, not applied", async () => {
  expect(await handle(ctx(db({ write: stale })))).toEqual({ detail: { skipped: "stale adjudication result" } });
});

test("a permanent failure still advances the run with an honest reason", async () => {
  const f = db();
  await handle.onPermanent(ctx(f), new Error("model gave up"));
  expect(writes(f)).toEqual([{ p_run_id: "run", p_stage: "adjudicate", p_args: {
    reason: "The marks on this paper do not add up to the total written on it. Check the questions below.",
  } }]);
});
