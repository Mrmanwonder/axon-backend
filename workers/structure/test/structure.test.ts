import { beforeEach, expect, test, vi } from "vitest";
import { applied, fakeDb, stale } from "../../../shared/src/__tests__/support/fake-db.js";
import { ConfigurationError } from "@mastery/shared/errors.js";
import { UNPLACEABLE_PAGE_REASON } from "@mastery/shared/page.js";

// Concurrency, re-planning and retake recovery for this worker are covered in
// workers/content/test/structure-*.test.ts. This suite covers routing and the
// terminal and unreadable paths (AXO-296).

const m = vi.hoisted(() => ({
  callModel: vi.fn(), imageRef: vi.fn(), failRun: vi.fn(), loadStructurePage: vi.fn(), pageDimensions: vi.fn(),
}));
vi.mock("@mastery/shared/worker.js", async (orig) => ({
  ...await orig<any>(),
  consumeQueue: (handle: any, onPermanent: any) => Object.assign(handle, { onPermanent }),
  failRun: m.failRun,
}));
vi.mock("@mastery/shared/model-client.js", async (orig) => ({ ...await orig<any>(), callModel: m.callModel }));
vi.mock("@mastery/shared/r2.js", () => ({ imageRef: m.imageRef }));
vi.mock("@mastery/shared/structure-page.js", () => ({ loadStructurePage: m.loadStructurePage }));
vi.mock("@mastery/shared/checkpoint.js", () => ({ digestHex: async () => "digest", markWorkApplied: async () => {} }));
vi.mock("@mastery/shared/page.js", async (orig) => ({ ...await orig<any>(), pageDimensions: m.pageDimensions }));

import worker, { enqueueFromAdvance } from "../src/index.js";
const handle = worker.queue as any;

const page = (over: any = {}) => ({
  id: "pg", paper_id: "paper", student_id: "student", page_number: 1, r2_bucket: "derived", r2_key: "page/1",
  mask_key: null, structure_status: "pending", teacher_marks: [], etag: null, sha256: null, preprocess_version: 1, layer_fallback: null, ...over,
});
const region = { label: "1", label_box: { x: 10, y: 10, w: 20, h: 20 }, box: { x: 10, y: 10, w: 900, h: 300 }, region_type: "text", continues_from_previous_page: false };

function db(over: { run?: any; regions?: any[]; write?: any; advance?: any; failedPage?: any } = {}) {
  return fakeDb({
    extraction_run: over.run === undefined ? { status: "structure", route_override: null, paper_id: "paper" } : over.run,
    question_region: over.regions ?? [],
    paper_page: over.failedPage === undefined ? { paper_id: "paper", student_id: "student", page_number: 1, r2_key: "page/1" } : over.failedPage,
  }, (name, args) => {
    if (name === "pipeline_write") return args.p_args?.patch?.structure_status === "running" ? applied : over.write ?? applied;
    if (name === "advance_after_structure") return over.advance ?? { advanced: true, enqueue_content: ["q1"] };
    return null;
  });
}
const env = () => ({
  CROP_QUEUE: { sendBatch: vi.fn(async () => {}) }, CONTENT_QUEUE: { sendBatch: vi.fn(async () => {}) }, RECONCILE_QUEUE: { send: vi.fn(async () => {}) },
});
const ctx = (f: ReturnType<typeof fakeDb>, e: any = env(), redelivered = false) =>
  ({ env: e, sb: f.sb, signal: new AbortController().signal, msg: { run_id: "run", page_id: "pg" }, attempt: 1, redelivered, beat: async () => {} });
const last = (f: ReturnType<typeof fakeDb>) => f.rpcNamed("pipeline_write").at(-1);

beforeEach(() => {
  for (const fn of Object.values(m)) fn.mockReset();
  m.loadStructurePage.mockResolvedValue(page());
  m.imageRef.mockImplementation(async (_e: any, bucket: string, key: string, detail: string) => ({ key, url: `${bucket}/${key}`, detail }));
  m.pageDimensions.mockReturnValue({ width: 1000, height: 1400, source: "recorded" });
  m.callModel.mockResolvedValue({ parsed: { is_graded_exam_paper: true, regions: [region] }, workId: "w1" });
});

test("advance routing: crop is preferred when the crop stage is on, content otherwise, then reconcile", async () => {
  const e = env();
  await enqueueFromAdvance(e as any, "run", { advanced: true, enqueue_crop: ["pg"], enqueue_content: ["q1"], enqueue_reconcile: true }, false);
  expect(e.CROP_QUEUE.sendBatch).toHaveBeenCalledWith([{ body: { run_id: "run", page_id: "pg" } }]);
  expect(e.CONTENT_QUEUE.sendBatch).not.toHaveBeenCalled();
  expect(e.RECONCILE_QUEUE.send).toHaveBeenCalledWith({ run_id: "run" });

  const e2 = env();
  await enqueueFromAdvance(e2 as any, "run", { advanced: true, enqueue_content: ["q1", "q2"] }, false);
  expect(e2.CONTENT_QUEUE.sendBatch).toHaveBeenCalledWith([{ body: { run_id: "run", region_id: "q1" } }, { body: { run_id: "run", region_id: "q2" } }]);
});

test("a crop request without a crop queue binding fails loudly and names the fix", async () => {
  await expect(enqueueFromAdvance({} as any, "run", { advanced: true, enqueue_crop: ["pg"] }, false)).rejects.toThrow(/CROP_QUEUE.*crop_stage/s);
});

test("a run that is finished or past structure is left alone", async () => {
  for (const status of ["failed", "rejected", "committed", "needs_review", "ready", "explaining"]) {
    const f = db({ run: { status, route_override: null } });
    expect(await handle(ctx(f))).toEqual({ detail: { skipped: status } });
    expect(f.rpcCalls).toEqual([]);
  }
  expect(await handle(ctx(db({ run: null })))).toEqual({ detail: { skipped: "no run" } });
});

test("a page that is already done only advances the run", async () => {
  m.loadStructurePage.mockResolvedValue(page({ structure_status: "done" }));
  const f = db();
  expect(await handle(ctx(f))).toEqual({ detail: { skipped: "already done" } });
  expect(m.callModel).not.toHaveBeenCalled();
  expect(f.rpcNamed("advance_after_structure")).toHaveLength(1);
});

test("the model sees the page at full detail and the ink mask at low detail", async () => {
  m.loadStructurePage.mockResolvedValue(page({ mask_key: "mask/1" }));
  await handle(ctx(db()));
  expect(m.callModel.mock.calls[0][0].images).toEqual([
    { key: "page/1", url: "derived/page/1", detail: "high" },
    { key: "mask/1", url: "derived/mask/1", detail: "low" },
  ]);
});

test("a page that is not part of a marked paper is marked unreadable with the model's reason", async () => {
  m.callModel.mockResolvedValue({ parsed: { is_graded_exam_paper: false, not_a_paper_reason: "This is a cover sheet.", regions: [] }, workId: "w1" });
  const f = db();
  expect(await handle(ctx(f))).toEqual({ detail: { unreadable: true } });
  expect(last(f).p_args).toMatchObject({ patch: { structure_status: "unreadable" }, unreadable_reason: "This is a cover sheet." });
});

test("a page with no known size cannot place boxes and says so", async () => {
  m.pageDimensions.mockReturnValue(null);
  const f = db();
  expect(await handle(ctx(f))).toEqual({ detail: { unreadable: "no page dimensions" } });
  expect(last(f).p_args.unreadable_reason).toBe(UNPLACEABLE_PAGE_REASON);
});

test("a read page writes its regions, then advances and sends content", async () => {
  const f = db();
  const e = env();
  const out = await handle(ctx(f, e));
  expect(out.detail).toEqual({ regions: 1, marks: 0 });
  const w = last(f);
  expect(w.p_args.patch).toEqual({ structure_status: "done" });
  expect(w.p_args.regions).toHaveLength(1);
  expect(w.p_args.regions[0]).toMatchObject({ run_id: "run", paper_id: "paper" });
  expect(e.CONTENT_QUEUE.sendBatch).toHaveBeenCalledTimes(1);
});

test("teacher marks found on the device are attached to the page's regions, all unsure", async () => {
  m.loadStructurePage.mockResolvedValue(page({ teacher_marks: [{ page_number: 1, box: { x: 950, y: 100, w: 30, h: 30 }, shape: "tick" }] }));
  const f = db();
  const out = await handle(ctx(f));
  expect(out.detail.marks).toBe(1);
  const marks = last(f).p_args.teacher_marks;
  expect(marks).toHaveLength(1);
  expect(marks[0].confidence_tier).toBe("unsure");
});

test("a stale result does not advance the run", async () => {
  const f = db({ write: stale });
  expect(await handle(ctx(f))).toEqual({ detail: { skipped: "stale structure result" } });
  expect(f.rpcNamed("advance_after_structure")).toEqual([]);
});

test("a configuration failure fails the run with pages kept", async () => {
  await handle.onPermanent(ctx(db()), new ConfigurationError("x", "missing column"));
  expect(m.failRun).toHaveBeenCalledWith(expect.anything(), "run", expect.stringMatching(/Your pages are kept/), expect.any(String));
});

test("any other permanent failure marks just the page failed and lets the run go on", async () => {
  const f = db();
  await handle.onPermanent(ctx(f), new Error("model gave up"));
  expect(last(f).p_args.patch).toEqual({ structure_status: "failed" });
  expect(f.rpcNamed("advance_after_structure")).toHaveLength(1);
  expect(m.failRun).not.toHaveBeenCalled();
});

test("a permanent failure for a page that no longer exists fails the run as pages missing", async () => {
  await handle.onPermanent(ctx(db({ failedPage: null })), new Error("gone"));
  expect(m.failRun).toHaveBeenCalledWith(expect.anything(), "run", expect.stringMatching(/could not find the pages/), "structure_pages_missing");
});
