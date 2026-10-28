import { beforeEach, expect, test, vi } from "vitest";
import { applied, fakeDb, stale } from "../../../shared/src/__tests__/support/fake-db.js";

const W = 100, H = 200;
const m = vi.hoisted(() => ({
  decodeImage: vi.fn(), encodeWebp: vi.fn(), imageDimensions: vi.fn(), pageDimensions: vi.fn(),
}));
vi.mock("@mastery/shared/worker.js", async (orig) => ({
  ...await orig<any>(),
  consumeQueue: (handle: any, onPermanent: any) => Object.assign(handle, { onPermanent }),
}));
vi.mock("../src/codecs.js", () => ({ decodeImage: m.decodeImage, encodeWebp: m.encodeWebp }));
vi.mock("@mastery/shared/page.js", () => ({ pageDimensions: m.pageDimensions }));
vi.mock("@mastery/shared/crop.js", async (orig) => ({ ...await orig<any>(), imageDimensions: m.imageDimensions }));
vi.mock("@mastery/shared/r2.js", () => ({
  objectKey: ({ kind, name }: any) => `${kind}/${name}.webp`,
}));

import worker from "../src/index.js";
const handle = worker.queue as any;

const page = (over: any = {}) => ({
  id: "pg", paper_id: "paper", student_id: "student", page_number: 1, r2_bucket: "derived", r2_key: "page/1",
  mask_key: null, crop_status: "pending", conditioning_meta: {}, quality_signals: null, ...over,
});
const span = (y: number, pageNumber = 1) => ({ page: pageNumber, box: { x: 10, y, w: 80, h: 30 } });
const regions = [
  { id: "q1", order_index: 0, page_spans: [span(10)] },
  { id: "q2", order_index: 1, page_spans: [span(100)] },
  { id: "q3", order_index: 2, page_spans: [span(10), span(10, 2)] }, // spans two pages: full-page path
];

function db(over: { page?: any; run?: any; regions?: any[]; running?: any; advance?: any } = {}) {
  return fakeDb({
    paper_page: over.page === undefined ? page() : over.page,
    extraction_run: over.run === undefined ? { status: "structure" } : over.run,
    question_region: over.regions ?? regions,
  }, (name, args) => {
    if (name === "pipeline_write") return args.p_args?.patch?.crop_status === "running" ? over.running ?? applied : applied;
    if (name === "advance_after_crop") return over.advance ?? { advanced: true, enqueue_content: ["q1", "q2", "q3"], enqueue_reconcile: false };
    return null;
  });
}
const bucket = () => {
  const puts = new Map<string, unknown>();
  return {
    puts,
    get: vi.fn(async (key: string) => ({ arrayBuffer: async () => new TextEncoder().encode(key).buffer })),
    put: vi.fn(async (key: string, bytes: unknown) => { puts.set(key, bytes); }),
  };
};
const env = () => ({ DERIVED: bucket(), ORIGINALS: bucket(), CONTENT_QUEUE: { sendBatch: vi.fn(async () => {}) }, RECONCILE_QUEUE: { send: vi.fn(async () => {}) } });
const ctx = (f: ReturnType<typeof fakeDb>, e: any = env(), redelivered = false) =>
  ({ env: e, sb: f.sb, msg: { run_id: "run", page_id: "pg" }, redelivered, attempt: 1, beat: async () => {} });
const statuses = (f: ReturnType<typeof fakeDb>) => f.rpcNamed("pipeline_write").map((a) => a.p_args.patch.crop_status);
const rgba = (w = W, h = H) => ({ width: w, height: h, data: new Uint8ClampedArray(w * h * 4) });

beforeEach(() => {
  for (const fn of Object.values(m)) fn.mockReset();
  m.pageDimensions.mockReturnValue({ width: W, height: H, source: "recorded" });
  m.imageDimensions.mockReturnValue({ width: W, height: H });
  m.decodeImage.mockImplementation(async () => rgba());
  m.encodeWebp.mockImplementation(async (img: any) => new Uint8Array([img.width, img.height]));
});

test("a single-page region is cut and stored, a multi-page one is left to the full page, and content is sent every region", async () => {
  const f = db();
  const e = env();
  const out = await handle(ctx(f, e));
  expect(out.detail).toMatchObject({ regions: 2, planned: 2, with_mask: 0, multi_page: 1, over_budget: 0 });
  expect([...e.DERIVED.puts.keys()].sort()).toEqual(["crop/run-q1.webp", "crop/run-q2.webp"]);
  const [applyCrops] = f.rpcNamed("apply_region_crops");
  expect(applyCrops.p_rows).toEqual(expect.arrayContaining([
    { id: "q1", crop_key: "crop/run-q1.webp", cropmask_key: null },
    { id: "q2", crop_key: "crop/run-q2.webp", cropmask_key: null },
  ]));
  expect(statuses(f)).toEqual(["running", "done"]);
  expect(e.CONTENT_QUEUE.sendBatch).toHaveBeenCalledWith([
    { body: { run_id: "run", region_id: "q1" } }, { body: { run_id: "run", region_id: "q2" } }, { body: { run_id: "run", region_id: "q3" } },
  ]);
});

test("a crop band covers the full page width", async () => {
  await handle(ctx(db({ regions: [regions[0]] })));
  const cut = m.encodeWebp.mock.calls[0][0];
  expect(cut.width).toBe(W);
  expect(cut.height).toBeLessThan(H);
});

test("one crop failing costs that crop only", async () => {
  m.encodeWebp.mockImplementationOnce(async () => { throw new Error("encode failed"); });
  const f = db();
  const out = await handle(ctx(f));
  expect(out.detail.regions).toBe(1);
  expect(statuses(f).at(-1)).toBe("done");
});

test("every crop failing marks the page failed, and content still goes ahead on full pages", async () => {
  m.encodeWebp.mockRejectedValue(new Error("encode failed"));
  const f = db();
  const e = env();
  await handle(ctx(f, e));
  expect(f.rpcNamed("apply_region_crops")).toEqual([]);
  expect(statuses(f).at(-1)).toBe("failed");
  expect(e.CONTENT_QUEUE.sendBatch).toHaveBeenCalled();
});

test("a page already done is not cut again but still advances the run", async () => {
  const f = db({ page: page({ crop_status: "done" }) });
  const out = await handle(ctx(f));
  expect(out.detail).toEqual({ skipped: "already done" });
  expect(m.decodeImage).not.toHaveBeenCalled();
  expect(f.rpcNamed("advance_after_crop")).toHaveLength(1);
});

test("a finished run and stale work are left alone", async () => {
  const done = db({ run: { status: "committed" } });
  expect(await handle(ctx(done))).toEqual({ detail: { skipped: "committed" } });
  expect(done.rpcCalls).toEqual([]);
  expect(await handle(ctx(db({ running: stale })))).toEqual({ detail: { skipped: "stale crop work" } });
});

test("a page too large to decode is skipped before it is read", async () => {
  m.pageDimensions.mockReturnValue({ width: 5000, height: 5000, source: "recorded" });
  const f = db();
  const e = env();
  expect((await handle(ctx(f, e))).detail.skipped).toBe("page too large to decode");
  expect(e.DERIVED.get).not.toHaveBeenCalled();
  expect(statuses(f).at(-1)).toBe("skipped");
});

test("an encoded image whose size disagrees with the record is not cut", async () => {
  m.imageDimensions.mockReturnValue({ width: W + 1, height: H });
  const f = db();
  expect((await handle(ctx(f))).detail.skipped).toBe("dimension mismatch");
  expect(m.decodeImage).not.toHaveBeenCalled();
});

test("a mask of the wrong size is refused, and the crops are kept", async () => {
  m.imageDimensions.mockImplementation((bytes: Uint8Array) =>
    new TextDecoder().decode(bytes) === "mask/1" ? { width: 50, height: 50 } : { width: W, height: H });
  const f = db({ page: page({ mask_key: "mask/1" }) });
  const out = await handle(ctx(f));
  expect(out.detail).toMatchObject({ regions: 2, with_mask: 0 });
});

test("a matching mask is cut alongside each crop", async () => {
  const f = db({ page: page({ mask_key: "mask/1" }) });
  const e = env();
  const out = await handle(ctx(f, e));
  expect(out.detail.with_mask).toBe(2);
  expect(e.DERIVED.puts.has("cropmask/run-q1.webp")).toBe(true);
  expect(f.rpcNamed("apply_region_crops")[0].p_rows).toEqual(expect.arrayContaining([
    { id: "q1", crop_key: "crop/run-q1.webp", cropmask_key: "cropmask/run-q1.webp" },
  ]));
});

test("the last page to finish starts reconcile when the advance says so", async () => {
  const e = env();
  await handle(ctx(db({ advance: { advanced: true, enqueue_content: [], enqueue_reconcile: true } }), e));
  expect(e.RECONCILE_QUEUE.send).toHaveBeenCalledWith({ run_id: "run" });
});

test("a page that did not advance the run fans out only on redelivery", async () => {
  const e = env();
  await handle(ctx(db({ advance: { advanced: false, enqueue_content: ["q1"] } }), e));
  expect(e.CONTENT_QUEUE.sendBatch).not.toHaveBeenCalled();
  await handle(ctx(db({ advance: { advanced: false, enqueue_content: ["q1"] } }), e, true));
  expect(e.CONTENT_QUEUE.sendBatch).toHaveBeenCalledTimes(1);
});

test("a permanent failure marks the page failed so the run is not stranded", async () => {
  const f = db();
  await handle.onPermanent(ctx(f), new Error("decoder crashed"));
  expect(statuses(f)).toEqual(["failed"]);
  expect(f.rpcNamed("advance_after_crop")).toHaveLength(1);
});
