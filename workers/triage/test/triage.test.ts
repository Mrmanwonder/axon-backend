import { beforeEach, expect, test, vi } from "vitest";
import { applied, fakeDb, stale } from "../../../shared/src/__tests__/support/fake-db.js";
import { REJECTION_REASON } from "@mastery/shared/prompts/triage.v1.js";

const m = vi.hoisted(() => ({
  callModel: vi.fn(), imageRef: vi.fn(), failRun: vi.fn(),
  cambridgeSchemeRef: vi.fn(), locateScheme: vi.fn(), resolveAssessmentIdentity: vi.fn(),
}));
vi.mock("@mastery/shared/worker.js", async (orig) => ({
  ...await orig<any>(),
  consumeQueue: (handle: any, onPermanent: any) => Object.assign(handle, { onPermanent }),
  failRun: m.failRun,
}));
vi.mock("@mastery/shared/model-client.js", async (orig) => ({ ...await orig<any>(), callModel: m.callModel }));
vi.mock("@mastery/shared/r2.js", () => ({ imageRef: m.imageRef }));
vi.mock("@mastery/shared/cambridge_scheme.js", () => ({ cambridgeSchemeRef: m.cambridgeSchemeRef, locateScheme: m.locateScheme }));
vi.mock("@mastery/shared/assessment.js", () => ({ resolveAssessmentIdentity: m.resolveAssessmentIdentity }));

import worker from "../src/index.js";
const handle = worker.queue as any;

const run = { id: "run", paper_id: "paper", student_id: "student", status: "queued", route_override: null };
const page = (n: number, over: any = {}) => ({
  id: `p${n}`, page_number: n, r2_bucket: "derived", r2_key: `page/${n}`, thumb_key: `thumb/${n}`,
  quality_verdict: "ok", quality_signals: null, ...over,
});
const graded = { classification: "graded_exam", confidence: "high", ink_colour: "red", assessment_identity: null };

function db(over: { run?: any; pages?: any[]; triage?: any } = {}) {
  const pages = over.pages ?? [page(1), page(2)];
  return fakeDb({
    extraction_run: over.run === undefined ? run : over.run,
    paper_page: pages,
  }, (name, args) => {
    if (name !== "pipeline_write") return null;
    return args.p_stage === "triage" ? over.triage ?? applied : applied;
  });
}
const queue = () => ({ sendBatch: vi.fn(async () => {}) });
const ctx = (f: ReturnType<typeof fakeDb>, env: any = { STRUCTURE_QUEUE: queue() }) =>
  ({ env, sb: f.sb, signal: new AbortController().signal, msg: { run_id: "run" }, attempt: 1, beat: async () => {} });
const stages = (f: ReturnType<typeof fakeDb>) => f.rpcNamed("pipeline_write").map((a) => a.p_stage);
const stage = (f: ReturnType<typeof fakeDb>, s: string) => f.rpcNamed("pipeline_write").find((a) => a.p_stage === s);

beforeEach(() => {
  for (const fn of Object.values(m)) fn.mockReset();
  m.imageRef.mockImplementation(async (_e: any, bucket: string, key: string, detail: string) => ({ url: `${bucket}/${key}`, detail }));
  m.callModel.mockResolvedValue({ parsed: graded });
  m.cambridgeSchemeRef.mockReturnValue(null);
  m.locateScheme.mockResolvedValue([]);
  m.resolveAssessmentIdentity.mockResolvedValue(null);
});

test("a run already at structure is re-dispatched to the structure queue", async () => {
  const env = { STRUCTURE_QUEUE: queue() };
  expect(await handle(ctx(db({ run: { ...run, status: "structure" } }), env))).toEqual({ detail: { resumed: "structure dispatch" } });
  expect(env.STRUCTURE_QUEUE.sendBatch).toHaveBeenCalledWith([
    { body: { run_id: "run", page_id: "p1" } }, { body: { run_id: "run", page_id: "p2" } },
  ]);
  expect(m.callModel).not.toHaveBeenCalled();
});

test("a retry after the run reached triaging is not skipped", async () => {
  await handle(ctx(db({ run: { ...run, status: "triaging" } })));
  expect(m.callModel).toHaveBeenCalledTimes(1);
});

test("any other status is skipped", async () => {
  expect(await handle(ctx(db({ run: { ...run, status: "content" } })))).toEqual({ detail: { skipped: "content" } });
});

test("no stored pages fails the run visibly", async () => {
  expect(await handle(ctx(db({ pages: [] })))).toEqual({ detail: { failed: "no pages" } });
  expect(m.failRun).toHaveBeenCalledWith(expect.anything(), "run", expect.stringMatching(/could not find the pages/), "triage_pages_missing");
});

test("when every sampled page failed the quality check, the paper is refused without a model call", async () => {
  const f = db({ pages: [page(1, { quality_verdict: "fail" }), page(2, { quality_verdict: "fail" })] });
  expect(await handle(ctx(f))).toEqual({ detail: { rejected: "quality", pages: 2 } });
  expect(stages(f)).toEqual(["triage_reject"]);
  expect(m.callModel).not.toHaveBeenCalled();
});

test("a long booklet is sampled across its whole length, on thumbnails, with the full page as fallback", async () => {
  const pages = Array.from({ length: 12 }, (_, i) => page(i + 1, i === 11 ? { thumb_key: null } : {}));
  await handle(ctx(db({ pages })));
  const { images } = m.callModel.mock.calls[0][0];
  expect(images).toHaveLength(6);
  expect(images[0]).toEqual({ url: "derived/thumb/1", detail: "low" });
  expect(images.at(-1)).toEqual({ url: "derived/page/12", detail: "low" });
});

test("a refusal as unmarked is looked at again at full detail, and the second answer stands", async () => {
  m.callModel.mockResolvedValueOnce({ parsed: { ...graded, classification: "ungraded_paper" } }).mockResolvedValueOnce({ parsed: graded });
  const f = db();
  const out = await handle(ctx(f));
  expect(m.callModel).toHaveBeenCalledTimes(2);
  expect(m.callModel.mock.calls[1][0].images.every((i: any) => i.detail === "high" && i.url.startsWith("derived/page/"))).toBe(true);
  expect(out.detail).toMatchObject({ classification: "graded_exam", second_look: true });
  expect(stages(f)).toEqual(["triage_start", "triage"]);
});

test("the student's word that a paper is marked overrides a refusal as unmarked", async () => {
  m.callModel.mockResolvedValue({ parsed: { ...graded, classification: "blank_paper" } });
  const out = await handle(ctx(db({ run: { ...run, route_override: { student_says_marked: true } } })));
  expect(out.detail).toMatchObject({ classification: "graded_exam", student_says_marked: true });
});

test("a paper that is not a marked exam is refused with the reason for its classification", async () => {
  m.callModel.mockResolvedValue({ parsed: { ...graded, classification: "not_schoolwork", confidence: "high" } });
  const f = db();
  expect(await handle(ctx(f))).toEqual({ detail: { rejected: "not_schoolwork", second_look: false } });
  expect(stage(f, "triage_reject").p_args.reason).toBe(REJECTION_REASON.not_schoolwork);
});

test("a marked paper is written and every stored page is sent to structure", async () => {
  const env = { STRUCTURE_QUEUE: queue() };
  const f = db();
  const out = await handle(ctx(f, env));
  expect(stages(f)).toEqual(["triage_start", "triage"]);
  expect(stage(f, "triage").p_args.tier_routing.triage).toEqual(graded);
  expect(env.STRUCTURE_QUEUE.sendBatch).toHaveBeenCalledTimes(1);
  expect(out.detail).toMatchObject({ classification: "graded_exam", pages: 2, looked_at: 2, on_thumbnails: 2 });
});

test("an unmarked Cambridge paper with a located scheme is checked against it, filename only", async () => {
  m.callModel.mockResolvedValue({ parsed: { ...graded, classification: "ungraded_paper", assessment_identity: { code: "9231/11/O/N/25" } } });
  // No thumbnails, so there is no second look to take.
  const pages = [page(1, { thumb_key: null })];
  m.cambridgeSchemeRef.mockReturnValue({ label: "9231/11/O/N/25", filename: "9231_w25_ms_11.pdf" });
  m.locateScheme.mockResolvedValue(["https://example.org/9231_w25_ms_11.pdf"]);
  const f = db({ pages });
  const out = await handle(ctx(f));
  expect(out.detail.scheme_check).toBe("9231_w25_ms_11.pdf");
  const check = f.writes.find((w) => w.table === "paper_check");
  expect(check?.op).toBe("upsert");
  expect(check?.payload).toMatchObject({ scheme_ref: "9231_w25_ms_11.pdf", status: "queued" });
  expect(JSON.stringify(check?.payload)).not.toContain("example.org");
});

test("an unmarked Cambridge paper with no scheme copy is refused as usual", async () => {
  m.callModel.mockResolvedValue({ parsed: { ...graded, classification: "ungraded_paper" } });
  m.cambridgeSchemeRef.mockReturnValue({ label: "x", filename: "x.pdf" });
  const f = db({ pages: [page(1, { thumb_key: null })] });
  expect((await handle(ctx(f))).detail.rejected).toBe("ungraded_paper");
  expect(f.writes.find((w) => w.table === "paper_check")).toBeUndefined();
});

test("a failed identity lookup does not lose the scan", async () => {
  m.callModel.mockResolvedValue({ parsed: { ...graded, assessment_identity: { code: "0580/42/M/J/23" } } });
  m.resolveAssessmentIdentity.mockRejectedValue(new Error("db timeout"));
  const f = db();
  const out = await handle(ctx(f));
  expect(out.detail.assessment_identity).toBeNull();
  expect(stage(f, "triage").p_args.tier_routing.assessment_identity_status).toBe("unresolved");
});

test("a stale triage result does not dispatch structure", async () => {
  const env = { STRUCTURE_QUEUE: queue() };
  expect(await handle(ctx(db({ triage: stale }), env))).toEqual({ detail: { skipped: "stale triage result" } });
  expect(env.STRUCTURE_QUEUE.sendBatch).not.toHaveBeenCalled();
});

test("a permanent failure says nothing was lost", async () => {
  await handle.onPermanent(ctx(db()), new Error("model gave up"));
  expect(m.failRun).toHaveBeenCalledWith(expect.anything(), "run", expect.stringMatching(/Nothing was lost/), expect.any(String));
});
