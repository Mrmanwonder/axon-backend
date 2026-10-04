import { beforeEach, expect, test, vi } from "vitest";
const f = vi.hoisted(() => ({
  runStatus: "content", answer: "original", extractStatus: "pending", gate: null as Promise<void> | null,
  modelStarted: false, empty: false, failRun: vi.fn(), writes: [] as string[],
}));
vi.mock("@mastery/shared/worker.js", () => ({ consumeQueue: (handle: any) => handle, failRun: f.failRun }));
vi.mock("@mastery/shared/r2.js", () => ({ imageRef: async () => ({ url: "fixture", key: "fixture", detail: "high" }) }));
vi.mock("@mastery/shared/page.js", () => ({ pageDimensions: () => ({ width: 1000, height: 2000 }) }));
vi.mock("@mastery/shared/model-client.js", () => ({
  callModel: async () => {
    f.modelStarted = true;
    if (f.gate) await f.gate;
    return { parsed: { student_answer: { value: "stale model", box: { x: 0, y: 0, w: 100, h: 100 }, page_index: 0 },
      recognition_confidence: "high", unreadable: false } };
  },
}));
import content from "../src/index.js";
import reconciliation from "../../reconcile/src/index.js";
function db() {
  return {
    rpc: async (name: string, args: any) => {
      if (name === "pipeline_write") {
        if (args.p_stage === "reconcile_start") {
          if (f.runStatus !== "content") return { data: { applied: false }, error: null };
          f.runStatus = "reconciliation"; return { data: { applied: true }, error: null };
        }
        if (f.runStatus !== "content") return { data: { applied: false }, error: null };
        const patch = args.p_args.patch;
        f.extractStatus = patch.extract_status;
        if (patch.student_answer) { f.answer = patch.student_answer; f.writes.push(f.answer); }
        return { data: { applied: true }, error: null };
      }
      return { data: { advanced: false }, error: null };
    },
    from(table: string) {
      const region = { id: "region", paper_id: "paper", student_id: "student", order_index: 0, question_label: "Q1",
        question_label_box: null, page_spans: [{ page: 1, box: { x: 0, y: 0, w: 100, h: 100 } }],
        extract_status: f.extractStatus, crop_key: null, cropmask_key: null, confidence_signals: {} };
      const b: any = {
        select: () => b, eq: () => b, in: () => b, order: () => b,
        maybeSingle: async () => ({ data: table === "extraction_run" ?
          { id: "run", paper_id: "paper", student_id: "student", status: f.runStatus, route_override: null } :
          table === "paper" ? { reported_total: null, stated_maximum: null } : region, error: null }),
        then: (yes: any) => Promise.resolve({ data: table === "paper_page" ?
          [{ page_number: 1, r2_bucket: "derived", r2_key: "student/paper/page/1", mask_key: null }] :
          f.empty ? [] : [region], error: null }).then(yes),
      };
      return b;
    },
  };
}
beforeEach(() => {
  f.runStatus = "content"; f.answer = "original"; f.extractStatus = "pending";
  f.gate = null; f.modelStarted = false; f.empty = false; f.writes = []; f.failRun.mockClear();
});
test("a stalled content response cannot overwrite an answer corrected after review opens", async () => {
  let release!: () => void;
  f.gate = new Promise<void>(resolve => { release = resolve; });
  const pending = (content.queue as any)({ env: {}, sb: db(), msg: { run_id: "run", region_id: "region" }, attempt: 1, beat: async () => {} });
  await vi.waitFor(() => expect(f.modelStarted).toBe(true));
  f.runStatus = "committed"; f.answer = "Human correction";
  release(); await pending;
  expect(f.answer).toBe("Human correction");
  expect(f.writes).toEqual([]);
});
test("an empty content run reaches the explicit no-questions failure instead of being acknowledged as stale", async () => {
  f.empty = true;
  await (reconciliation.queue as any)({ env: {}, sb: db(), msg: { run_id: "run" }, attempt: 1, beat: async () => {} });
  expect(f.runStatus).toBe("reconciliation");
  expect(f.failRun).toHaveBeenCalledWith(expect.anything(), "run", expect.any(String), "reconcile_no_questions");
});
