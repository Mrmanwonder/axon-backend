/**
 * AXO-211/224: the structure stage processes all paper pages under a bounded
 * per-invocation concurrency limit. Production batches contain at most two
 * messages; this harness sends thirteen to prove that its worker lanes stay
 * bounded without losing the stage transition or writes.
 *
 * Driven through the real batch harness (`consumeBatch`) and the real
 * planning code, against a fake database that enforces what production
 * enforces: unique (run_id, order_index), the numbered-label index, the paper
 * lock around `pipeline_write`, and the run lock plus re-issue branch of
 * `advance_after_structure`.
 */
import { beforeEach, expect, test, vi } from "vitest";

const f = vi.hoisted(() => ({
  pages: 13,
  gate: null as Promise<void> | null,
  modelCalls: 0,
  modelInFlight: 0,
  modelPeak: 0,
  /** Page numbers whose next done-write collides with a row another invocation just wrote. */
  foreignRace: new Set<number>(),
  conflicts: 0,
}));

vi.mock("@mastery/shared/worker.js", async (importOriginal) => {
  const actual = await importOriginal<any>();
  return { ...actual, consumeQueue: (handle: any, onPermanent: any, options: any) => ({ handle, onPermanent, options }) };
});
vi.mock("@mastery/shared/structure-page.js", () => ({
  loadStructurePage: async (_sb: any, pageId: string) => {
    const n = Number(pageId.replace("page-", ""));
    return { id: pageId, paper_id: "paper", student_id: "student", page_number: n, r2_bucket: "derived",
      r2_key: `student/paper/page/${n}`, mask_key: null, structure_status: db.pageStatus.get(n), teacher_marks: [] };
  },
}));
vi.mock("@mastery/shared/r2.js", () => ({ imageRef: async () => ({ url: "fixture", key: "fixture", detail: "high" }) }));
vi.mock("@mastery/shared/page.js", () => ({ pageDimensions: () => ({ width: 1000, height: 2000 }), UNPLACEABLE_PAGE_REASON: "no dimensions" }));
vi.mock("@mastery/shared/model-client.js", async (importOriginal) => ({
  ...await importOriginal<any>(),
  callModel: async ({ instruction }: { instruction: string }) => {
    f.modelCalls++;
    f.modelInFlight++;
    f.modelPeak = Math.max(f.modelPeak, f.modelInFlight);
    if (f.gate) await f.gate;
    f.modelInFlight--;
    void instruction;
    return { parsed: { is_graded_exam_paper: true, not_a_paper_reason: null, reported_total: null, stated_maximum: null,
      regions: [
        { candidate_number: null, number_box: null, box: { x: 50, y: 50, w: 900, h: 400 }, continues_from_previous: false, structure_confidence: "high" },
        { candidate_number: null, number_box: null, box: { x: 50, y: 500, w: 900, h: 400 }, continues_from_previous: false, structure_confidence: "high" },
      ] } };
  },
}));

import structure from "../../structure/src/index.js";
import { consumeBatch } from "@mastery/shared/worker.js";
import { KeyedMutex } from "@mastery/shared/keyed_mutex.js";

type Row = { id: string; run_id: string; order_index: number; page_spans: any[]; question_label: string | null; extract_status: string };

const db = {
  status: "structure",
  pageStatus: new Map<number, string>(),
  regions: [] as Row[],
  advances: 0,
  paperLock: new KeyedMutex(),
  runLock: new KeyedMutex(),
};
const tick = () => new Promise((r) => setTimeout(r, 0));

function sb() {
  return {
    rpc: async (name: string, args: any) => {
      if (name === "run_heartbeat") return { data: null, error: null };
      if (name === "pipeline_write") {
        return db.paperLock.run("paper", async () => {
          await tick();
          if (db.status !== "structure") return { data: { applied: false }, error: null };
          const n = Number(String(args.p_args.page_id).replace("page-", ""));
          const patch = args.p_args.patch;
          if (["done", "failed", "unreadable"].includes(db.pageStatus.get(n) ?? "")) return { data: { applied: false }, error: null };
          if (patch.structure_status === "done" && args.p_args.regions) {
            if (f.foreignRace.delete(n)) {
              // Another invocation's page landed between this page's read and its write.
              const taken = Math.min(...args.p_args.regions.map((r: Row) => r.order_index));
              db.regions.push({ id: `foreign-${n}`, run_id: "run", order_index: taken, page_spans: [{ page: 100 + n }], question_label: null, extract_status: "pending" });
            }
            const kept = db.regions.filter((r) => Number(r.page_spans[0]?.page) !== n);
            const incoming: Row[] = args.p_args.regions.map((r: any) => ({ ...r, run_id: "run", extract_status: "pending" }));
            const indexes = new Set(kept.map((r) => r.order_index));
            if (incoming.some((r) => indexes.has(r.order_index)) || new Set(incoming.map((r) => r.order_index)).size !== incoming.length) {
              f.conflicts++;
              return { data: null, error: { code: "23505", message: "duplicate key value violates unique constraint \"question_region_run_id_order_index_key\"" } };
            }
            db.regions = [...kept, ...incoming];
          }
          db.pageStatus.set(n, patch.structure_status);
          return { data: { applied: true }, error: null };
        });
      }
      if (name === "advance_after_structure") {
        return db.runLock.run("run", async () => {
          await tick();
          const pending = [...db.pageStatus.values()].filter((s) => s === "pending" || s === "running").length;
          if (pending > 0) return { data: { advanced: false }, error: null };
          const ids = [...db.regions].sort((a, b) => a.order_index - b.order_index).map((r) => r.id);
          if (db.status === "content") return { data: { advanced: false, enqueue_content: ids, enqueue_reconcile: ids.length === 0 }, error: null };
          if (db.status !== "structure") return { data: { advanced: false }, error: null };
          db.status = "content";
          db.advances++;
          return { data: { advanced: true, enqueue_content: ids, enqueue_reconcile: false }, error: null };
        });
      }
      throw new Error("unexpected rpc " + name);
    },
    from(table: string) {
      const b: any = {
        select: () => b, eq: () => b, in: () => b, not: () => b, order: () => b,
        maybeSingle: async () => ({ data: table === "extraction_run" ? { status: db.status, route_override: null } : null, error: null }),
        then: (yes: any, no: any) => {
          const result = table === "paper_page" ? { data: null, error: null, count: f.pages }
            : table === "question_region" ? { data: db.regions.map((r) => ({ ...r })), error: null }
            : { data: [], error: null };
          return Promise.resolve(result).then(yes, no);
        },
      };
      return b;
    },
  };
}

function message(n: number, attempts = 1) {
  const m = { body: { run_id: "run", page_id: `page-${n}` }, attempts, acked: 0, retried: 0,
    ack: () => { m.acked++; }, retry: () => { m.retried++; } };
  return m;
}

const worker = structure.queue as unknown as { handle: any; onPermanent: any; options: { concurrency: number } };

beforeEach(() => {
  f.pages = 13; f.gate = null; f.modelCalls = 0; f.modelInFlight = 0; f.modelPeak = 0; f.foreignRace = new Set(); f.conflicts = 0;
  db.status = "structure"; db.regions = []; db.advances = 0;
  db.pageStatus = new Map(Array.from({ length: f.pages }, (_, i) => [i + 1, "pending"]));
});

test("a 13-page workload respects bounded concurrency, lands every page and advances once", async () => {
  let release!: () => void;
  f.gate = new Promise<void>((r) => { release = r; });
  const sendBatch = vi.fn().mockResolvedValue(undefined);
  const env: any = { CONTENT_QUEUE: { sendBatch }, RECONCILE_QUEUE: { send: vi.fn() } };
  const messages = Array.from({ length: 13 }, (_, i) => message(i + 1));

  expect(worker.options.concurrency).toBe(2);
  const run = consumeBatch(messages as any, sb() as any, env, worker.handle, worker.onPermanent, worker.options);
  await vi.waitFor(() => expect(f.modelInFlight).toBe(2));
  expect(f.modelPeak).toBe(2); // never trust a 13-wide isolated Worker invocation
  release();
  await run;

  expect(messages.every((m) => m.acked === 1 && m.retried === 0)).toBe(true);
  expect(f.modelCalls).toBe(13);
  expect([...db.pageStatus.values()].every((s) => s === "done")).toBe(true);
  // 26 regions, each with its own order_index.
  expect(db.regions).toHaveLength(26);
  expect(new Set(db.regions.map((r) => r.order_index)).size).toBe(26);
  // Pages in one isolate take turns for the write: nothing collided.
  expect(f.conflicts).toBe(0);
  // The stage advanced once and the content stage was queued once, with every region.
  expect(db.advances).toBe(1);
  expect(sendBatch).toHaveBeenCalledTimes(1);
  expect(sendBatch.mock.calls[0][0]).toHaveLength(26);
});

test("a page that loses the index to another invocation re-plans from the same answer, without a second model call", async () => {
  f.pages = 3;
  db.pageStatus = new Map([[1, "pending"], [2, "pending"], [3, "pending"]]);
  f.foreignRace = new Set([2]);
  const sendBatch = vi.fn().mockResolvedValue(undefined);
  const messages = [1, 2, 3].map((n) => message(n));
  await consumeBatch(messages as any, sb() as any, { CONTENT_QUEUE: { sendBatch } } as any, worker.handle, worker.onPermanent, worker.options);

  expect(f.conflicts).toBe(1);
  expect(f.modelCalls).toBe(3);
  expect(messages.every((m) => m.acked === 1 && m.retried === 0)).toBe(true);
  expect([...db.pageStatus.values()]).toEqual(["done", "done", "done"]);
  expect(new Set(db.regions.map((r) => r.order_index)).size).toBe(db.regions.length);
  expect(db.advances).toBe(1);
  expect(sendBatch).toHaveBeenCalledTimes(1);
});

test("a late duplicate of a finished page does not queue the content stage again; its redelivery would", async () => {
  f.pages = 2;
  db.pageStatus = new Map([[1, "pending"], [2, "pending"]]);
  const sendBatch = vi.fn().mockResolvedValue(undefined);
  const env: any = { CONTENT_QUEUE: { sendBatch } };
  await consumeBatch([message(1), message(2)] as any, sb() as any, env, worker.handle, worker.onPermanent, worker.options);
  expect(sendBatch).toHaveBeenCalledTimes(1);

  // First delivery of a duplicate: the run has moved on, nothing is resent.
  await consumeBatch([message(2)] as any, sb() as any, env, worker.handle, worker.onPermanent, worker.options);
  expect(sendBatch).toHaveBeenCalledTimes(1);
  // A redelivery is the case the re-issued list exists for.
  await consumeBatch([message(2, 2)] as any, sb() as any, env, worker.handle, worker.onPermanent, worker.options);
  expect(sendBatch).toHaveBeenCalledTimes(2);
  expect(db.advances).toBe(1);
});
