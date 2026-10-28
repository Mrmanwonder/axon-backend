import { beforeEach, expect, test, vi } from "vitest";
import { fakeDb } from "../../../shared/src/__tests__/support/fake-db.js";

const m = vi.hoisted(() => ({
  sb: null as any,
  deleteObject: vi.fn(), deletePrefix: vi.fn(),
  deliverCostAlerts: vi.fn(), runTutorPurges: vi.fn(), queueTopicTagWork: vi.fn(),
  dispatchStageOutbox: vi.fn(), runParentDigests: vi.fn(),
  claimDeletions: vi.fn(), finishDeletion: vi.fn(), maintenanceStore: vi.fn(),
}));
vi.mock("@mastery/shared/http.js", () => ({ serviceClient: () => m.sb }));
vi.mock("@mastery/shared/r2.js", () => ({ deleteObject: m.deleteObject, deletePrefix: m.deletePrefix }));
vi.mock("@mastery/shared/cost_alerts.js", () => ({ deliverCostAlerts: m.deliverCostAlerts }));
vi.mock("@mastery/shared/tutor_purge.js", () => ({ runTutorPurges: m.runTutorPurges }));
vi.mock("@mastery/shared/topic_tag.js", () => ({ queueTopicTagWork: m.queueTopicTagWork }));
vi.mock("@mastery/shared/stage_outbox.js", () => ({ dispatchStageOutbox: m.dispatchStageOutbox }));
vi.mock("@mastery/shared/parent_digest.js", () => ({ runParentDigests: m.runParentDigests, resendMailer: () => ({ send: vi.fn() }) }));
vi.mock("@mastery/shared/maintenance_store.js", () => ({ maintenanceStore: m.maintenanceStore }));

import sweep from "../src/index.js";

const queues = () => ({ STRUCTURE_QUEUE: { sendBatch: vi.fn(async () => {}) }, CONTENT_QUEUE: { sendBatch: vi.fn(async () => {}) }, EXPLAIN_QUEUE: { sendBatch: vi.fn(async () => {}) } });
const tick = (cron: string, env: any = queues()) => sweep.scheduled({ cron } as any, env);

beforeEach(() => {
  for (const f of Object.values(m)) if (typeof f === "function" && "mockReset" in f) (f as any).mockReset();
  m.sb = fakeDb().sb;
  m.maintenanceStore.mockImplementation(() => ({ alerts: {}, tutorPurges: {}, claimDeletions: m.claimDeletions, finishDeletion: m.finishDeletion }));
  m.claimDeletions.mockResolvedValue([]);
  m.dispatchStageOutbox.mockResolvedValue({ delivered: 0, failed: 0 });
  m.runParentDigests.mockResolvedValue({ configured: true, queued: 0, claimed: 0, sent: 0, failed: 0 });
  m.queueTopicTagWork.mockResolvedValue(0);
});

test("the one-minute tick dispatches the stage outbox and nothing else", async () => {
  await tick("* * * * *");
  expect(m.dispatchStageOutbox).toHaveBeenCalledTimes(1);
  expect(m.runParentDigests).not.toHaveBeenCalled();
  expect(m.maintenanceStore).not.toHaveBeenCalled();
  expect(m.deliverCostAlerts).not.toHaveBeenCalled();
});

test("the outbox sends each stage's messages to that stage's queue", async () => {
  const env = queues();
  m.dispatchStageOutbox.mockImplementation(async (deps: any) => {
    await deps.send("structure", [{ run_id: "a" }]);
    await deps.send("content", [{ run_id: "b", region_id: "q" }]);
    return { delivered: 2, failed: 0 };
  });
  await tick("* * * * *", env);
  expect(env.STRUCTURE_QUEUE.sendBatch).toHaveBeenCalledWith([{ body: { run_id: "a" } }]);
  expect(env.CONTENT_QUEUE.sendBatch).toHaveBeenCalledWith([{ body: { run_id: "b", region_id: "q" } }]);
});

test("without the stage queues bound, the outbox is not claimed at all", async () => {
  await tick("* * * * *", { EXPLAIN_QUEUE: { sendBatch: vi.fn(async () => {}) } });
  expect(m.dispatchStageOutbox).not.toHaveBeenCalled();
});

test("the Sunday tick sends parent digests and nothing else, enabled only by the explicit flag", async () => {
  await tick("30 12 * * SUN", { ...queues(), DIGEST_SEND_ENABLED: "true", RESEND_API_KEY: "k", DIGEST_FROM_ADDRESS: "updates@axonstudy.online" });
  expect(m.runParentDigests).toHaveBeenCalledTimes(1);
  expect(m.runParentDigests.mock.calls[0][1]).toMatchObject({ enabled: true });
  expect(m.dispatchStageOutbox).not.toHaveBeenCalled();
  expect(m.maintenanceStore).not.toHaveBeenCalled();

  m.runParentDigests.mockClear();
  await tick("30 12 * * SUN", { ...queues(), DIGEST_SEND_ENABLED: "yes" });
  expect(m.runParentDigests.mock.calls[0][1]).toMatchObject({ enabled: false, mailer: null });
});

test("the fifteen-minute tick runs every maintenance job, outbox included", async () => {
  await tick("*/15 * * * *");
  expect(m.dispatchStageOutbox).toHaveBeenCalledTimes(1);
  expect(m.deliverCostAlerts).toHaveBeenCalledTimes(1);
  expect(m.runTutorPurges).toHaveBeenCalledTimes(1);
  expect(m.queueTopicTagWork).toHaveBeenCalledTimes(1);
  expect(m.claimDeletions).toHaveBeenCalledWith(20);
  expect(m.runParentDigests).not.toHaveBeenCalled();
});

test("one maintenance job failing does not stop the others", async () => {
  m.dispatchStageOutbox.mockRejectedValue(new Error("outbox down"));
  m.deliverCostAlerts.mockRejectedValue(new Error("webhook down"));
  m.runTutorPurges.mockRejectedValue(new Error("intelligence down"));
  m.queueTopicTagWork.mockRejectedValue(new Error("rpc down"));
  await tick("*/15 * * * *");
  expect(m.claimDeletions).toHaveBeenCalledTimes(1);
});

test("topic tags are queued only when the explain queue is bound", async () => {
  const { EXPLAIN_QUEUE: _, ...noExplain } = queues();
  await tick("*/15 * * * *", noExplain);
  expect(m.queueTopicTagWork).not.toHaveBeenCalled();
});

test("deletions: a key is deleted and finished, a long prefix reports progress, a failure is recorded", async () => {
  m.claimDeletions.mockResolvedValue([
    { id: "d1", bucket: "originals", key: "a/b.png" },
    { id: "d2", bucket: "derived", prefix: "student/paper/" },
    { id: "d3", bucket: "derived", key: "broken" },
  ]);
  m.deleteObject.mockImplementation(async (_env: any, _bucket: string, key: string) => { if (key === "broken") throw new Error("R2 503"); });
  m.deletePrefix.mockResolvedValue({ done: false, deleted: 200 });
  await tick("*/15 * * * *");
  expect(m.deleteObject).toHaveBeenCalledWith(expect.anything(), "originals", "a/b.png");
  expect(m.deletePrefix).toHaveBeenCalledWith(expect.anything(), "derived", "student/paper/", { maxKeys: 200 });
  expect(m.finishDeletion.mock.calls).toEqual([
    ["d1"],
    ["d2", "200 deleted, more to go"],
    ["d3", "Error: R2 503"],
  ]);
});

test("a finished prefix walk closes the claim", async () => {
  m.claimDeletions.mockResolvedValue([{ id: "d2", bucket: "derived", prefix: "p/" }]);
  m.deletePrefix.mockResolvedValue({ done: true, deleted: 12 });
  await tick("*/15 * * * *");
  expect(m.finishDeletion.mock.calls).toEqual([["d2"]]);
});
