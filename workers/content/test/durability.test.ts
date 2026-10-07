import { describe, expect, test, vi } from "vitest";
import { advanceAndEnqueue } from "../src/index.js";

describe("AXO-176 durable content advancement", () => {
  test("a database error cannot be mistaken for no work left", async () => {
    const send = vi.fn();
    const sb = { rpc: vi.fn().mockResolvedValue({ data: null, error: { code: "08006", message: "connection failed" } }) };
    await expect(advanceAndEnqueue({ RECONCILE_QUEUE: { send } } as any, sb, "run", false)).rejects.toThrow();
    expect(send).not.toHaveBeenCalled();
  });
  test("a resumed durable transition still dispatches the next stage on redelivery", async () => {
    const send = vi.fn().mockResolvedValue(undefined);
    const sb = { rpc: vi.fn().mockResolvedValue({ data: { advanced: false, enqueue_reconcile: true }, error: null }) };
    await advanceAndEnqueue({ RECONCILE_QUEUE: { send } } as any, sb, "run", true);
    expect(send).toHaveBeenCalledWith({ run_id: "run" });
  });
  test("AXO-211: a concurrent peer that did not advance the run does not queue reconciliation again", async () => {
    const send = vi.fn().mockResolvedValue(undefined);
    const sb = { rpc: vi.fn().mockResolvedValue({ data: { advanced: false, enqueue_reconcile: true }, error: null }) };
    await advanceAndEnqueue({ RECONCILE_QUEUE: { send } } as any, sb, "run", false);
    expect(send).not.toHaveBeenCalled();
  });
  test("AXO-211: the caller that advanced the run queues reconciliation", async () => {
    const send = vi.fn().mockResolvedValue(undefined);
    const sb = { rpc: vi.fn().mockResolvedValue({ data: { advanced: true, enqueue_reconcile: true }, error: null }) };
    await advanceAndEnqueue({ RECONCILE_QUEUE: { send } } as any, sb, "run", false);
    expect(send).toHaveBeenCalledTimes(1);
  });
  test("a missing queue binding fails visibly instead of acknowledging undelivered work", async () => {
    const sb = { rpc: vi.fn().mockResolvedValue({ data: { enqueue_reconcile: true }, error: null }) };
    await expect(advanceAndEnqueue({} as any, sb, "run", true)).rejects.toThrow("queue");
  });
});
