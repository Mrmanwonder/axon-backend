import { test } from "node:test";
import assert from "node:assert/strict";
import { finishSkippedExplanation } from "../explanation_status.js";

function client({ error = null, rows = [{ id: "region" }], rpcError = null }: { error?: unknown; rows?: { id: string }[]; rpcError?: unknown } = {}) {
  const trace: string[] = [];
  let written: Record<string, unknown> | undefined;
  const sb = {
    from: (table: string) => {
      assert.equal(table, "question_region");
      return { update: (patch: Record<string, unknown>) => {
        written = patch;
        return { eq: (column: string, id: string) => {
          assert.equal(column, "id"); assert.equal(id, "region");
          return { select: async () => { trace.push("persist"); return { data: rows, error }; } };
        } };
      } };
    },
    rpc: async (name: string, args: unknown) => {
      trace.push("advance"); assert.equal(name, "advance_after_explain");
      assert.deepEqual(args, { p_run_id: "run" }); return { data: {}, error: rpcError };
    },
  };
  return { sb: sb as any, trace, written: () => written };
}
for (const reason of ["no_marks_lost", "insufficient_evidence"] as const) {
  test(reason + " records a durable absence before advancing, without model prose or marks", async () => {
    const f = client(); await finishSkippedExplanation(f.sb, "region", "run", reason);
    assert.deepEqual(f.trace, ["persist", "advance"]);
    assert.equal(f.written()?.explain_status, "skipped");
    assert.equal(f.written()?.explain_failure_reason, reason);
    assert.ok(Number.isFinite(Date.parse(String(f.written()?.explain_status_at))));
    assert.deepEqual(Object.keys(f.written()!).sort(), ["explain_failure_reason", "explain_status", "explain_status_at"]);
  });
}
test("a failed skip write cannot advance over an unrecorded absence", async () => {
  const f = client({ error: { code: "08006", message: "connection unavailable" } });
  await assert.rejects(finishSkippedExplanation(f.sb, "region", "run", "insufficient_evidence"));
  assert.deepEqual(f.trace, ["persist"]);
});
test("an already removed region cannot be silently treated as a recorded skip", async () => {
  const f = client({ rows: [] });
  await assert.rejects(finishSkippedExplanation(f.sb, "region", "run", "insufficient_evidence"));
  assert.deepEqual(f.trace, ["persist"]);
});
test("advance failure remains a retryable failure rather than a false success", async () => {
  const f = client({ rpcError: { code: "08006", message: "connection unavailable" } });
  await assert.rejects(finishSkippedExplanation(f.sb, "region", "run", "no_marks_lost"));
  assert.deepEqual(f.trace, ["persist", "advance"]);
});
