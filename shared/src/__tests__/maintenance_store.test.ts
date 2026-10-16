import { test } from "node:test";
import assert from "node:assert/strict";
import type { SupabaseClient } from "@supabase/supabase-js";
import { maintenanceStore } from "../maintenance_store.js";
import { deliverCostAlerts, type CostAlertRow } from "../cost_alerts.js";

function client(result: { data: unknown; error: unknown }) {
  const calls: Array<{ name: string; args: unknown }> = [];
  const sb = {
    rpc(name: string, args: unknown) {
      calls.push({ name, args });
      return Promise.resolve(result);
    },
    from(name: string) {
      return {
        update(values: unknown) {
          calls.push({ name, args: values });
          return {
            eq(column: string, value: unknown) {
              calls.push({ name: "eq", args: { column, value } });
              return {
                select(columns: string) {
                  calls.push({ name: "select", args: columns });
                  return Promise.resolve(result);
                },
              };
            },
          };
        },
      };
    },
  } as unknown as SupabaseClient;
  return { store: maintenanceStore(sb), calls };
}

const failures: Array<[string, (s: ReturnType<typeof maintenanceStore>) => Promise<unknown>]> = [
  ["alert reads", (s) => s.alerts.pending(20)],
  ["alert delivery receipt", (s) => s.alerts.markDelivered(1, null)],
  ["alert failure receipt", (s) => s.alerts.markDelivered(1, "webhook returned 503")],
  ["tutor claims", (s) => s.tutorPurges.claim(20)],
  ["tutor completion", (s) => s.tutorPurges.finish(1, null)],
  ["tutor failure receipt", (s) => s.tutorPurges.finish(1, "unavailable")],
  ["deletion claims", (s) => s.claimDeletions(20)],
  ["deletion completion", (s) => s.finishDeletion("1")],
  ["deletion failure receipt", (s) => s.finishDeletion("1", "unavailable")],
];
for (const [name, run] of failures) {
  test(name + " rejects a database failure", async () => {
    const { store } = client({ data: null, error: { message: "temporary database outage", code: "08006" } });
    await assert.rejects(run(store), /temporary database outage/);
  });
}

test("a successful empty queue is distinct from a missing response", async () => {
  const empty = client({ data: [], error: null }).store;
  assert.deepEqual(await empty.alerts.pending(20), []);
  assert.deepEqual(await empty.tutorPurges.claim(20), []);
  assert.deepEqual(await empty.claimDeletions(20), []);
  const missing = client({ data: null, error: null }).store;
  await assert.rejects(missing.alerts.pending(20), /no result returned/);
  await assert.rejects(missing.tutorPurges.claim(20), /no result returned/);
  await assert.rejects(missing.claimDeletions(20), /no result returned/);
});

test("delivery cannot succeed if its receipt matched no row", async () => {
  const { store } = client({ data: [], error: null });
  await assert.rejects(store.alerts.markDelivered(1, null), /matched no rows/);
});

test("confirmed delivery writes a scoped receipt and requests matched row ids", async () => {
  const { store, calls } = client({ data: [{ id: 7 }], error: null });
  await store.alerts.markDelivered(7, null);
  assert.equal(calls[0].name, "cost_alert");
  const values = calls[0].args as { delivered_at: string; delivery_error: null };
  assert.ok(Number.isFinite(Date.parse(values.delivered_at)));
  assert.equal(values.delivery_error, null);
  assert.deepEqual(calls[1], { name: "eq", args: { column: "id", value: 7 } });
  assert.deepEqual(calls[2], { name: "select", args: "id" });
});

test("failed delivery records only its error, never a delivered timestamp", async () => {
  const { store, calls } = client({ data: [{ id: 7 }], error: null });
  await store.alerts.markDelivered(7, "webhook returned 503");
  assert.deepEqual(calls[0].args, { delivery_error: "webhook returned 503" });
});

test("void completion RPCs may return null successfully", async () => {
  const { store, calls } = client({ data: null, error: null });
  await store.tutorPurges.finish(7, null);
  await store.finishDeletion("8");
  assert.deepEqual(calls, [
    { name: "finish_tutor_purge", args: { p_id: 7, p_error: null } },
    { name: "finish_deletion", args: { p_id: "8", p_error: null } },
  ]);
});

test("webhook success followed by database outage is not reported as delivered", async () => {
  const { store } = client({ data: null, error: { message: "receipt unavailable", code: "08006" } });
  const alert: CostAlertRow = { id: 1, kind: "per_day", subject: "2026-10-10",
    observed_usd: 2, threshold_usd: 1, observed_inr: null, unpriced_calls: 0 };
  let posts = 0;
  await assert.rejects(deliverCostAlerts(
    { pending: async () => [alert], markDelivered: store.alerts.markDelivered },
    "https://example.test/hook",
    async () => { posts++; return { ok: true, status: 200 }; },
  ), /receipt unavailable/);
  assert.equal(posts, 1);
});
