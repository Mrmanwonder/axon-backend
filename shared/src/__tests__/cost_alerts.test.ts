import { test } from "node:test";
import assert from "node:assert/strict";
import { deliverCostAlerts, formatCostAlert, type AlertSink, type CostAlertRow } from "../cost_alerts.js";

const paper: CostAlertRow = {
  id: 1, kind: "per_paper", subject: "cccccccc-0000-4000-8000-000000000001",
  observed_usd: "0.6", threshold_usd: "0.5", observed_inr: "57.6", unpriced_calls: 0,
};

function sink(rows: CostAlertRow[]) {
  const marks: Array<[number, string | null]> = [];
  const s: AlertSink = {
    pending: async () => rows,
    markDelivered: async (id, error) => { marks.push([id, error]); },
  };
  return { s, marks };
}

test("a paper alert states the cost in USD and INR against the threshold", () => {
  assert.equal(
    formatCostAlert(paper),
    "Paper cccccccc has cost $0.60 (about Rs 58), over the $0.50 alert threshold.",
  );
});

test("an alert with unpriced calls says the real figure is higher", () => {
  const msg = formatCostAlert({ ...paper, kind: "per_day", subject: "2026-10-02", unpriced_calls: 2 });
  assert.match(msg, /^Spend on 2026-10-02 has cost/);
  assert.match(msg, /2 calls had no price, so the real figure is higher\.$/);
});

test("with no destination configured nothing is delivered or marked", async () => {
  const { s, marks } = sink([paper]);
  assert.equal(await deliverCostAlerts(s, undefined, async () => ({ ok: true, status: 200 })), 0);
  assert.deepEqual(marks, []);
});

test("a delivered alert is stamped; a failed one records why and stays pending", async () => {
  const ok = sink([paper]);
  assert.equal(await deliverCostAlerts(ok.s, "https://example.test/hook", async () => ({ ok: true, status: 200 })), 1);
  assert.deepEqual(ok.marks, [[1, null]]);

  const bad = sink([paper]);
  assert.equal(await deliverCostAlerts(bad.s, "https://example.test/hook", async () => ({ ok: false, status: 500 })), 0);
  assert.deepEqual(bad.marks, [[1, "webhook returned 500"]]);

  const thrown = sink([paper]);
  assert.equal(await deliverCostAlerts(thrown.s, "https://example.test/hook", async () => { throw new Error("offline"); }), 0);
  assert.match(String(thrown.marks[0][1]), /offline/);
});
