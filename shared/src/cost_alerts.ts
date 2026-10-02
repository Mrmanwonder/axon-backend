/**
 * AXO-124 spend alerts. The database detects a breach and writes a cost_alert row
 * (private.detect_cost_alerts, pg_cron). This module turns waiting rows into a message and
 * delivers them to a webhook, then stamps the row. Nothing here is student-derived: a paper id,
 * a date, and numbers.
 */

export interface CostAlertRow {
  id: number;
  kind: "per_paper" | "per_day";
  subject: string;
  observed_usd: number | string;
  threshold_usd: number | string;
  observed_inr: number | string | null;
  unpriced_calls: number;
}

export function formatCostAlert(a: CostAlertRow): string {
  const usd = Number(a.observed_usd).toFixed(2);
  const limit = Number(a.threshold_usd).toFixed(2);
  const inr = a.observed_inr == null ? "" : ` (about Rs ${Number(a.observed_inr).toFixed(0)})`;
  const what = a.kind === "per_paper" ? `Paper ${a.subject.slice(0, 8)}` : `Spend on ${a.subject}`;
  const unpriced =
    a.unpriced_calls > 0
      ? ` ${a.unpriced_calls} call${a.unpriced_calls === 1 ? "" : "s"} had no price, so the real figure is higher.`
      : "";
  return `${what} has cost $${usd}${inr}, over the $${limit} alert threshold.${unpriced}`;
}

export interface AlertSink {
  /** Rows waiting for delivery, oldest first. */
  pending(limit: number): Promise<CostAlertRow[]>;
  /** error null: stamp delivered_at. error set: record it and leave the row pending for the next tick. */
  markDelivered(id: number, error: string | null): Promise<void>;
}

/**
 * Deliver waiting alerts. With no destination configured nothing is marked, so the rows wait,
 * visible in the table, rather than being dropped. Returns how many were delivered.
 */
export async function deliverCostAlerts(
  sink: AlertSink,
  webhookUrl: string | undefined,
  post: (url: string, body: string) => Promise<{ ok: boolean; status: number }>,
  limit = 20,
): Promise<number> {
  if (!webhookUrl) return 0;
  let delivered = 0;
  for (const alert of await sink.pending(limit)) {
    try {
      const res = await post(webhookUrl, JSON.stringify({ text: formatCostAlert(alert) }));
      if (res.ok) {
        await sink.markDelivered(alert.id, null);
        delivered += 1;
      } else {
        await sink.markDelivered(alert.id, `webhook returned ${res.status}`.slice(0, 200));
      }
    } catch (cause) {
      await sink.markDelivered(alert.id, String(cause).slice(0, 200));
    }
  }
  return delivered;
}
