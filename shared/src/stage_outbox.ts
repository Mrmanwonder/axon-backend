/**
 * Dispatcher for `private.stage_outbox` (AXO-225, Axon-Site migration
 * 20261012190000_axo_225_task_level_recovery.sql).
 *
 * The database writes an outbox row in the same transaction that puts a stalled
 * page or region back to pending. This sends those rows to the stage queues and
 * marks them delivered. A message is only a wake-up: the structure and content
 * workers re-read the page or region and skip anything already done.
 *
 * Delivery is at-least-once. If the send lands and the `finish` write does not,
 * the row is claimed again after two minutes and sent again, which the workers
 * already tolerate (queue delivery is at-least-once anyway).
 */

export interface OutboxEvent {
  event_id: string;
  stage: "structure" | "content";
  run_id: string;
  unit_id: string;
}

export type StageMessage =
  | { run_id: string; page_id: string }
  | { run_id: string; region_id: string };

export interface OutboxDeps {
  claim(limit: number): Promise<OutboxEvent[]>;
  finish(delivered: string[], failed: string[], error: string | null): Promise<void>;
  send(stage: OutboxEvent["stage"], messages: StageMessage[]): Promise<void>;
}

export function messageFor(event: OutboxEvent): StageMessage {
  return event.stage === "structure"
    ? { run_id: event.run_id, page_id: event.unit_id }
    : { run_id: event.run_id, region_id: event.unit_id };
}

/** Claims, sends per stage, and reports each stage's outcome separately. Returns how many were delivered. */
export async function dispatchStageOutbox(deps: OutboxDeps, limit = 50): Promise<{ delivered: number; failed: number }> {
  const events = await deps.claim(limit);
  if (!events.length) return { delivered: 0, failed: 0 };

  let delivered = 0;
  let failed = 0;
  for (const stage of ["structure", "content"] as const) {
    const batch = events.filter((e) => e.stage === stage);
    if (!batch.length) continue;
    const ids = batch.map((e) => e.event_id);
    try {
      await deps.send(stage, batch.map(messageFor));
    } catch (cause) {
      failed += ids.length;
      await deps.finish([], ids, String(cause).slice(0, 300));
      continue;
    }
    delivered += ids.length;
    await deps.finish(ids, [], null);
  }
  return { delivered, failed };
}
