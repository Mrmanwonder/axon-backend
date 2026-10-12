/**
 * Durable checkpoints for paid model work (AXO-225).
 *
 * A run that times out is failed and its pages are reset, and the retry is a new
 * run. Nothing used to carry the earlier paid answers across, so the retry paid
 * for them again. These helpers talk to `public.work_*` (see the Axon-Site
 * migration 20261012180000_axo_225_paper_work_journal.sql), which keys a saved
 * answer by WHAT was read (page evidence, prompt, model, versions), never by run id.
 *
 * What this does not promise: exactly-once provider billing. If the provider
 * answers and the worker dies before `saveWork` lands, that answer is lost and
 * the next attempt pays again. The journal counts those as unknown outcomes.
 */
import type { SupabaseClient } from "@supabase/supabase-js";
import type { Env } from "./env.js";
import { RetryableError } from "./errors.js";
import { mustRpc } from "./db.js";

/** Describes one unit of paid work. The caller owns the evidence digest; callModel adds the prompt, model and version parts. */
export interface CheckpointSpec {
  /** Stable name of the unit inside the paper, e.g. "structure:page:3". */
  unitKey: string;
  /** Digest of the evidence the model will see (see `digestHex`). */
  imageDigest: string;
}

export type WorkBegin =
  | { action: "reuse"; work_id: string; status: "output_saved" | "applied"; output: Record<string, unknown>; output_meta: Record<string, unknown> | null; checksum: string }
  | { action: "busy"; work_id: string; retry_after_seconds: number }
  | { action: "run"; work_id: string; lease_token: string; generation: number; attempt: number; unknown_outcomes: number };

/** The journal is off unless the deployment says otherwise, so merging this changes nothing by itself. */
export function journalEnabled(env: Pick<Env, "WORK_JOURNAL">): boolean {
  return env.WORK_JOURNAL === "1";
}

/** Lowercase hex SHA-256 of a string. */
export async function sha256Hex(text: string): Promise<string> {
  const bytes = new TextEncoder().encode(text);
  const hash = await crypto.subtle.digest("SHA-256", bytes);
  return [...new Uint8Array(hash)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

/** Digest of an ordered list of JSON-safe parts. Order matters; `undefined` becomes `null`. */
export function digestHex(parts: readonly unknown[]): Promise<string> {
  return sha256Hex(JSON.stringify(parts, (_k, v) => (v === undefined ? null : v)));
}

/** Names the output schema by its content, so changing the schema can never reuse an answer saved under the old one. */
export async function schemaVersionOf(schema: unknown): Promise<string> {
  return (await sha256Hex(JSON.stringify(schema))).slice(0, 16);
}

export interface WorkIdentity {
  paperId: string;
  studentId: string;
  stage: string;
  unitKey: string;
  inputDigest: string;
  promptDigest: string;
  modelId: string;
  promptVersion: string;
  schemaVersion: string;
  leaseSeconds?: number;
}

export async function beginWork(sb: SupabaseClient, w: WorkIdentity): Promise<WorkBegin> {
  const out = await mustRpc(
    sb.rpc("work_begin", {
      p_paper_id: w.paperId,
      p_student_id: w.studentId,
      p_stage: w.stage,
      p_unit_key: w.unitKey,
      p_input_digest: w.inputDigest,
      p_prompt_digest: w.promptDigest,
      p_model_id: w.modelId,
      p_prompt_version: w.promptVersion,
      p_schema_version: w.schemaVersion,
      p_lease_seconds: w.leaseSeconds ?? 300,
    }),
    "work_begin",
  );
  return out as WorkBegin;
}

/** A live lease holds this work. The queue redelivers later; nothing is paid for meanwhile. */
export function busyError(stage: string, retryAfterSeconds: number): RetryableError {
  return new RetryableError("work_busy", `another worker holds this ${stage} work`, retryAfterSeconds);
}

export type SaveResult =
  | { result: "saved" }
  | { result: "already_saved"; output: Record<string, unknown>; output_meta: Record<string, unknown> | null }
  | { result: "fenced" };

const SAVE_TRIES = 3;

/**
 * Save the paid answer. A valid answer is never thrown away because the journal
 * could not be written: after a few quick tries this reports `null` and the caller
 * carries on with the answer it holds (and logs that the checkpoint is missing).
 */
export async function saveWork(
  sb: SupabaseClient,
  workId: string,
  leaseToken: string,
  output: Record<string, unknown>,
  meta: Record<string, unknown>,
): Promise<SaveResult | null> {
  for (let attempt = 1; attempt <= SAVE_TRIES; attempt++) {
    try {
      const { data, error } = await sb.rpc("work_save_output", {
        p_work_id: workId,
        p_lease_token: leaseToken,
        p_output: output,
        p_output_meta: meta,
      });
      if (!error) return data as SaveResult;
      console.error("work_save_output failed", attempt, error.code ?? "", error.message);
      // A malformed or oversize output will not improve on retry.
      if (error.code && /^(22|23|42|54)/.test(error.code)) return null;
    } catch (cause) {
      console.error("work_save_output threw", attempt, String(cause));
    }
    if (attempt < SAVE_TRIES) await new Promise((r) => setTimeout(r, 150 * attempt));
  }
  return null;
}

/** Best effort: a failed release only means the next attempt waits for the lease to expire. */
export async function releaseWork(sb: SupabaseClient, workId: string, leaseToken: string, errorCode: string): Promise<void> {
  try {
    const { error } = await sb.rpc("work_release", { p_work_id: workId, p_lease_token: leaseToken, p_error_code: errorCode });
    if (error) console.error("work_release failed", error.message);
  } catch (cause) {
    console.error("work_release threw", String(cause));
  }
}

export async function invalidateWork(sb: SupabaseClient, workId: string, reason: string): Promise<void> {
  const { error } = await sb.rpc("work_invalidate", { p_work_id: workId, p_reason: reason });
  if (error) console.error("work_invalidate failed", error.message);
}

/** The downstream write landed. Informational: reuse works from either saved state. */
export async function markWorkApplied(sb: SupabaseClient, workId: string | null | undefined): Promise<void> {
  if (!workId) return;
  try {
    const { error } = await sb.rpc("work_mark_applied", { p_work_id: workId });
    if (error) console.error("work_mark_applied failed", error.message);
  } catch (cause) {
    console.error("work_mark_applied threw", String(cause));
  }
}
