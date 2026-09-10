/**
 * Checked Supabase operations.
 *
 * The pattern this exists to delete:
 *
 *     const { data } = await sb.from("paper_page").select("*").eq(...);
 *     if (!data?.length) return;          // "no pages"
 *
 * `data` is null both when the query succeeded and matched nothing, and when
 * it failed. The two are not the same fact, and collapsing them turns a
 * Supabase blip into "this paper has no pages", which the pipeline then acts
 * on: it advances a stage, marks a page done, or writes an empty result over a
 * real one. The failure is silent and the damage is durable.
 *
 * So every correctness-relevant read and write goes through one of these. They
 * throw a classified error, which the harness routes to retry — an outage
 * delays a paper instead of corrupting it.
 *
 * `[]` still means an empty result. That distinction is the whole product's
 * hard rule 4, one layer down: an admitted gap is recoverable, an invisible
 * one is not.
 */

import { classifyDbError, PermanentError } from "./errors.js";

/** The shape every PostgREST builder resolves to. */
type Result<T> = { data: T; error: unknown };

/** A read or write whose rows the caller needs. Throws on failure. */
export async function mustData<T>(op: PromiseLike<Result<T>>, context: string): Promise<T> {
  const { data, error } = await op;
  if (error) throw classifyDbError(error, context);
  return data;
}

/** A write whose rows the caller does not need, but whose failure matters. */
export async function mustOk(op: PromiseLike<Result<unknown>>, context: string): Promise<void> {
  const { error } = await op;
  if (error) throw classifyDbError(error, context);
}

/**
 * A read of at most one row, where zero rows is legitimate.
 *
 * Use with `.maybeSingle()`. `.single()` reports "no rows" as an error, which
 * puts a real and expected state through the same channel as a failure and
 * forces every caller to special-case PGRST116 — one of which will forget.
 */
export async function mustMaybe<T>(op: PromiseLike<Result<T | null>>, context: string): Promise<T | null> {
  const { data, error } = await op;
  if (error) throw classifyDbError(error, context);
  return data ?? null;
}

/**
 * A read that must return exactly one row.
 *
 * Zero rows is a PermanentError rather than a retry: the row is addressed by
 * an id that came off the queue message, so if it is not there, waiting will
 * not produce it. That is a real state — a paper deleted while its run was in
 * flight — and it should fail the run rather than spin.
 */
export async function mustOne<T>(op: PromiseLike<Result<T | null>>, context: string): Promise<T> {
  const row = await mustMaybe(op, context);
  if (row == null) {
    throw new PermanentError("db_row_missing", `${context}: expected exactly one row, found none`);
  }
  return row;
}

/**
 * An RPC whose return value the caller needs.
 *
 * Named separately from mustData only so the context strings read as RPC
 * names in logs, which is what an operator greps for when a stage stalls.
 */
export async function mustRpc<T>(op: PromiseLike<Result<T>>, context: string): Promise<T> {
  const { data, error } = await op;
  if (error) throw classifyDbError(error, `rpc ${context}`);
  return data;
}

/**
 * A write that must have matched at least one row.
 *
 * An UPDATE matching nothing resolves with no error and an empty array, which
 * every caller reads as success. It is not: it means the row moved, was
 * claimed by another worker, or never existed, and continuing as though the
 * write landed is how a stage advances over work that was never done.
 *
 * Requires the builder to carry `.select(...)`, since PostgREST returns no
 * rows otherwise and there would be nothing to count.
 */
export async function mustAffectRows<T>(
  op: PromiseLike<Result<T[] | null>>,
  context: string,
): Promise<T[]> {
  const rows = await mustData(op, context);
  if (!rows || rows.length === 0) {
    throw new PermanentError("db_no_rows_affected", `${context}: matched no rows`);
  }
  return rows;
}
