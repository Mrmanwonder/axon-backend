/**
 * What kind of failure was that, and may the queue message be acknowledged?
 *
 * The harness used to answer with:
 *
 *     export function isRetryable(error: unknown): boolean {
 *       if (error instanceof ModelError) return error.retryable;
 *       return false;
 *     }
 *
 * which reads as "only the model can fail transiently". Everything else — a
 * Supabase 503, an R2 timeout, a Queue send that did not land, a plain `throw
 * new Error(...)` written immediately below a comment saying this should retry
 * — was classified permanent and the message was acknowledged. A blip in an
 * upstream service became a paper that stopped moving.
 *
 * The default is inverted here, and that is the whole point: an error nobody
 * has classified is treated as infrastructure and retried, because the cost of
 * retrying a permanent failure is a handful of wasted attempts, while the cost
 * of not retrying a transient one is a student's paper stuck forever with no
 * signal. Bounded attempts stop the first cost from growing.
 *
 * Retrying more is only safe once handlers are idempotent. That ordering is
 * deliberate and it is why the claim/lease work travels with this.
 */

/** A failure that will not get better by trying again. */
export class PermanentError extends Error {
  code: string;
  constructor(code: string, message: string) {
    super(message);
    this.name = "PermanentError";
    this.code = code;
  }
}

/** A failure that is expected to clear on its own. */
export class RetryableError extends Error {
  code: string;
  /** Hint for the backoff, where the failure told us how long to wait. */
  retryAfterSeconds?: number;
  constructor(code: string, message: string, retryAfterSeconds?: number) {
    super(message);
    this.name = "RetryableError";
    this.code = code;
    this.retryAfterSeconds = retryAfterSeconds;
  }
}

/**
 * A binding, secret or route is missing or wrong.
 *
 * Neither of the above: retrying cannot fix it, and it is not a data problem
 * the run should be failed for — every message will hit it until someone
 * changes the deployment. It is separated so it can be alerted on loudly
 * rather than buried among per-run failures.
 */
export class ConfigurationError extends Error {
  code: string;
  constructor(code: string, message: string) {
    super(message);
    this.name = "ConfigurationError";
    this.code = code;
  }
}

/** Postgres classes that mean "the statement was wrong", not "the server was busy". */
const PERMANENT_PG_PREFIXES = [
  "22", // data exception — bad input for the type
  "23", // integrity constraint violation
  "42", // syntax error or access rule violation
];

/** Postgres/PostgREST codes that are transient no matter what class they sit in. */
const RETRYABLE_PG_CODES = new Set([
  "40001", // serialization failure
  "40P01", // deadlock detected
  "53300", // too many connections
  "53400", // configuration limit exceeded
  "57014", // query canceled (statement timeout)
  "58030", // io error
  "08000", "08003", "08006", "08001", "08004", // connection exceptions
  "XX000", // internal error — often a transient Supabase-side fault
]);

/**
 * Turn a PostgREST/Postgres error into one of the classes above.
 *
 * A unique-violation is left PERMANENT here on purpose. It is usually the
 * signature of a duplicate delivery, and the caller is the only thing that
 * knows whether the row it collided with represents the work already being
 * done (success) or a genuine clash (failure). Swallowing it centrally would
 * hide both.
 */
export function classifyDbError(error: unknown, context: string): Error {
  const e = (error ?? {}) as { code?: string; message?: string; details?: string; hint?: string };
  const code = e.code ?? "";
  const message = `${context}: ${code ? `[${code}] ` : ""}${e.message ?? String(error)}`;

  if (code === "PGRST301" || code === "PGRST302") {
    // JWT/role problem: a deployment or grant issue, not a busy server.
    return new ConfigurationError("db_unauthorized", message);
  }
  if (RETRYABLE_PG_CODES.has(code)) return new RetryableError("db_transient", message);
  if (code.startsWith("28") || code === "42501") {
    return new ConfigurationError("db_permission", message);
  }
  if (PERMANENT_PG_PREFIXES.some((p) => code.startsWith(p))) {
    return new PermanentError("db_rejected", message);
  }
  // No code at all is the shape of a fetch that never reached Postgres.
  if (!code) return new RetryableError("db_unreachable", message);
  return new RetryableError("db_unclassified", message);
}

/**
 * May this message be retried?
 *
 * Unknown errors answer yes. See the module note: an unclassified failure is
 * far more often a transient one nobody has got round to naming than a
 * permanent one, and the asymmetry of the two mistakes is not close.
 */
export function isRetryable(error: unknown): boolean {
  if (error instanceof PermanentError) return false;
  if (error instanceof ConfigurationError) return false;
  if (error instanceof RetryableError) return true;
  // ModelError carries its own judgement, and it is the one that was already
  // being respected.
  const m = error as { name?: string; retryable?: boolean };
  if (m?.name === "ModelError") return Boolean(m.retryable);
  // A handler that ran out of budget has not proved anything is broken.
  if (m?.name === "HandlerTimeout") return true;
  return true;
}
