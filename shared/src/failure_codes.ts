/**
 * Stable machine reasons for terminal failures.
 *
 * `status_reason` is user copy and changes with the wording. A failed run also needs a reason a
 * query can group by and an alert can key on, written at the moment of failure from the error
 * itself (the cause used to be reconstructed from Postgres logs after the fact).
 *
 * Format: `code` or `code:detail`, lower snake case, matching the CHECK on
 * extraction_run.failure_reason and question_region.explain_failure_reason. Codes are
 * `<stage>_<what>`; the sweep writes `sweep_timeout:<stage>` itself in SQL.
 */

export type FailureStage = "triage" | "structure" | "crop" | "content" | "reconcile" | "adjudicate" | "explain";

export const FAILURE_CODE_RE = /^[a-z0-9_]+(:[a-z0-9_]+)?$/;

const MODEL_SUFFIX: Record<string, string> = {
  bad_shape: "schema_invalid",
  empty_response: "empty_response",
  timeout: "timeout",
  network: "network",
  rate_limited: "rate_limited",
  provider_error: "provider_error",
  bad_request: "bad_request",
  invalid_key: "provider_auth",
  forbidden: "provider_auth",
  billing: "provider_auth",
  no_key: "model_config",
  no_route: "model_config",
  route_disabled: "model_config",
  route_lookup_failed: "model_config",
  route_model_mismatch: "model_config",
  served_model_mismatch: "model_config",
  no_compliant_provider: "model_config",
  tool_loop_limit: "tool_loop",
  unexpected_tool_call: "tool_loop",
};

const DB_SUFFIX: Record<string, string> = {
  db_rejected: "write_failed",
  db_row_missing: "row_missing",
  db_schema: "db_config",
  db_permission: "db_config",
  db_unauthorized: "db_config",
  db_transient: "db_unavailable",
  db_unreachable: "db_unavailable",
  db_unclassified: "db_unavailable",
};

/** The machine reason for a terminal failure of `stage`, derived from the error that caused it. */
export function failureCodeFor(stage: FailureStage, error: unknown): string {
  const e = (error ?? {}) as { name?: string; code?: unknown };
  const code = typeof e.code === "string" ? e.code : "";

  let suffix: string;
  if (e.name === "ModelError") suffix = MODEL_SUFFIX[code] ?? "model_failure";
  else if (e.name === "HandlerTimeout") suffix = "handler_timeout";
  else if (DB_SUFFIX[code]) suffix = DB_SUFFIX[code];
  else suffix = "unexpected";

  const out = `${stage}_${suffix}`;
  return FAILURE_CODE_RE.test(out) ? out : `${stage}_unexpected`;
}
