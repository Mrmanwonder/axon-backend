import { test } from "node:test";
import assert from "node:assert/strict";
import { FAILURE_CODE_RE, failureCodeFor } from "../failure_codes.js";
import { ConfigurationError, PermanentError, RetryableError, classifyDbError } from "../errors.js";
import { ModelError } from "../model-client.js";

test("model errors map to stage-prefixed stable codes", () => {
  assert.equal(failureCodeFor("explain", new ModelError("bad_shape", "x", 200, true)), "explain_schema_invalid");
  assert.equal(failureCodeFor("explain", new ModelError("empty_response", "x")), "explain_empty_response");
  assert.equal(failureCodeFor("content", new ModelError("timeout", "x")), "content_timeout");
  assert.equal(failureCodeFor("structure", new ModelError("rate_limited", "x")), "structure_rate_limited");
  assert.equal(failureCodeFor("triage", new ModelError("invalid_key", "x")), "triage_provider_auth");
  assert.equal(failureCodeFor("explain", new ModelError("route_disabled", "x")), "explain_model_config");
  assert.equal(failureCodeFor("explain", new ModelError("something_new", "x")), "explain_model_failure");
});

test("database errors map to write_failed / db_config / db_unavailable", () => {
  // 23505 duplicate label (the AXO-116 signature) is a rejected write.
  assert.equal(failureCodeFor("structure", classifyDbError({ code: "23505", message: "dup" }, "ctx")), "structure_write_failed");
  assert.equal(failureCodeFor("structure", classifyDbError({ code: "42703", message: "col" }, "ctx")), "structure_db_config");
  assert.equal(failureCodeFor("reconcile", classifyDbError({ code: "40001", message: "ser" }, "ctx")), "reconcile_db_unavailable");
  assert.equal(failureCodeFor("reconcile", classifyDbError({ message: "fetch failed" }, "ctx")), "reconcile_db_unavailable");
  assert.equal(failureCodeFor("structure", new PermanentError("db_row_missing", "gone")), "structure_row_missing");
});

test("handler timeouts and unknown errors are named, never empty", () => {
  const timeout = Object.assign(new Error("slow"), { name: "HandlerTimeout" });
  assert.equal(failureCodeFor("content", timeout), "content_handler_timeout");
  assert.equal(failureCodeFor("content", new Error("boom")), "content_unexpected");
  assert.equal(failureCodeFor("content", undefined), "content_unexpected");
  assert.equal(failureCodeFor("content", new RetryableError("other", "x")), "content_unexpected");
  assert.equal(failureCodeFor("content", new ConfigurationError("other", "x")), "content_unexpected");
});

test("every produced code satisfies the database CHECK", () => {
  const errors: unknown[] = [
    new ModelError("bad_shape", "x"), new ModelError("weird CODE!", "x"), new Error("x"), undefined,
    classifyDbError({ code: "23505" }, "c"),
  ];
  for (const stage of ["triage", "structure", "crop", "content", "reconcile", "adjudicate", "explain"] as const) {
    for (const e of errors) assert.match(failureCodeFor(stage, e), FAILURE_CODE_RE);
  }
});
