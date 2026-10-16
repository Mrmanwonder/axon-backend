import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";
import ts from "typescript";

// Execute the shipping handler with inert boundary adapters. No database,
// queues, mail or storage is contacted by these routing tests.
const source = fs.readFileSync(new URL("../../workers/sweep/src/index.ts", import.meta.url), "utf8");
const compiled = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
}).outputText;

function fixture({ failDispatch = false, boundQueues = true } = {}) {
  const calls = [];
  const adapters = {
    "@mastery/shared/http.js": { serviceClient: () => ({}) },
    "@mastery/shared/r2.js": {},
    "@mastery/shared/cost_alerts.js": { deliverCostAlerts: async () => { calls.push("alerts"); } },
    "@mastery/shared/tutor_purge.js": { runTutorPurges: async () => { calls.push("purges"); } },
    "@mastery/shared/maintenance_store.js": { maintenanceStore: () => ({
      alerts: {}, tutorPurges: {},
      claimDeletions: async () => { calls.push("deletions"); return []; },
    }) },
    "@mastery/shared/topic_tag.js": {},
    "@mastery/shared/chunked_send.js": {},
    "@mastery/shared/stage_outbox.js": { dispatchStageOutbox: async () => {
      calls.push("outbox");
      if (failDispatch) throw new Error("queue unavailable");
      return { delivered: 0, failed: 0 };
    } },
    "@mastery/shared/parent_digest.js": { runParentDigests: async () => {
      calls.push("digest"); return { configured: true, queued: 0, claimed: 0, sent: 0, failed: 0 };
    } },
  };
  const module = { exports: {} };
  vm.runInNewContext(compiled, {
    module, exports: module.exports,
    require: name => { assert(name in adapters, "Unexpected dependency: " + name); return adapters[name]; },
    console: { info() {}, error() {} },
  });
  const env = boundQueues ? { STRUCTURE_QUEUE: {}, CONTENT_QUEUE: {} } : {};
  return { calls, run: cron => module.exports.default.scheduled({ cron }, env) };
}

test("minute cron dispatches only the outbox", async () => {
  const f = fixture(); await f.run("* * * * *");
  assert.deepEqual(f.calls, ["outbox"]);
});
test("weekly digest cron never dispatches outbox or maintenance", async () => {
  const f = fixture(); await f.run("30 12 * * SUN");
  assert.deepEqual(f.calls, ["digest"]);
});
test("maintenance cron dispatches outbox before independent maintenance", async () => {
  const f = fixture(); await f.run("*/15 * * * *");
  assert.deepEqual(f.calls, ["outbox", "alerts", "purges", "deletions"]);
});
test("failed fallback cannot suppress alerts, purges or deletion claims", async () => {
  const f = fixture({ failDispatch: true }); await f.run("*/15 * * * *");
  assert.deepEqual(f.calls, ["outbox", "alerts", "purges", "deletions"]);
});
test("missing queue bindings skip dispatch and still maintain retention", async () => {
  const f = fixture({ boundQueues: false }); await f.run("*/15 * * * *");
  assert.deepEqual(f.calls, ["alerts", "purges", "deletions"]);
});
