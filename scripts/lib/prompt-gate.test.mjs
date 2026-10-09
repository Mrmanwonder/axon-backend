import assert from "node:assert/strict";
import test from "node:test";
import { checkPromptGate, sha256, parseUniqueJson } from "./prompt-gate.mjs";

const ID = "11111111-2222-4333-8444-555555555555";
const FILE = "shared/src/prompts/explain_tier1.v2.ts";
const BASELINE = "baseline-2026-10-02";

function setup({ text = "prompt v1", entry, base, evidence }) {
  return {
    promptFiles: [FILE],
    read: () => text,
    manifest: { prompts: { [FILE]: entry } },
    baseManifest: base === undefined ? { prompts: { [FILE]: { sha256: sha256("prompt v1"), eval_run: BASELINE } } } : base,
    readEvidence: (id) => (id === ID ? evidence ?? null : null),
  };
}
const entryFor = (text, eval_run) => ({ sha256: sha256(text), eval_run });
const passing = (over = {}) => ({
  id: ID, kind: "eval", passed: true, prompt_files: [FILE],
  stage: "explain", run_at: "2026-10-09", golden_set_version: "human-labelled-v1",
  golden_set_source: "Controlled human-labelled test corpus", candidate_key: "candidate-v1",
  candidate_models: ["test-model"], cases: 10,
  metrics: { model_calls: 10, failed_calls: 0 }, thresholds: { schema_valid_min: 0.98 },
  provenance: { kind: "supabase_eval_run", project_id: "test-project", eval_run_id: ID,
    result_count: 10, result_sha256: "a".repeat(64), verified_at: "2026-10-09" },
  human_labelled_extraction: true, ...over,
});

test("unchanged prompt text matching the manifest passes", () => {
  assert.deepEqual(checkPromptGate(setup({ entry: entryFor("prompt v1", BASELINE) })), []);
});

test("editing a prompt without touching the manifest is refused", () => {
  const v = checkPromptGate(setup({ text: "prompt v2", entry: entryFor("prompt v1", BASELINE) }));
  assert.match(v[0], /prompt text changed but its manifest hash did not/);
});

test("a changed prompt cannot be waved through as a baseline", () => {
  const v = checkPromptGate(setup({ text: "prompt v2", entry: entryFor("prompt v2", BASELINE) }));
  assert.match(v[0], /needs a linked passing eval run, not a baseline/);
});

test("a changed prompt with a passing run that evaluated it passes", () => {
  const v = checkPromptGate(setup({ text: "prompt v2", entry: entryFor("prompt v2", ID), evidence: passing() }));
  assert.deepEqual(v, []);
});

test("a changed prompt whose run is missing, failing, or about another file is refused", () => {
  const run = (evidence) => checkPromptGate(setup({ text: "prompt v2", entry: entryFor("prompt v2", ID), evidence })).join("\n");
  assert.match(run(null), /no committed evidence/);
  assert.match(run(passing({ passed: false })), /did not pass/);
  assert.match(run(passing({ prompt_files: ["shared/src/prompts/other.ts"] })), /did not evaluate this prompt file/);
});

test("draft labels are never release-gate truth for a prompt either", () => {
  const v = checkPromptGate(setup({ text: "prompt v2", entry: entryFor("prompt v2", ID), evidence: passing({ release_gate_extraction: true, human_labelled_extraction: false }) }));
  assert.match(v.join("\n"), /Draft labels are never release-gate truth/);
});

test("a brand new prompt file needs an entry and a passing run", () => {
  const args = setup({ text: "x", entry: undefined, base: { prompts: {} } });
  args.manifest = { prompts: {} };
  assert.match(checkPromptGate(args)[0], /no entry in eval\/prompt-manifest\.json/);
});

test("the first time the manifest exists, a dated baseline is accepted", () => {
  assert.deepEqual(checkPromptGate(setup({ entry: entryFor("prompt v1", BASELINE), base: null })), []);
});

test("a manifest entry for a deleted prompt file is flagged", () => {
  const args = setup({ entry: entryFor("prompt v1", BASELINE) });
  args.promptFiles = [];
  assert.match(checkPromptGate(args)[0], /no longer exists/);
});

test("a bare passed declaration is not measured evidence even for an unchanged prompt", () => {
  const args = setup({ entry: entryFor("prompt v1", ID),
    evidence: { kind: "eval", passed: true, prompt_files: [FILE] } });
  const v = checkPromptGate(args).join("\n");
  assert.match(v, /measured run id/);
  assert.match(v, /positive measured case count/);
  assert.match(v, /per-case result provenance/);
});

test("changing only an unchanged prompt's run reference cannot bypass validation", () => {
  const args = setup({ entry: entryFor("prompt v1", ID),
    evidence: passing({ id: "99999999-2222-4333-8444-555555555555" }) });
  assert.match(checkPromptGate(args).join("\n"), /measured run id does not match/);
});

test("all-zero run IDs are refused before reading declared evidence", () => {
  const args = setup({ text: "prompt v2", entry: entryFor("prompt v2", "00000000-0000-0000-0000-000000000000") });
  args.readEvidence = () => passing();
  assert.match(checkPromptGate(args).join("\n"), /needs a linked passing eval run/);
});

for (const [field, value, reason] of [
  ["cases", 0, /positive measured case count/],
  ["candidate_models", [], /model identity/],
  ["metrics", { model_calls: 0, failed_calls: 0 }, /model-call count/],
  ["metrics", { model_calls: 10, failed_calls: 11 }, /failed-call count/],
  ["thresholds", {}, /thresholds/],
  ["run_at", "not-a-date", /run date/],
  ["provenance", { kind: "supabase_eval_run", eval_run_id: ID, result_count: 10 }, /per-case result provenance/],
]) {
  test(`incomplete measured evidence is refused: ${field}`, () => {
    const args = setup({ text: "prompt v2", entry: entryFor("prompt v2", ID),
      evidence: passing({ [field]: value }) });
    assert.match(checkPromptGate(args).join("\n"), reason);
  });
}

test("per-case result count must cover the declared measured cases", () => {
  const evidence = passing();
  evidence.provenance.result_count = 9;
  assert.match(checkPromptGate(setup({ text: "prompt v2", entry: entryFor("prompt v2", ID), evidence })).join("\n"), /per-case result provenance/);
});

test("duplicate manifest keys cannot shadow measured evidence", () => {
  assert.throws(() => parseUniqueJson('{"prompts":{"x.ts":{"eval_run":"real"},"x.ts":{"eval_run":"placeholder"}}}'), /Duplicate JSON key: x.ts/);
  assert.deepEqual(parseUniqueJson('{"prompts":{"a.ts":{"sha256":"a"},"b.ts":{"sha256":"b"}}}'), {
    prompts: { "a.ts": { sha256: "a" }, "b.ts": { sha256: "b" } },
  });
});
