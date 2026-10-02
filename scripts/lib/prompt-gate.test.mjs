import assert from "node:assert/strict";
import test from "node:test";
import { checkPromptGate, sha256 } from "./prompt-gate.mjs";

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
const passing = (over = {}) => ({ kind: "eval", passed: true, prompt_files: [FILE], human_labelled_extraction: true, ...over });

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
