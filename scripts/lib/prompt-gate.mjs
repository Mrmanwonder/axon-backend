// The prompt half of the eval gate (AXO-41/44/125). A prompt change moves answers as much as a
// model change does, so it needs the same linked passing eval run.
//
// eval/prompt-manifest.json records the sha256 of every prompt source file and the eval run that
// last passed for that exact text. A prompt edit changes the hash, which fails CI until the
// manifest is updated; and an entry whose hash changed against the base branch must link a real,
// passing eval run (eval/runs/<uuid>.json) that names the file. The initial hashes are a
// baseline: grandfathered at the date shown, never evaluated, and only valid while unchanged.
//
//   node scripts/check-prompt-gate.mjs --base origin/main

import { createHash } from "node:crypto";

const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

export const sha256 = (text) => createHash("sha256").update(text).digest("hex");

/**
 * @param {object} args
 * @param {string[]} args.promptFiles   every prompt source file in the repo
 * @param {(file: string) => string} args.read  file text
 * @param {{prompts: Record<string, {sha256: string, eval_run: string}>}} args.manifest
 * @param {{prompts: Record<string, {sha256: string}>} | null} args.baseManifest  manifest on the base branch
 * @param {(id: string) => object | null} args.readEvidence
 */
export function checkPromptGate({ promptFiles, read, manifest, baseManifest, readEvidence }) {
  const violations = [];
  const entries = manifest?.prompts ?? {};
  const base = baseManifest?.prompts ?? {};

  for (const file of promptFiles) {
    const entry = entries[file];
    if (!entry) {
      violations.push(`${file}: prompt has no entry in eval/prompt-manifest.json.`);
      continue;
    }
    if (sha256(read(file)) !== entry.sha256) {
      violations.push(`${file}: prompt text changed but its manifest hash did not. Re-run the eval, then update the hash and eval_run together.`);
      continue;
    }
    // The first time the manifest exists there is nothing to compare with: the current text is
    // recorded as a dated baseline. After that a baseline can never back a change.
    if (baseManifest === null && /^baseline-\d{4}-\d{2}-\d{2}$/.test(String(entry.eval_run ?? ""))) continue;
    const changedFromBase = !base[file] || base[file].sha256 !== entry.sha256;
    if (!changedFromBase) continue;

    if (!UUID.test(String(entry.eval_run ?? ""))) {
      violations.push(`${file}: prompt is new or changed against the base branch but its eval_run is "${entry.eval_run}". It needs a linked passing eval run, not a baseline.`);
      continue;
    }
    const evidence = readEvidence(entry.eval_run.toLowerCase());
    if (!evidence) violations.push(`${file}: eval run ${entry.eval_run} has no committed evidence at eval/runs/${entry.eval_run}.json.`);
    else if (evidence.kind !== "eval" || evidence.passed !== true) violations.push(`${file}: eval run ${entry.eval_run} did not pass.`);
    else if (!(evidence.prompt_files ?? []).includes(file)) violations.push(`${file}: eval run ${entry.eval_run} did not evaluate this prompt file.`);
    else if (evidence.release_gate_extraction === true && evidence.human_labelled_extraction !== true) {
      violations.push(`${file}: eval run ${entry.eval_run} gates extraction on draft labels. Draft labels are never release-gate truth.`);
    }
  }

  for (const file of Object.keys(entries)) {
    if (!promptFiles.includes(file)) violations.push(`${file}: manifest lists a prompt file that no longer exists. Remove it deliberately.`);
  }
  return violations;
}
