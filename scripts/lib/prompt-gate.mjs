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



/** JSON.parse silently keeps the last duplicate key; release manifests must not. */
export function parseUniqueJson(source) {
  const value = JSON.parse(source);
  const tokens = source.match(/"(?:\\.|[^"\\])*"|[{}\[\],:]|true|false|null|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?/g) ?? [];
  const stack = [];
  for (let i = 0; i < tokens.length; i++) {
    const token = tokens[i];
    if (token === "{") stack.push(new Set());
    else if (token === "[") stack.push(null);
    else if (token === "}" || token === "]") stack.pop();
    else if (token.startsWith('"') && tokens[i + 1] === ":") {
      const key = JSON.parse(token);
      const keys = stack.at(-1);
      if (keys?.has(key)) throw new Error(`Duplicate JSON key: ${key}`);
      keys?.add(key);
    }
  }
  return value;
}

const ZERO_UUID = "00000000-0000-0000-0000-000000000000";
const textValue = (value) => typeof value === "string" && value.trim().length > 0;
const record = (value) => value !== null && typeof value === "object" && !Array.isArray(value);

/** Checks artifact completeness/identity, not the truth of a declared result. */
export function measuredEvidenceProblems(evidence, runId) {
  const problems = [];
  if (evidence.id !== runId) problems.push("measured run id does not match its evidence reference");
  for (const key of ["stage", "golden_set_version", "golden_set_source", "candidate_key"]) {
    if (!textValue(evidence[key])) problems.push(`missing measured ${key}`);
  }
  if (!textValue(evidence.run_at) || !/^\d{4}-\d{2}-\d{2}(?:T.*)?$/.test(evidence.run_at)
      || !Number.isFinite(Date.parse(evidence.run_at))) problems.push("missing or invalid measured run date");
  if (!Array.isArray(evidence.candidate_models) || !evidence.candidate_models.length
      || evidence.candidate_models.some((model) => !textValue(model))) problems.push("missing measured model identity");
  if (!Number.isSafeInteger(evidence.cases) || evidence.cases <= 0) problems.push("missing positive measured case count");
  if (!record(evidence.metrics) || !Number.isSafeInteger(evidence.metrics.model_calls)
      || evidence.metrics.model_calls <= 0) problems.push("missing positive measured model-call count");
  if (!record(evidence.metrics) || !Number.isSafeInteger(evidence.metrics.failed_calls)
      || evidence.metrics.failed_calls < 0 || evidence.metrics.failed_calls > evidence.metrics.model_calls) {
    problems.push("missing or invalid measured failed-call count");
  }
  if (!record(evidence.thresholds) || !Object.keys(evidence.thresholds).length
      || Object.values(evidence.thresholds).some((value) =>
        !(typeof value === "number" && Number.isFinite(value)) && !textValue(value))) {
    problems.push("missing or invalid measured thresholds");
  }
  const source = evidence.provenance;
  if (!record(source) || source.kind !== "supabase_eval_run" || source.eval_run_id !== runId
      || !textValue(source.project_id) || source.result_count !== evidence.cases
      || !/^[a-f0-9]{64}$/.test(String(source.result_sha256 ?? ""))
      || !textValue(source.verified_at) || !Number.isFinite(Date.parse(source.verified_at))) {
    problems.push("missing exact-run per-case result provenance");
  }
  return problems;
}

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
    // Only unchanged dated baselines are grandfathered. Every measured artifact
    // remains validated even when the prompt hash is unchanged: swapping just
    // its run reference must not bypass evidence checks.
    if (!changedFromBase && /^baseline-\d{4}-\d{2}-\d{2}$/.test(String(entry.eval_run ?? ""))) continue;

    if (!UUID.test(String(entry.eval_run ?? "")) || entry.eval_run === ZERO_UUID) {
      violations.push(`${file}: prompt is new or changed against the base branch but its eval_run is "${entry.eval_run}". It needs a linked passing eval run, not a baseline.`);
      continue;
    }
    const evidence = readEvidence(entry.eval_run.toLowerCase());
    if (!evidence) violations.push(`${file}: eval run ${entry.eval_run} has no committed evidence at eval/runs/${entry.eval_run}.json.`);
    else if (evidence.kind !== "eval" || evidence.passed !== true) violations.push(`${file}: eval run ${entry.eval_run} did not pass.`);
    else if (!(evidence.prompt_files ?? []).includes(file)) violations.push(`${file}: eval run ${entry.eval_run} did not evaluate this prompt file.`);
    else if (evidence.release_gate_extraction === true && evidence.human_labelled_extraction !== true) {
      violations.push(`${file}: eval run ${entry.eval_run} gates extraction on draft labels. Draft labels are never release-gate truth.`);
    } else {
      for (const problem of measuredEvidenceProblems(evidence, entry.eval_run.toLowerCase())) {
        violations.push(`${file}: eval run ${entry.eval_run}: ${problem}.`);
      }
    }
  }

  for (const file of Object.keys(entries)) {
    if (!promptFiles.includes(file)) violations.push(`${file}: manifest lists a prompt file that no longer exists. Remove it deliberately.`);
  }
  return violations;
}
