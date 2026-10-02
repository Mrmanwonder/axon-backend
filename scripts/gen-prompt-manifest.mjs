// Writes eval/prompt-manifest.json for the current prompt text. Run it ONLY to record a baseline,
// or after an eval run has passed for the new text (pass --eval-run <uuid>).
import { readFileSync, writeFileSync, existsSync, readdirSync } from "node:fs";
import path from "node:path";
import { sha256 } from "./lib/prompt-gate.mjs";

const root = path.resolve(import.meta.dirname, "..");
const arg = (n) => process.argv[process.argv.indexOf(n) + 1];
const evalRun = arg("--eval-run");
if (!evalRun) { console.error("pass --eval-run <uuid | baseline-YYYY-MM-DD>"); process.exit(2); }

const files = [
  ...readdirSync(path.join(root, "shared/src/prompts")).filter((f) => f.endsWith(".ts")).map((f) => `shared/src/prompts/${f}`),
  "shared/src/prompts.ts",
  "workers/intelligence/src/prompts/kernel.v3.ts",
  "workers/intelligence/src/prompts/registry.ts",
  "workers/intelligence/src/prompts/scanner/index.ts",
].filter((f) => existsSync(path.join(root, f))).sort();

const file = path.join(root, "eval/prompt-manifest.json");
const previous = existsSync(file) ? JSON.parse(readFileSync(file, "utf8")).prompts : {};
const prompts = {};
for (const f of files) {
  const hash = sha256(readFileSync(path.join(root, f), "utf8"));
  // Only entries whose text changed take the new eval run; the rest keep what they had.
  prompts[f] = previous[f]?.sha256 === hash ? previous[f] : { sha256: hash, eval_run: evalRun };
}
writeFileSync(file, JSON.stringify({ comment: "See scripts/lib/prompt-gate.mjs. Do not edit by hand.", prompts }, null, 2) + "\n");
console.log(`wrote ${files.length} entries`);
