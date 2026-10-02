import { execFileSync } from "node:child_process";
import { existsSync, readFileSync, readdirSync } from "node:fs";
import path from "node:path";
import { checkPromptGate } from "./lib/prompt-gate.mjs";

const root = path.resolve(import.meta.dirname, "..");
const base = process.argv[process.argv.indexOf("--base") + 1] ?? "origin/main";

// Every file that carries prompt text a model is shown.
const PROMPT_FILES = [
  ...readdirSync(path.join(root, "shared/src/prompts")).filter((f) => f.endsWith(".ts")).map((f) => `shared/src/prompts/${f}`),
  "shared/src/prompts.ts",
  "workers/intelligence/src/prompts/kernel.v3.ts",
  "workers/intelligence/src/prompts/registry.ts",
  "workers/intelligence/src/prompts/scanner/index.ts",
].filter((f) => existsSync(path.join(root, f))).sort();

const manifest = JSON.parse(readFileSync(path.join(root, "eval/prompt-manifest.json"), "utf8"));
let baseManifest = null;
try {
  baseManifest = JSON.parse(execFileSync("git", ["show", `${base}:eval/prompt-manifest.json`], { cwd: root, encoding: "utf8", stdio: ["ignore", "pipe", "ignore"] }));
} catch { /* the manifest is new in this change: everything counts as changed */ }

const violations = checkPromptGate({
  promptFiles: PROMPT_FILES,
  read: (f) => readFileSync(path.join(root, f), "utf8"),
  manifest,
  baseManifest,
  readEvidence: (id) => {
    const file = path.join(root, "eval", "runs", `${id}.json`);
    if (!existsSync(file)) return null;
    try { return JSON.parse(readFileSync(file, "utf8")); } catch { return { kind: "unreadable" }; }
  },
});

if (violations.length) {
  console.error("Prompt eval gate failed:\n" + violations.map((v) => `  - ${v}`).join("\n"));
  process.exit(1);
}
console.log(`Prompt eval gate: ${PROMPT_FILES.length} prompt files match the manifest; none changed without a linked passing eval run.`);
