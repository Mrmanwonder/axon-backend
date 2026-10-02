import assert from "node:assert/strict";
import { readdir, readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";

const root = path.resolve(import.meta.dirname, "../..");

function quotedList(value) {
  return value.split(",").map((item) => item.trim()).filter(Boolean);
}

function configName(text) {
  return text.match(/^name\s*=\s*"([^"]+)"/m)?.[1]
    ?? text.match(/"name"\s*:\s*"([^"]+)"/)?.[1];
}

function serviceTargets(text) {
  return [...text.matchAll(/(?:^service\s*=|"service"\s*:)\s*"([^"]+)"/gm)].map((match) => match[1]);
}

test("a deployed Worker never binds to a locally managed Worker excluded from the deploy plan", async () => {
  const workflow = await readFile(path.join(root, ".github/workflows/deploy.yml"), "utf8");
  const matrices = [...workflow.matchAll(/worker:\s*\[([^\]]+)\]/g)];
  assert.ok(matrices.length >= 2, "expected dry-run and deploy worker matrices");
  const deployedDirectories = new Set(quotedList(matrices.at(-1)[1]));

  // Workers deployed by their own step rather than the matrix (axon-intelligence, Tutor-only
  // profile, AXO-126). A real deploy is a `wrangler deploy` that is not a dry run; the config it
  // ships is the one named by -c, else the directory's default. That config, not the default, is
  // what is checked for bindings.
  const configOverrides = new Map();
  for (const match of workflow.matchAll(/- run: npx wrangler deploy(?! --dry-run)([^\n]*)\n\s+working-directory:\s*workers\/([\w-]+)\s*\n/g)) {
    deployedDirectories.add(match[2]);
    const config = match[1].match(/(?:^|\s)-c\s+(\S+)/)?.[1];
    if (config) configOverrides.set(match[2], config);
  }

  const workerDirectories = await readdir(path.join(root, "workers"), { withFileTypes: true });
  const configs = new Map();
  for (const directory of workerDirectories.filter((entry) => entry.isDirectory())) {
    let configPath;
    for (const filename of [configOverrides.get(directory.name), "wrangler.toml", "wrangler.jsonc"].filter(Boolean)) {
      const candidate = path.join(root, "workers", directory.name, filename);
      try { await readFile(candidate, "utf8"); configPath = candidate; break; }
      catch (error) { if (error?.code !== "ENOENT") throw error; }
    }
    if (!configPath) continue;
    const text = await readFile(configPath, "utf8");
    const name = configName(text);
    assert.ok(name, `missing Worker name in ${path.relative(root, configPath)}`);
    configs.set(directory.name, { name, targets: serviceTargets(text) });
  }

  const deployedServices = new Set([...deployedDirectories].map((directory) => {
    assert.ok(configs.has(directory), `deploy plan references unknown Worker directory ${directory}`);
    return configs.get(directory).name;
  }));
  const localServices = new Set([...configs.values()].map((config) => config.name));

  for (const directory of deployedDirectories) {
    for (const target of configs.get(directory).targets) {
      if (!localServices.has(target)) continue;
      assert.ok(deployedServices.has(target), `${directory} binds to locally managed ${target}, but that target is excluded from the deploy plan`);
    }
  }
});

test("document-vision stays out of the deploy plan, and the Tutor profile binds nothing local to it", async () => {
  const workflow = await readFile(path.join(root, ".github/workflows/deploy.yml"), "utf8");
  const deploySteps = [...workflow.matchAll(/- run: npx wrangler deploy(?! --dry-run)[^\n]*\n\s+working-directory:\s*([^\n]+)/g)].map((m) => m[1].trim());
  assert.ok(!deploySteps.some((dir) => dir.includes("document-vision")), "document-vision must not be deployed");
  const tutor = await readFile(path.join(root, "workers/intelligence/wrangler.tutor.jsonc"), "utf8");
  assert.deepEqual(serviceTargets(tutor), [], "the Tutor-only profile must not bind any service");
  assert.ok(!/DOCUMENT_VISION|axon-paper-processing/.test(tutor.replace(/\/\/[^\n]*/g, "")), "the Tutor-only profile must not bind the document pipeline");
});

test("services that bind axon-intelligence deploy after it", async () => {
  const workflow = await readFile(path.join(root, ".github/workflows/deploy.yml"), "utf8");
  assert.match(workflow, /deploy-intelligence:/);
  assert.match(workflow, /needs:\s*\[typecheck-and-test, dry-run, deploy-intelligence\]/);
});
