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

  const workerDirectories = await readdir(path.join(root, "workers"), { withFileTypes: true });
  const configs = new Map();
  for (const directory of workerDirectories.filter((entry) => entry.isDirectory())) {
    let configPath;
    for (const filename of ["wrangler.toml", "wrangler.jsonc"]) {
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
