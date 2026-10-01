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

test("AXO-126: mastery-api binds INTELLIGENCE only when the tutor Worker is deployed first", async () => {
  const workflow = await readFile(path.join(root, ".github/workflows/deploy.yml"), "utf8");
  const tutorJob = workflow.slice(workflow.indexOf("deploy-intelligence-tutor:"), workflow.indexOf("\n  deploy:\n"));
  assert.match(tutorJob, /if:.*vars\.TUTOR_DEPLOY_ENABLED == 'true'/, "tutor deploy must be switched off by default");
  const migrate = tutorJob.indexOf("d1 migrations apply axon-intelligence --remote --env tutor");
  const deploy = tutorJob.indexOf("wrangler deploy --env tutor");
  assert.ok(migrate > -1 && deploy > migrate, "D1 migrations must run before the tutor deploy");

  const deployJob = workflow.slice(workflow.indexOf("\n  deploy:\n"));
  assert.match(deployJob, /needs:\s*\[[^\]]*deploy-intelligence-tutor[^\]]*\]/, "api deploy must wait for the tutor deploy");
  assert.match(deployJob, /deploy-intelligence-tutor\.result == 'success' \|\| needs\.deploy-intelligence-tutor\.result == 'skipped'/,
    "api must not deploy after a failed tutor deploy");
  const bind = deployJob.split("\n").findIndex((line) => line.includes('binding = "INTELLIGENCE"'));
  assert.ok(bind > -1, "the INTELLIGENCE binding step is missing");
  const guard = deployJob.split("\n").slice(Math.max(0, bind - 2), bind + 1).join("\n");
  assert.match(guard, /matrix\.worker == 'api' && vars\.TUTOR_DEPLOY_ENABLED == 'true'/, "binding must be guarded by the same switch");
  assert.match(deployJob, /TUTOR_ROLLOUT:\$\{\{ vars\.TUTOR_ROLLOUT \|\| 'off' \}\}/, "rollout stage must default to off");

  const api = await readFile(path.join(root, "workers/api/wrangler.toml"), "utf8");
  assert.ok(!/axon-intelligence/.test(api), "the static api config must not bind a Worker that may not exist");
});
