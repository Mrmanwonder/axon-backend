import { readFile, stat } from "node:fs/promises";
import { isAbsolute, relative, resolve } from "node:path";
import process from "node:process";
import { RELEASE_ARTIFACT_HASH_FIELDS, ROLLOUT_ORDER, validateReleaseEvidence } from "./release-evidence.mjs";

const failures = [];
const pass = (condition, message) => { if (!condition) failures.push(message); };

const config = JSON.parse(await readFile(new URL("../wrangler.jsonc", import.meta.url), "utf8"));
const databaseId = config.d1_databases?.[0]?.database_id;
const kvId = config.kv_namespaces?.[0]?.id;
pass(databaseId && !/^0+$|^00000000-0000-0000-0000-000000000000$/.test(databaseId), "real D1 database_id is not configured");
pass(kvId && !/^0+$/.test(kvId), "real KV namespace id is not configured");
pass(Boolean(config.r2_buckets?.[0]?.bucket_name), "R2 paper bucket is not configured");
pass(Boolean(config.queues?.producers?.[0]?.queue), "paper-processing queue is not configured");
pass(Boolean(config.queues?.consumers?.[0]?.dead_letter_queue), "paper dead-letter queue is not configured");

let certification;
try { certification = JSON.parse(await readFile(new URL("../certification/release.json", import.meta.url), "utf8")); }
catch { failures.push("certification/release.json is absent; create it only from real reviewed evidence"); }

const evidenceDirectory = process.env.AXON_RELEASE_EVIDENCE_DIR;
const targetStage = process.env.AXON_RELEASE_TARGET_STAGE ?? "FULL";
let verifiedEvidence;
pass(Boolean(evidenceDirectory), "AXON_RELEASE_EVIDENCE_DIR is not present in the release environment");
pass(ROLLOUT_ORDER.includes(targetStage), `AXON_RELEASE_TARGET_STAGE ${targetStage} is invalid`);
if (certification && evidenceDirectory && ROLLOUT_ORDER.includes(targetStage)) {
  const evidenceRoot = resolve(evidenceDirectory);
  const artifactBytes = {};
  for (const artifactName of Object.keys(RELEASE_ARTIFACT_HASH_FIELDS)) {
    const filename = certification.artifacts?.[artifactName];
    if (typeof filename !== "string" || filename.length === 0) continue;
    const artifactPath = resolve(evidenceRoot, filename);
    const pathFromRoot = relative(evidenceRoot, artifactPath);
    if (isAbsolute(filename) || pathFromRoot.startsWith("..") || isAbsolute(pathFromRoot)) {
      failures.push(`artifact ${artifactName} escapes AXON_RELEASE_EVIDENCE_DIR`);
      continue;
    }
    try {
      const metadata = await stat(artifactPath);
      pass(metadata.isFile(), `artifact ${artifactName} is not a regular file`);
      pass(metadata.size <= 64 * 1024 * 1024, `artifact ${artifactName} exceeds the 64 MiB verification limit`);
      if (metadata.isFile() && metadata.size <= 64 * 1024 * 1024) artifactBytes[artifactName] = await readFile(artifactPath);
    } catch {
      failures.push(`artifact ${artifactName} is unavailable at its certified path`);
    }
  }
  const evidence = validateReleaseEvidence(certification, artifactBytes, { targetStage });
  failures.push(...evidence.errors);
  if (evidence.valid) verifiedEvidence = evidence;
}

for (const name of ["CLOUDFLARE_API_TOKEN", "GOOGLE_API_KEY", "SUPABASE_SERVICE_ROLE_KEY", "TAVILY_API_KEY", "AXON_INTERNAL_TOKEN", "AXON_ADMIN_TOKEN", "AXON_PSEUDONYM_KEY", "AXON_VISION_TOKEN", "GEMINI_INPUT_USD_PER_MILLION", "GEMINI_OUTPUT_USD_PER_MILLION"]) {
  pass(Boolean(process.env[name]), `${name} is not present in the release environment`);
}
pass(Boolean(process.env.AXON_VISION_API_BASE), "AXON_VISION_API_BASE is not present in the release environment");
pass(process.env.GEMINI_PRIVACY_MODE === "zdr", "GEMINI_PRIVACY_MODE is not certified as zdr");
pass(process.env.AXON_VISION_PRIVACY_MODE === "zdr", "AXON_VISION_PRIVACY_MODE is not certified as zdr");

if (failures.length > 0) {
  console.error(`AXON release preflight blocked:\n- ${failures.join("\n- ")}`);
  process.exitCode = 1;
} else {
  console.log(JSON.stringify({ releaseEvidence: "verified", targetStage, metrics: verifiedEvidence.metrics }));
  console.log("AXON release preflight passed.");
}
