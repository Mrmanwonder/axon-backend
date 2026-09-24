import { readFile } from "node:fs/promises";
import process from "node:process";

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
catch { failures.push("certification/release.json is absent; copy the example only after real evidence exists"); }
if (certification) {
  pass(certification.scannerPapers >= 100, "scanner benchmark has fewer than 100 papers");
  pass(certification.scannerQuestions >= 1_500, "scanner benchmark has fewer than 1500 questions");
  pass(certification.tutorCases >= 500, "tutor benchmark has fewer than 500 cases");
  pass(certification.handReviewedTutorCases >= 500, "fewer than 500 tutor cases are hand reviewed");
  pass(certification.privacyCertified === true, "zero-retention privacy is not certified");
  pass(certification.rollbackValidated === true, "rollback drill is not validated");
  pass(typeof certification.evidenceUri === "string" && certification.evidenceUri.length > 0, "certification evidence URI is absent");
  for (const field of ["scannerDatasetSha256", "tutorDatasetSha256", "reviewManifestSha256", "geminiZdrEvidenceSha256", "visionZdrEvidenceSha256", "rollbackEvidenceSha256"]) {
    pass(typeof certification[field] === "string" && /^[a-f0-9]{64}$/.test(certification[field]), `${field} is not a SHA-256 digest`);
  }
  pass(typeof certification.reviewer === "string" && certification.reviewer.length > 0, "certification reviewer is absent");
  pass(typeof certification.certifiedAt === "string" && Number.isFinite(Date.parse(certification.certifiedAt)), "certification timestamp is invalid");
}

for (const name of ["CLOUDFLARE_API_TOKEN", "GOOGLE_API_KEY", "SUPABASE_SERVICE_ROLE_KEY", "TAVILY_API_KEY", "AXON_INTERNAL_TOKEN", "AXON_ADMIN_TOKEN", "AXON_PSEUDONYM_KEY", "AXON_VISION_TOKEN", "GEMINI_INPUT_USD_PER_MILLION", "GEMINI_OUTPUT_USD_PER_MILLION"]) {
  pass(Boolean(process.env[name]), `${name} is not present in the release environment`);
}

if (failures.length > 0) {
  console.error(`AXON release preflight blocked:\n- ${failures.join("\n- ")}`);
  process.exitCode = 1;
} else console.log("AXON release preflight passed.");
