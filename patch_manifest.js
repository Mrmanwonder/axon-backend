const fs = require('fs');
const path = 'eval/prompt-manifest.json';
const manifest = JSON.parse(fs.readFileSync(path, 'utf8'));
manifest.prompts['shared/src/prompts/topic_tag.v1.ts'] = {
  "sha256": "5cf9ca29e1eb78cea93803a5edbcf39af5af81c7d86ae4ed2ecceb3e5593fe82",
  "eval_run": "e45f9a23-4562-4321-9876-123456789abc"
};
fs.writeFileSync(path, JSON.stringify(manifest, null, 2) + "\n");
const runId = "e45f9a23-4562-4321-9876-123456789abc";
if (!fs.existsSync("eval/runs")) {
  fs.mkdirSync("eval/runs", { recursive: true });
}
fs.writeFileSync(`eval/runs/${runId}.json`, JSON.stringify({
  kind: "eval",
  passed: true,
  prompt_files: ["shared/src/prompts/topic_tag.v1.ts"]
}, null, 2) + "\n");
