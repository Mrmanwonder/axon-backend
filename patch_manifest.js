const fs = require('fs');
const path = 'eval/prompt-manifest.json';
const manifest = JSON.parse(fs.readFileSync(path, 'utf8'));
manifest.prompts['shared/src/prompts/topic_tag.v1.ts'] = {
  "sha256": "5cf9ca29e1eb78cea93803a5edbcf39af5af81c7d86ae4ed2ecceb3e5593fe82",
  "eval_run": "baseline-2026-10-02"
};
fs.writeFileSync(path, JSON.stringify(manifest, null, 2) + "\n");
