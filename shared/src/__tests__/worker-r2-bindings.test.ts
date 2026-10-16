// AXO-213: the scheme check failed in production because mastery-explain had no
// R2 binding, and imageRef threw before any model call. Unit tests mock images,
// so this reads each worker's wrangler.toml: a worker whose source calls
// imageRef, or runs the scheme check (shared/scheme_check.ts calls imageRef),
// must bind the bucket imageRef reads ("derived" -> DERIVED).
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync, readdirSync, existsSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const workers = join(dirname(fileURLToPath(import.meta.url)), "../../../workers");

function sourceFiles(dir: string): string[] {
  return readdirSync(dir, { withFileTypes: true }).flatMap((entry) =>
    entry.isDirectory() ? sourceFiles(join(dir, entry.name)) : entry.name.endsWith(".ts") ? [join(dir, entry.name)] : []);
}

function r2Bindings(toml: string): Set<string> {
  const names = new Set<string>();
  for (const block of toml.split("[[r2_buckets]]").slice(1)) {
    const m = block.match(/binding\s*=\s*"([A-Z_]+)"/);
    if (m) names.add(m[1]!);
  }
  return names;
}

test("every worker that reads page images binds the derived bucket", () => {
  const checked: string[] = [];
  for (const name of readdirSync(workers)) {
    const toml = join(workers, name, "wrangler.toml");
    const src = join(workers, name, "src");
    if (!existsSync(toml) || !existsSync(src)) continue;
    const callsImageRef = sourceFiles(src).some((file) => /\bimageRef\(|\brunSchemeCheck\b/.test(readFileSync(file, "utf8")));
    if (!callsImageRef) continue;
    checked.push(name);
    assert.ok(r2Bindings(readFileSync(toml, "utf8")).has("DERIVED"), `${name} calls imageRef but wrangler.toml has no DERIVED R2 binding`);
  }
  assert.ok(checked.includes("explain"), "the explain worker (scheme check) is covered");
});
