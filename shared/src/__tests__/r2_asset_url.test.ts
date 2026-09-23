import { test } from "node:test";
import assert from "node:assert/strict";
import { signAssetUrl } from "../r2.js";

test("signed asset URLs can be pinned to the API request origin", async () => {
  const env = { ASSET_SIGNING_SECRET: "test-secret" } as any;
  const url = await signAssetUrl(
    env,
    "derived",
    "student/paper/page/p1.webp",
    600,
    "https://mastery-api.tanmay-harkawat.workers.dev",
  );
  const parsed = new URL(url);
  assert.equal(parsed.origin, "https://mastery-api.tanmay-harkawat.workers.dev");
  assert.equal(parsed.pathname, "/asset/derived/student%2Fpaper%2Fpage%2Fp1.webp");
  assert.ok(parsed.searchParams.get("exp"));
  assert.ok(parsed.searchParams.get("sig"));
});

test("signed asset URL fallback uses the deployed worker, not the dead generic hostname", async () => {
  const env = { ASSET_SIGNING_SECRET: "test-secret" } as any;
  const url = await signAssetUrl(env, "derived", "key");
  assert.equal(new URL(url).hostname, "mastery-api.tanmay-harkawat.workers.dev");
});
