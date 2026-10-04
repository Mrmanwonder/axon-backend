import { test } from "node:test";
import assert from "node:assert/strict";
import { sealUpload, stagingKey, presignPut } from "../r2.js";

function fixture() {
  const objects = new Map<string, { size: number; etag: string; httpMetadata: { contentType: string }; body: Uint8Array }>();
  const value = (size: number, etag: string) => ({ size, etag, httpMetadata: { contentType: "image/jpeg" }, body: new Uint8Array(size) });
  const bucket = {
    head: async (key: string) => objects.get(key) ?? null,
    get: async (key: string) => objects.get(key) ?? null,
    delete: async (key: string) => { objects.delete(key); },
    put: async (key: string, body: Uint8Array, options: any) => {
      assert.equal(options.onlyIf.etagDoesNotMatch, "*");
      if (objects.has(key)) return null;
      const object = value(body.byteLength, "canonical");
      objects.set(key, object);
      return object;
    },
  };
  return { objects, value, env: { DERIVED: bucket } as any };
}
test("presigned PUTs target staging keys, keeping canonical objects outside client capabilities", async () => {
  const env = { R2_ENDPOINT: "https://account.r2.cloudflarestorage.com", R2_BUCKET_DERIVED: "derived",
    R2_ACCESS_KEY_ID: "fixture", R2_SECRET_ACCESS_KEY: "fixture" } as any;
  const url = await presignPut(env, "derived", "student/paper/page/image.jpg", "image/jpeg");
  assert.equal(new URL(url).pathname, "/derived/student/paper/page/image.jpg.pending");
});
test("confirmed bytes stay immutable when the same PUT capability is reused", async () => {
  const f = fixture(), key = "student/paper/page/image.jpg";
  f.objects.set(stagingKey(key), f.value(10, "first"));
  assert.equal((await sealUpload(f.env, "derived", key, 10, 25))?.bytes, 10);
  f.objects.set(stagingKey(key), f.value(100, "oversized-replay"));
  assert.equal((await sealUpload(f.env, "derived", key, 10, 25))?.bytes, 10);
  assert.equal(f.objects.get(key)?.size, 10);
  assert.equal(f.objects.has(stagingKey(key)), false);
});
test("oversized or undeclared staging objects never become consumable and are cleaned", async () => {
  for (const actual of [0, 11, 100]) {
    const f = fixture(), key = "student/paper/page/image.jpg";
    f.objects.set(stagingKey(key), f.value(actual, "bad"));
    await assert.rejects(sealUpload(f.env, "derived", key, 10, 25));
    assert.equal(f.objects.has(key), false);
    assert.equal(f.objects.has(stagingKey(key)), false);
  }
});
test("promotion validates the same GET object rather than a separate HEAD snapshot", async () => {
  const f = fixture(), key = "student/paper/page/image.jpg";
  f.objects.set(stagingKey(key), f.value(100, "replacement"));
  await assert.rejects(sealUpload(f.env, "derived", key, 10, 25));
  assert.equal(f.objects.has(key), false);
});
