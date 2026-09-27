import { test } from "node:test";
import assert from "node:assert/strict";
import { signAssetUrl } from "../r2.js";

test("asset signing reuses one imported HMAC key until the secret rotates", async () => {
  const subtle = globalThis.crypto.subtle as any;
  const original = subtle.importKey.bind(subtle);
  let imports = 0;
  subtle.importKey = (...args: any[]) => {
    imports++;
    return original(...args);
  };

  try {
    const envA = {
      ASSET_SIGNING_SECRET: "axo110-secret-a",
      MASTERY_ASSET_URL: "https://api.test",
    } as any;
    const envB = {
      ASSET_SIGNING_SECRET: "axo110-secret-b",
      MASTERY_ASSET_URL: "https://api.test",
    } as any;

    const before = imports;
    await signAssetUrl(envA, "derived", "student/paper/page-1.webp");
    const afterFirst = imports;
    await signAssetUrl(envA, "derived", "student/paper/page-2.webp");
    const afterSecond = imports;
    await signAssetUrl(envB, "derived", "student/paper/page-3.webp");
    const afterRotation = imports;

    assert.equal(afterFirst - before, 1, "first use imports one HMAC key");
    assert.equal(afterSecond, afterFirst, "same secret reuses the imported key");
    assert.equal(afterRotation - afterSecond, 1, "secret rotation imports a fresh key");
  } finally {
    subtle.importKey = original;
  }
});
