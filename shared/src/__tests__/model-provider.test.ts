import { test } from "node:test";
import assert from "node:assert/strict";
import { webcrypto } from "node:crypto";
import {
  clearProviderTokenCache,
  normalizeServedModel,
  resolveProviderTarget,
  signServiceAccountJwt,
  ProviderConfigError,
} from "../model-provider.js";

function b64urlToBytes(s: string): Buffer {
  const pad = "=".repeat((4 - (s.length % 4)) % 4);
  return Buffer.from(s.replaceAll("-", "+").replaceAll("_", "/") + pad, "base64");
}

function toArrayBuffer(b: Buffer): ArrayBuffer {
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer;
}

async function serviceAccount() {
  const pair = await webcrypto.subtle.generateKey(
    { name: "RSASSA-PKCS1-v1_5", modulusLength: 2048, publicExponent: new Uint8Array([1, 0, 1]), hash: "SHA-256" },
    true,
    ["sign", "verify"],
  );
  const pkcs8 = Buffer.from(await webcrypto.subtle.exportKey("pkcs8", pair.privateKey)).toString("base64");
  const pem = `-----BEGIN PRIVATE KEY-----\n${pkcs8.match(/.{1,64}/g)!.join("\n")}\n-----END PRIVATE KEY-----\n`;
  return { pair, sa: { client_email: "svc@axon-test.iam.gserviceaccount.com", private_key: pem } };
}

test("ai_studio resolves the AI Studio endpoint with the API key and the bare model name", async () => {
  const target = await resolveProviderTarget({ GOOGLE_API_KEY: "k" }, "ai_studio", "gemini-3.8-flash");
  assert.equal(target.url, "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions");
  assert.equal(target.headers.Authorization, "Bearer k");
  assert.equal(target.model, "gemini-3.8-flash");
});

test("a missing provider defaults to ai_studio, and a missing key is a config error", async () => {
  await assert.rejects(resolveProviderTarget({}, undefined, "gemini-3.8-flash"), ProviderConfigError);
});

test("vertex uses the project/location endpoint, a publisher-prefixed model and a minted bearer token", async (t) => {
  clearProviderTokenCache();
  const { pair, sa } = await serviceAccount();
  let exchange: URLSearchParams | undefined;
  t.mock.method(globalThis, "fetch", async (url: string | URL | Request, init?: RequestInit) => {
    assert.equal(String(url), "https://oauth2.googleapis.com/token");
    exchange = init?.body as URLSearchParams;
    return Response.json({ access_token: "ya29.test", expires_in: 3600 });
  });

  const target = await resolveProviderTarget(
    { VERTEX_PROJECT: "axon-prod", VERTEX_LOCATION: "asia-south1", VERTEX_SERVICE_ACCOUNT_JSON: JSON.stringify(sa) },
    "vertex",
    "gemini-3.8-flash",
  );

  assert.equal(
    target.url,
    "https://asia-south1-aiplatform.googleapis.com/v1beta1/projects/axon-prod/locations/asia-south1/endpoints/openapi/chat/completions",
  );
  assert.equal(target.model, "google/gemini-3.8-flash");
  assert.equal(target.headers.Authorization, "Bearer ya29.test");

  // The assertion the token endpoint received is a genuine RS256 JWT for this account.
  assert.equal(exchange?.get("grant_type"), "urn:ietf:params:oauth:grant-type:jwt-bearer");
  const [h, c, sig] = String(exchange?.get("assertion")).split(".");
  const claims = JSON.parse(Buffer.from(b64urlToBytes(c)).toString());
  assert.equal(claims.iss, sa.client_email);
  assert.equal(claims.scope, "https://www.googleapis.com/auth/cloud-platform");
  assert.equal(claims.exp - claims.iat, 3600);
  const ok = await webcrypto.subtle.verify(
    "RSASSA-PKCS1-v1_5",
    pair.publicKey,
    toArrayBuffer(b64urlToBytes(sig)),
    toArrayBuffer(Buffer.from(`${h}.${c}`)),
  );
  assert.equal(ok, true);
});

test("the global Vertex location uses the unprefixed host, and the token is cached", async (t) => {
  clearProviderTokenCache();
  const { sa } = await serviceAccount();
  let exchanges = 0;
  t.mock.method(globalThis, "fetch", async () => {
    exchanges += 1;
    return Response.json({ access_token: "ya29.cached", expires_in: 3600 });
  });
  const env = { VERTEX_PROJECT: "p", VERTEX_SERVICE_ACCOUNT_JSON: JSON.stringify(sa) };
  const first = await resolveProviderTarget(env, "vertex", "gemini-3.8-flash");
  await resolveProviderTarget(env, "vertex", "gemini-3.8-flash");
  assert.match(first.url, /^https:\/\/aiplatform\.googleapis\.com\/v1beta1\/projects\/p\/locations\/global\//);
  assert.equal(exchanges, 1);
});

test("vertex without its configuration fails with a config error, never a silent AI Studio fallback", async () => {
  clearProviderTokenCache();
  await assert.rejects(resolveProviderTarget({ GOOGLE_API_KEY: "k" }, "vertex", "gemini-3.8-flash"), ProviderConfigError);
});

test("served model names are normalised across providers", () => {
  assert.equal(normalizeServedModel("google/gemini-3.8-flash"), "gemini-3.8-flash");
  assert.equal(normalizeServedModel("models/gemini-3.8-flash"), "gemini-3.8-flash");
  assert.equal(normalizeServedModel("gemini-3.8-flash"), "gemini-3.8-flash");
  assert.equal(normalizeServedModel(undefined), null);
});

test("signServiceAccountJwt rejects a malformed key", async () => {
  await assert.rejects(signServiceAccountJwt({ client_email: "x", private_key: "not a key" }, 0));
});
