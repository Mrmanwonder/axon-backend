import { test } from "node:test";
import assert from "node:assert/strict";
import { corsFor, withCors } from "../http.js";

test("CORS accepts both production Axon hostnames", () => {
  for (const origin of ["https://axonstudy.online", "https://www.axonstudy.online"]) {
    const headers = corsFor(new Request("https://api.example/paper-submit", {
      headers: { Origin: origin },
    }));
    assert.equal(headers["Access-Control-Allow-Origin"], origin);
    assert.equal(headers.Vary, "Origin");
  }
});

test("CORS does not reflect untrusted origins", () => {
  const headers = corsFor(new Request("https://api.example/paper-submit", {
    headers: { Origin: "https://evil.example" },
  }));
  assert.equal(headers["Access-Control-Allow-Origin"], "https://axonstudy.online");
});

test("withCors replaces a stale static origin on a response", async () => {
  const req = new Request("https://api.example/paper-submit", {
    headers: { Origin: "https://www.axonstudy.online" },
  });
  const res = withCors(req, new Response("ok", {
    headers: { "Access-Control-Allow-Origin": "https://axonstudy.online" },
  }));
  assert.equal(res.headers.get("Access-Control-Allow-Origin"), "https://www.axonstudy.online");
  assert.equal(await res.text(), "ok");
});

import { readBoundedJson, JsonBodyError } from "../http-body.js";
const body = (text: string, headers: Record<string, string> = {}) => new Request("https://example.test", {
  method: "POST", body: text, headers: { "content-type": "application/json", ...headers },
});
test("JSON boundary accepts bounded metadata and rejects malformed/deep invalid input", async () => {
  assert.deepEqual(await readBoundedJson(body('{"ok":true}')), { ok: true });
  assert.equal(await readBoundedJson(body('{broken')), null);
  assert.equal(await readBoundedJson(body('['.repeat(1000))), null);
  assert.deepEqual(await readBoundedJson(body('{"ok":true}', { "content-type": "application/problem+json; charset=utf-8" })), { ok: true });
});
test("byte cap cannot be bypassed by missing/spoofed length or multibyte text", async () => {
  for (const headers of [{}, { "content-length": "1" }, { "content-length": "1000" }] as Record<string, string>[]) {
    await assert.rejects(readBoundedJson(body('"éééé"', headers), 8), (e: unknown) => e instanceof JsonBodyError && e.status === 413);
  }
});
test("wrong media type is a predictable 415", async () => {
  await assert.rejects(readBoundedJson(body('{}', { "content-type": "text/plain" })), (e: unknown) => e instanceof JsonBodyError && e.status === 415);
});
test("a stalled body times out and cancels the input stream", async () => {
  let cancelled = false;
  const stream = new ReadableStream<Uint8Array>({ cancel() { cancelled = true; } });
  const req = new Request("https://example.test", { method: "POST", headers: { "content-type": "application/json" }, body: stream, duplex: "half" } as RequestInit);
  await assert.rejects(readBoundedJson(req, 32, 5), (e: unknown) => e instanceof JsonBodyError && e.status === 408);
  assert.equal(cancelled, true);
});
test("oversized chunked body stops consumption and cancels", async () => {
  let cancelled = false;
  let pulls = 0;
  const stream = new ReadableStream<Uint8Array>({ pull(c) { pulls++; c.enqueue(new Uint8Array(8)); }, cancel() { cancelled = true; } });
  const req = new Request("https://example.test", { method: "POST", headers: { "content-type": "application/json" }, body: stream, duplex: "half" } as RequestInit);
  await assert.rejects(readBoundedJson(req, 10), (e: unknown) => e instanceof JsonBodyError && e.status === 413);
  assert.equal(cancelled, true); assert.ok(pulls <= 3);
});
