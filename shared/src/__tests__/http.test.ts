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
