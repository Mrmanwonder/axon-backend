import { test } from "node:test";
import assert from "node:assert/strict";
import {
  TAVILY_TOOLS,
  filterExtractUrls,
  normaliseSearchContext,
  publicWebUrl,
} from "../tavily.js";

test("Tavily exposes only bounded search and extract tools", () => {
  assert.deepEqual(
    TAVILY_TOOLS.map((tool) => tool.function.name),
    ["web_search", "web_extract"],
  );

  const search = TAVILY_TOOLS[0].function.parameters;
  assert.equal("query" in search.properties, false, "the model must not control the Tavily query");
});

test("approved search context is bounded and redacts obvious contact/account tokens", () => {
  const value = normaliseSearchContext(
    "Biology photosynthesis student@example.com +91 98765 43210 123e4567-e89b-12d3-a456-426614174000",
  );
  assert.match(value, /^Biology photosynthesis/);
  assert.doesNotMatch(value, /student@example\.com/);
  assert.doesNotMatch(value, /98765/);
  assert.doesNotMatch(value, /123e4567/);
  assert.ok(value.length <= 400);
});

test("publicWebUrl accepts normal public http(s) URLs", () => {
  assert.equal(publicWebUrl("https://example.com/a?b=1"), "https://example.com/a?b=1");
  assert.equal(publicWebUrl("http://example.org/"), "http://example.org/");
});

test("publicWebUrl rejects local, reserved and literal-IP escape hatches", () => {
  for (const url of [
    "http://localhost:8787/",
    "http://127.0.0.1/",
    "http://10.0.0.2/",
    "http://100.64.0.1/",
    "http://172.16.0.2/",
    "http://192.168.1.2/",
    "http://169.254.169.254/latest/meta-data/",
    "http://192.0.2.1/",
    "http://198.51.100.2/",
    "http://203.0.113.2/",
    "http://224.0.0.1/",
    "http://[::1]/",
    "http://metadata.google.internal/",
    "file:///tmp/a",
  ]) {
    assert.equal(publicWebUrl(url), null, url);
  }
});

test("publicWebUrl strips embedded credentials", () => {
  assert.equal(
    publicWebUrl("https://user:pass@example.com/path"),
    "https://example.com/path",
  );
});

test("extract is limited to URLs discovered by the same search", () => {
  assert.deepEqual(
    filterExtractUrls(
      ["https://example.com/a", "https://attacker.example/b", "http://127.0.0.1/"],
      ["https://example.com/a"],
    ),
    ["https://example.com/a"],
  );
});
