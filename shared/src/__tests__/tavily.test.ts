import { test } from "node:test";
import assert from "node:assert/strict";
import { TAVILY_TOOLS, publicWebUrl } from "../tavily.js";

test("Tavily tool declarations expose search and extract only", () => {
  assert.deepEqual(
    TAVILY_TOOLS.map((tool) => tool.function.name),
    ["web_search", "web_extract"],
  );
});

test("publicWebUrl accepts public http(s) URLs", () => {
  assert.equal(publicWebUrl("https://example.com/a?b=1"), "https://example.com/a?b=1");
  assert.equal(publicWebUrl("http://example.org/"), "http://example.org/");
});

test("publicWebUrl rejects local and private-network URLs", () => {
  for (const url of [
    "http://localhost:8787/",
    "http://127.0.0.1/",
    "http://10.0.0.2/",
    "http://172.16.0.2/",
    "http://192.168.1.2/",
    "http://169.254.1.1/",
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
