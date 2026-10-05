import { test, afterEach } from "node:test";
import assert from "node:assert/strict";
import { isRestrictedSchemeUrl, mergeWebResults, firecrawlSearch } from "../web_sources.js";
import { runTavilyTool, publicWebUrl } from "../tavily.js";

const realFetch = globalThis.fetch;
afterEach(() => { globalThis.fetch = realFetch; });

test("mark-scheme mirrors and scheme-shaped files never reach a model (hard rule 2)", () => {
  for (const url of [
    "https://bestexamhelp.com/exam/cambridge-international-a-level/mathematics-further-9231/2025/9231_w25_ms_11.pdf",
    "https://pastpapers.papacambridge.com/directories/CAIE/9709_s24_ms_12.pdf",
    "https://www.physicsandmathstutor.com/download/Maths/A-level/9709_w23_ms_32.pdf",
    "https://example.org/Edexcel/IAL/Mark%20Scheme%20(Results)%20June%202024.pdf",
    "https://example.org/files/markscheme-paper1.pdf",
    "https://cbseacademic.nic.in/web_material/SQP/ClassXII_2026_27/Physics-MS.pdf",
    "not a url",
  ]) assert.equal(isRestrictedSchemeUrl(url), true, url);
});

test("ordinary teaching pages stay allowed", () => {
  for (const url of [
    "https://www.cambridgeinternational.org/programmes-and-qualifications/cambridge-international-as-and-a-level-mathematics-9709/",
    "https://www.physicsandmathstutor.com/maths-revision/a-level-mechanics/",
    "https://en.wikipedia.org/wiki/Simple_harmonic_motion",
    "https://www.khanacademy.org/math/calculus-1",
  ]) assert.equal(isRestrictedSchemeUrl(url), false, url);
});

test("merge interleaves engines, dedupes by URL and caps the list", () => {
  const a = [{ title: "A1", url: "https://x.org/1", content: "" }, { title: "A2", url: "https://x.org/2", content: "" }];
  const b = [{ title: "B1", url: "https://x.org/1/", content: "" }, { title: "B2", url: "https://y.org/3", content: "" }];
  assert.deepEqual(mergeWebResults(a, b, 5).map((r) => r.title), ["A1", "A2", "B2"]);
  assert.equal(mergeWebResults(a, b, 2).length, 2);
});

test("firecrawl search filters scheme mirrors and reads v2 data.web", async () => {
  globalThis.fetch = (async () => new Response(JSON.stringify({ success: true, data: { web: [
    { url: "https://bestexamhelp.com/x/9231_w25_ms_11.pdf", title: "MS", description: "scheme" },
    { url: "https://en.wikipedia.org/wiki/Matrix", title: "Matrix", description: "A matrix is..." },
  ] } }), { status: 200 })) as typeof fetch;
  const rows = await firecrawlSearch({ FIRECRAWL_API_KEY: "k" } as any, "matrices", { limit: 5, publicUrl: publicWebUrl });
  assert.deepEqual(rows.map((r) => r.url), ["https://en.wikipedia.org/wiki/Matrix"]);
});

test("web_search uses Tavily alone when it answers", async () => {
  const hosts: string[] = [];
  globalThis.fetch = (async (input: RequestInfo | URL) => {
    hosts.push(new URL(String(input)).hostname);
    return new Response(JSON.stringify({ results: [
      { url: "https://t.org/a", title: "T", content: "t" },
      { url: "https://bestexamhelp.com/9231_w25_ms_11.pdf", title: "MS", content: "ms" },
    ] }), { status: 200 });
  }) as typeof fetch;
  const out = await runTavilyTool({ TAVILY_API_KEY: "t", FIRECRAWL_API_KEY: "f" } as any,
    { function: { name: "web_search", arguments: "{}" } }, { searchContext: "Mathematics matrices" });
  assert.deepEqual(hosts, ["api.tavily.com"]);
  assert.deepEqual(out.sources, ["https://t.org/a"]);
});

test("web_search falls back to Firecrawl only when Tavily fails", async () => {
  const hosts: string[] = [];
  globalThis.fetch = (async (input: RequestInfo | URL) => {
    hosts.push(new URL(String(input)).hostname);
    return String(input).includes("tavily")
      ? new Response("down", { status: 503 })
      : new Response(JSON.stringify({ data: { web: [{ url: "https://f.org/b", title: "F", description: "f" }] } }), { status: 200 });
  }) as typeof fetch;
  const out = await runTavilyTool({ TAVILY_API_KEY: "t", FIRECRAWL_API_KEY: "f" } as any,
    { function: { name: "web_search", arguments: "{}" } }, { searchContext: "Physics waves" });
  assert.deepEqual(hosts, ["api.tavily.com", "api.firecrawl.dev"]);
  assert.deepEqual(out.sources, ["https://f.org/b"]);
});

test("web_search falls back when Tavily's only results are filtered out", async () => {
  globalThis.fetch = (async (input: RequestInfo | URL) => String(input).includes("tavily")
    ? new Response(JSON.stringify({ results: [{ url: "https://bestexamhelp.com/9231_w25_ms_11.pdf", title: "MS", content: "ms" }] }), { status: 200 })
    : new Response(JSON.stringify({ data: { web: [{ url: "https://f.org/b", title: "F", description: "f" }] } }), { status: 200 })) as typeof fetch;
  const out = await runTavilyTool({ TAVILY_API_KEY: "t", FIRECRAWL_API_KEY: "f" } as any,
    { function: { name: "web_search", arguments: "{}" } }, { searchContext: "Physics waves" });
  assert.deepEqual(out.sources, ["https://f.org/b"]);
});

test("web_extract falls back to a Firecrawl scrape for pages Tavily could not read", async () => {
  globalThis.fetch = (async (input: RequestInfo | URL) => String(input).includes("tavily")
    ? new Response(JSON.stringify({ results: [] }), { status: 200 })
    : new Response(JSON.stringify({ data: { markdown: "# Waves\nA wave transfers energy." } }), { status: 200 })) as typeof fetch;
  const out = await runTavilyTool({ TAVILY_API_KEY: "t", FIRECRAWL_API_KEY: "f" } as any,
    { function: { name: "web_extract", arguments: JSON.stringify({ urls: ["https://f.org/b"] }) } },
    { searchContext: "Physics waves", allowedUrls: ["https://f.org/b"] });
  assert.match(out.content, /transfers energy/);
});

test("web_extract never opens a scheme URL even if it was somehow allowed", async () => {
  let called = false;
  globalThis.fetch = (async () => { called = true; return new Response("{}", { status: 200 }); }) as typeof fetch;
  const ms = "https://bestexamhelp.com/9231_w25_ms_11.pdf";
  const out = await runTavilyTool({ TAVILY_API_KEY: "t", FIRECRAWL_API_KEY: "f" } as any,
    { function: { name: "web_extract", arguments: JSON.stringify({ urls: [ms] }) } }, { searchContext: "x", allowedUrls: [ms] });
  assert.equal(called, false);
  assert.match(out.content, /only accepts public URLs/);
});
