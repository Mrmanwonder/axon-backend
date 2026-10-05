import { afterEach, describe, expect, it, vi } from "vitest";
import { CombinedRetrievalService, FirecrawlRetrievalService } from "../src/providers/firecrawl";
import type { Evidence } from "../src/schemas";
import type { RetrievalService } from "../src/intelligence/retrieval/types";

afterEach(() => vi.unstubAllGlobals());

const page = (url: string, extra: Record<string, unknown> = {}) => ({ url, title: url, description: "d", markdown: "body text", ...extra });

describe("FirecrawlRetrievalService", () => {
  it("returns authority-ranked evidence and drops mark-scheme sources", async () => {
    const fetchMock = vi.fn(async () => new Response(JSON.stringify({ success: true, data: { web: [
      page("https://bestexamhelp.com/exam/x/9231_w25_ms_11.pdf"),
      page("https://example.com/notes"),
      page("https://www.cambridgeinternational.org/programmes/9231"),
    ] } }), { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);
    const out = await new FirecrawlRetrievalService("k").retrieve({ query: "Cambridge 9231 Further Mathematics syllabus", purpose: "current_fact", maxSources: 5 });
    expect(out.map((e) => e.provenance.url)).toEqual(["https://www.cambridgeinternational.org/programmes/9231", "https://example.com/notes"]);
    expect(out[0]).toMatchObject({ source: "official_source", authority: "primary", verification: "verified" });
    const body = JSON.parse((fetchMock.mock.calls[0] as unknown as [string, RequestInit])[1].body as string);
    expect(body.query).toBe("Cambridge 9231 Further Mathematics syllabus");
  });

  it("refuses to send private text as a query", async () => {
    vi.stubGlobal("fetch", vi.fn());
    await expect(new FirecrawlRetrievalService("k").retrieve({ query: "my answer was 4 student@example.com", purpose: "current_fact" })).rejects.toThrow();
    expect(fetch).not.toHaveBeenCalled();
  });

  it("an official-rule question with no primary source fails rather than guessing", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify({ data: { web: [page("https://blog.example.com/x")] } }), { status: 200 })));
    await expect(new FirecrawlRetrievalService("k").retrieve({ query: "CBSE exam rules", purpose: "official_rule" })).rejects.toThrow(/no authoritative source/);
  });
});

const ev = (url: string, authority: Evidence["authority"]): Evidence => ({
  id: url, informationClass: "VERIFIED_EXTERNAL", source: authority === "primary" ? "official_source" : "retrieval", authority,
  value: { title: url }, provenance: { url }, verification: authority === "derived" ? "probable" : "verified",
});
const svc = (impl: () => Promise<Evidence[]>): RetrievalService => ({ retrieve: impl });

describe("CombinedRetrievalService", () => {
  it("merges both engines by URL and keeps the higher-authority copy", async () => {
    const combined = new CombinedRetrievalService([
      svc(async () => [ev("https://a.org/x", "derived"), ev("https://b.gov/y", "primary")]),
      svc(async () => [ev("https://a.org/x/", "secondary"), ev("https://c.edu/z", "secondary")]),
    ]);
    const out = await combined.retrieve({ query: "q", purpose: "current_fact", maxSources: 5 });
    expect(out.map((e) => [e.provenance.url, e.authority])).toEqual([
      ["https://b.gov/y", "primary"], ["https://a.org/x/", "secondary"], ["https://c.edu/z", "secondary"],
    ]);
  });

  it("one engine failing still answers from the other", async () => {
    const combined = new CombinedRetrievalService([svc(async () => { throw new Error("RETRIEVAL_FAILURE tavily 503"); }), svc(async () => [ev("https://b.gov/y", "primary")])]);
    expect(await combined.retrieve({ query: "q", purpose: "current_fact" })).toHaveLength(1);
  });

  it("all engines failing is reported, never papered over", async () => {
    const combined = new CombinedRetrievalService([svc(async () => { throw new Error("RETRIEVAL_FAILURE a"); }), svc(async () => { throw new Error("RETRIEVAL_FAILURE b"); })]);
    await expect(combined.retrieve({ query: "q", purpose: "current_fact" })).rejects.toThrow("RETRIEVAL_FAILURE a");
  });
});
