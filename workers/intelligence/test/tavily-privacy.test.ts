import { afterEach, describe, expect, it, vi } from "vitest";
import { publicRetrievalUrl, TavilyRetrievalService } from "../src/providers/tavily";

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("Tavily public retrieval privacy boundary", () => {
  it("rejects local, private, credentialed, literal IPv6 and signed capability URLs", () => {
    const rejected = [
      "http://example.com/plain-http",
      "https://localhost/private",
      "https://localhost./private",
      "https://service.internal/path",
      "https://service.internal./path",
      "https://metadata.google.internal/computeMetadata/v1/",
      "https://metadata.google.internal./computeMetadata/v1/",
      "https://10.0.0.1/private",
      "https://127.0.0.1/private",
      "https://169.254.169.254/latest/meta-data",
      "https://172.16.0.1/private",
      "https://192.168.1.10/private",
      "https://[::1]/private",
      "https://user:password@example.com/private",
      "https://example.com/file?sig=secret",
      "https://example.com/file?X-Amz-Signature=secret",
      "https://example.com/file?token=secret",
      "https://example.com/file?expires=9999999999",
    ];
    for (const url of rejected) expect(publicRetrievalUrl(url), url).toBeNull();

    expect(publicRetrievalUrl("https://www.gov.uk/guidance/example#section"))
      .toBe("https://www.gov.uk/guidance/example");
  });

  it("serializes only validated public URLs into Tavily extract", async () => {
    const bodies: Array<Record<string, unknown>> = [];
    const fetchMock = vi.fn(async (_url: string | URL | Request, init?: RequestInit) => {
      bodies.push(JSON.parse(String(init?.body ?? "{}")) as Record<string, unknown>);
      if (bodies.length === 1) {
        return new Response(JSON.stringify({
          results: [
            { url: "https://www.gov.uk/guidance/example", title: "Official", content: "summary", score: 0.95 },
            { url: "https://127.0.0.1/private", title: "Local", content: "private", score: 0.99 },
            { url: "https://example.com/file?sig=secret", title: "Signed", content: "private", score: 0.99 },
            { url: "https://user:password@example.com/private", title: "Credentials", content: "private", score: 0.99 },
          ],
        }), { status: 200, headers: { "content-type": "application/json" } });
      }
      return new Response(JSON.stringify({
        results: [
          { url: "https://www.gov.uk/guidance/example", raw_content: "official extracted text" },
          { url: "https://127.0.0.1/private", raw_content: "must never be trusted" },
        ],
      }), { status: 200, headers: { "content-type": "application/json" } });
    });
    vi.stubGlobal("fetch", fetchMock);

    const service = new TavilyRetrievalService("tavily-key", "https://api.tavily.com");
    const evidence = await service.retrieve({
      query: "CBSE Class XII Physics official syllabus specification",
      purpose: "current_fact",
      maxSources: 5,
    });

    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(bodies[0]).toMatchObject({
      query: "CBSE Class XII Physics official syllabus specification",
      include_raw_content: false,
      include_images: false,
    });
    expect(bodies[1]?.urls).toEqual(["https://www.gov.uk/guidance/example"]);
    expect(JSON.stringify(bodies[1])).not.toContain("127.0.0.1");
    expect(JSON.stringify(bodies[1])).not.toContain("sig=secret");
    expect(evidence).toHaveLength(1);
    expect(evidence[0]?.provenance.url).toBe("https://www.gov.uk/guidance/example");
    expect(evidence[0]?.value).toMatchObject({ content: "official extracted text" });
  });

  it("blocks sensitive query material before any outbound Tavily request", async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
    const service = new TavilyRetrievalService("tavily-key", "https://api.tavily.com");

    await expect(service.retrieve({
      query: "CBSE Physics alice@example.com",
      purpose: "current_fact",
    })).rejects.toThrow("RETRIEVAL_BLOCKED_SENSITIVE_QUERY");
    await expect(service.retrieve({
      query: "CBSE Physics https://private.example/file?sig=secret",
      purpose: "current_fact",
    })).rejects.toThrow("RETRIEVAL_BLOCKED_SENSITIVE_QUERY");
    expect(fetchMock).not.toHaveBeenCalled();
  });
});
