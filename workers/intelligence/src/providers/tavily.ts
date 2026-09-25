import type { Evidence } from "../schemas";
import type { RetrievalRequest, RetrievalService } from "../intelligence/retrieval/types";
import { readBoundedJsonBody } from "../shared/bounded-json";
import { oneWayHash } from "../intelligence/security/privacy";

interface TavilySearchResult { url: string; title?: string; content?: string; score?: number; published_date?: string }
interface TavilySearchResponse { results?: TavilySearchResult[] }
interface TavilyExtractResult { url: string; raw_content?: string }
interface TavilyExtractResponse { results?: TavilyExtractResult[] }

const OFFICIAL_ACADEMIC_DOMAINS = [
  "aqa.org.uk", "qualifications.pearson.com", "cambridgeinternational.org", "ocr.org.uk",
  "jcq.org.uk", "eduqas.co.uk", "wjec.co.uk", "ibo.org"
] as const;

function domainMatches(host: string, domain: string): boolean { return host === domain || host.endsWith(`.${domain}`); }

export function sourceAuthority(url: string, purpose: RetrievalRequest["purpose"]): Evidence["authority"] {
  const host = new URL(url).hostname.toLowerCase();
  const official = host.endsWith(".gov") || host.endsWith(".gov.uk") || OFFICIAL_ACADEMIC_DOMAINS.some((domain) => domainMatches(host, domain));
  if (official) return "primary";
  if (purpose === "official_rule") return "low";
  return host.endsWith(".edu") || host.endsWith(".ac.uk") ? "secondary" : "derived";
}

export class TavilyRetrievalService implements RetrievalService {
  constructor(readonly apiKey: string, readonly apiBase: string, readonly cache?: KVNamespace, readonly timeoutMs = 8_000) {}

  async retrieve(request: RetrievalRequest): Promise<Evidence[]> {
    const query = request.query.replace(/\s+/g, " ").trim().slice(0, 400);
    if (!query) throw new Error("RETRIEVAL_FAILURE empty query");
    const cacheIdentity = [request.purpose, query.toLowerCase(), String(request.maxSources ?? 5), request.requiredFreshness ?? "", [...(request.preferredDomains ?? [])].sort().join(",")].join("\u001f");
    const cacheKey = `retrieval:v1:${await oneWayHash(cacheIdentity)}`;
    const cached = await this.cache?.get<Evidence[]>(cacheKey, "json");
    if (cached) return cached;
    const maximum = Math.max(1, Math.min(20, request.maxSources ?? 5));
    const searchResponse = await fetch(`${this.apiBase}/search`, {
      method: "POST",
      headers: { "content-type": "application/json", authorization: `Bearer ${this.apiKey}` },
      signal: AbortSignal.timeout(this.timeoutMs),
      body: JSON.stringify({
        query, search_depth: "advanced", chunks_per_source: 3, max_results: maximum,
        include_domains: request.preferredDomains ?? [], include_answer: false,
        include_raw_content: false, include_images: false, include_published_date: true,
        safe_search: true, ...(request.requiredFreshness ? { start_date: request.requiredFreshness.slice(0, 10) } : {})
      })
    });
    if (!searchResponse.ok) throw new Error(`RETRIEVAL_FAILURE search ${searchResponse.status}`);
    const search = await readBoundedJsonBody<TavilySearchResponse>(searchResponse.body, 2_000_000);
    const candidates = (search.results ?? [])
      .filter((item) => item.url.startsWith("https://") && (item.score ?? 0) >= 0.35)
      .filter((item) => !request.requiredFreshness || Boolean(item.published_date && Date.parse(item.published_date) >= Date.parse(request.requiredFreshness)))
      .sort((left, right) => {
        const rank = { primary: 3, secondary: 2, derived: 1, low: 0 } as const;
        return rank[sourceAuthority(right.url, request.purpose)] - rank[sourceAuthority(left.url, request.purpose)] || (right.score ?? 0) - (left.score ?? 0);
      })
      .slice(0, maximum);
    if (request.purpose === "official_rule" && !candidates.some((item) => sourceAuthority(item.url, request.purpose) === "primary")) {
      throw new Error("RETRIEVAL_FAILURE no authoritative source");
    }
    if (candidates.length === 0) return [];
    const extractResponse = await fetch(`${this.apiBase}/extract`, {
      method: "POST",
      headers: { "content-type": "application/json", authorization: `Bearer ${this.apiKey}` },
      signal: AbortSignal.timeout(this.timeoutMs),
      body: JSON.stringify({ urls: candidates.map((item) => item.url), extract_depth: "advanced", query, chunks_per_source: 3, format: "markdown", include_images: false })
    });
    if (!extractResponse.ok) throw new Error(`RETRIEVAL_FAILURE extract ${extractResponse.status}`);
    const extract = await readBoundedJsonBody<TavilyExtractResponse>(extractResponse.body, 5_000_000);
    const extracted = new Map((extract.results ?? []).map((item) => [item.url, item.raw_content ?? ""]));
    const retrievedAt = new Date().toISOString();
    const evidence = candidates.map((item, index): Evidence => ({
      id: `retrieval_${index}_${crypto.randomUUID()}`,
      informationClass: "VERIFIED_EXTERNAL",
      source: sourceAuthority(item.url, request.purpose) === "primary" ? "official_source" : "retrieval",
      authority: sourceAuthority(item.url, request.purpose),
      value: { title: item.title ?? item.url, content: extracted.get(item.url) || item.content || "", relevance: item.score ?? 0 },
      provenance: { url: item.url, retrievedAt, ...(item.published_date ? { publishedAt: item.published_date } : {}) },
      verification: sourceAuthority(item.url, request.purpose) === "primary" || sourceAuthority(item.url, request.purpose) === "secondary" ? "verified" : "probable"
    }));
    await this.cache?.put(cacheKey, JSON.stringify(evidence), { expirationTtl: request.purpose === "current_fact" ? 900 : 3_600 });
    return evidence;
  }
}
