import type { Evidence } from "../schemas";
import type { RetrievalRequest, RetrievalService } from "../intelligence/retrieval/types";
import { readBoundedJsonBody } from "../shared/bounded-json";
import { minimizePublicRetrievalQuery, oneWayHash } from "../intelligence/security/privacy";

interface TavilySearchResult { url: string; title?: string; content?: string; score?: number; published_date?: string }
interface TavilySearchResponse { results?: TavilySearchResult[] }
interface TavilyExtractResult { url: string; raw_content?: string }
interface TavilyExtractResponse { results?: TavilyExtractResult[] }

const OFFICIAL_ACADEMIC_DOMAINS = [
  "aqa.org.uk", "qualifications.pearson.com", "cambridgeinternational.org", "ocr.org.uk",
  "jcq.org.uk", "eduqas.co.uk", "wjec.co.uk", "ibo.org"
] as const;

const SIGNED_QUERY_KEYS = new Set([
  "sig", "signature", "token", "key", "api_key", "apikey", "access_token",
  "x-amz-signature", "x-amz-credential", "x-amz-security-token",
  "x-goog-signature", "x-goog-credential", "expires", "exp"
]);

function domainMatches(host: string, domain: string): boolean { return host === domain || host.endsWith(`.${domain}`); }

function privateOrReservedIpv4(host: string): boolean {
  if (!/^\d{1,3}(?:\.\d{1,3}){3}$/.test(host)) return false;
  const octets = host.split(".").map(Number);
  if (octets.some((value) => !Number.isInteger(value) || value < 0 || value > 255)) return true;
  const a = octets[0] ?? 999;
  const b = octets[1] ?? 999;
  return (
    a === 0 ||
    a === 10 ||
    a === 127 ||
    (a === 100 && b >= 64 && b <= 127) ||
    (a === 169 && b === 254) ||
    (a === 172 && b >= 16 && b <= 31) ||
    (a === 192 && (b === 0 || b === 168)) ||
    (a === 198 && (b === 18 || b === 19 || b === 51)) ||
    (a === 203 && b === 0) ||
    a >= 224
  );
}

/**
 * Tavily may only hand Axon ordinary public HTTPS pages.
 *
 * Reject credentials, local/private/reserved network targets, literal IPv6 and
 * signed/capability URLs before either authority classification or extraction.
 * This is a second boundary in addition to Tavily itself: retrieved content is
 * untrusted and may contain malicious or privacy-sensitive links.
 */
export function publicRetrievalUrl(raw: unknown): string | null {
  if (typeof raw !== "string" || raw.length === 0 || raw.length > 2_048) return null;
  try {
    const url = new URL(raw);
    if (url.protocol !== "https:" || url.username || url.password) return null;

    const host = url.hostname.toLowerCase().replace(/\.$/, "");
    if (
      !host ||
      host === "localhost" ||
      host.endsWith(".local") ||
      host.endsWith(".internal") ||
      host === "metadata.google.internal" ||
      host.includes(":") ||
      privateOrReservedIpv4(host)
    ) return null;

    url.hostname = host;
    for (const key of url.searchParams.keys()) {
      if (SIGNED_QUERY_KEYS.has(key.toLowerCase())) return null;
    }

    url.hash = "";
    return url.toString();
  } catch {
    return null;
  }
}

export function sourceAuthority(url: string, purpose: RetrievalRequest["purpose"]): Evidence["authority"] {
  const safe = publicRetrievalUrl(url);
  if (!safe) return "low";
  const host = new URL(safe).hostname.toLowerCase();
  const official = host.endsWith(".gov") || host.endsWith(".gov.uk") || OFFICIAL_ACADEMIC_DOMAINS.some((domain) => domainMatches(host, domain));
  if (official) return "primary";
  if (purpose === "official_rule") return "low";
  return host.endsWith(".edu") || host.endsWith(".ac.uk") ? "secondary" : "derived";
}

export class TavilyRetrievalService implements RetrievalService {
  constructor(readonly apiKey: string, readonly apiBase: string, readonly cache?: KVNamespace, readonly timeoutMs = 8_000) {}

  async retrieve(request: RetrievalRequest): Promise<Evidence[]> {
    const query = minimizePublicRetrievalQuery(request.query);
    const cacheIdentity = [request.purpose, query.toLowerCase(), String(request.maxSources ?? 5), request.requiredFreshness ?? "", [...(request.preferredDomains ?? [])].sort().join(",")].join("\u001f");
    const cacheKey = `retrieval:v2:${await oneWayHash(cacheIdentity)}`;
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
      .map((item) => {
        const url = publicRetrievalUrl(item.url);
        return url ? { ...item, url } : null;
      })
      .filter((item): item is TavilySearchResult => item !== null && (item.score ?? 0) >= 0.35)
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

    const candidateUrls = new Set(candidates.map((item) => item.url));
    const extractResponse = await fetch(`${this.apiBase}/extract`, {
      method: "POST",
      headers: { "content-type": "application/json", authorization: `Bearer ${this.apiKey}` },
      signal: AbortSignal.timeout(this.timeoutMs),
      body: JSON.stringify({ urls: [...candidateUrls], extract_depth: "advanced", query, chunks_per_source: 3, format: "markdown", include_images: false })
    });
    if (!extractResponse.ok) throw new Error(`RETRIEVAL_FAILURE extract ${extractResponse.status}`);
    const extract = await readBoundedJsonBody<TavilyExtractResponse>(extractResponse.body, 5_000_000);
    const extracted = new Map(
      (extract.results ?? []).flatMap((item) => {
        const url = publicRetrievalUrl(item.url);
        return url && candidateUrls.has(url) ? [[url, item.raw_content ?? ""] as const] : [];
      })
    );

    const retrievedAt = new Date().toISOString();
    const evidence = candidates.map((item, index): Evidence => {
      const authority = sourceAuthority(item.url, request.purpose);
      return {
        id: `retrieval_${index}_${crypto.randomUUID()}`,
        informationClass: "VERIFIED_EXTERNAL",
        source: authority === "primary" ? "official_source" : "retrieval",
        authority,
        value: { title: item.title ?? item.url, content: extracted.get(item.url) || item.content || "", relevance: item.score ?? 0 },
        provenance: { url: item.url, retrievedAt, ...(item.published_date ? { publishedAt: item.published_date } : {}) },
        verification: authority === "primary" || authority === "secondary" ? "verified" : "probable"
      };
    });
    await this.cache?.put(cacheKey, JSON.stringify(evidence), { expirationTtl: request.purpose === "current_fact" ? 900 : 3_600 });
    return evidence;
  }
}
