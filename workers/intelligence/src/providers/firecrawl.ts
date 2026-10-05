import type { Evidence } from "../schemas";
import type { RetrievalRequest, RetrievalService } from "../intelligence/retrieval/types";
import { readBoundedJsonBody } from "../shared/bounded-json";
import { minimizePublicRetrievalQuery } from "../intelligence/security/privacy";
import { isRestrictedSchemeUrl } from "@mastery/shared/web_sources.js";
import { publicRetrievalUrl, sourceAuthority } from "./tavily";

interface FirecrawlWebRow { url?: string; title?: string; description?: string; markdown?: string }
interface FirecrawlSearchResponse { success?: boolean; data?: { web?: FirecrawlWebRow[] } | FirecrawlWebRow[] }

const RANK = { primary: 3, secondary: 2, derived: 1, low: 0 } as const;
const MAX_CONTENT = 6_000;

/**
 * Firecrawl search + page content in one call (scrapeOptions), under the same
 * boundary as the Tavily adapter: a minimised server-authored query, public
 * HTTPS pages only, authority-ranked, and no mark-scheme sources (hard rule 2).
 */
export class FirecrawlRetrievalService implements RetrievalService {
  constructor(readonly apiKey: string, readonly apiBase = "https://api.firecrawl.dev/v2", readonly timeoutMs = 8_000) {}

  async retrieve(request: RetrievalRequest): Promise<Evidence[]> {
    const query = minimizePublicRetrievalQuery(request.query);
    const maximum = Math.max(1, Math.min(10, request.maxSources ?? 5));
    const response = await fetch(`${this.apiBase}/search`, {
      method: "POST",
      headers: { "content-type": "application/json", authorization: `Bearer ${this.apiKey}` },
      signal: AbortSignal.timeout(this.timeoutMs),
      body: JSON.stringify({
        query, limit: maximum, sources: ["web"], timeout: this.timeoutMs,
        ...(request.preferredDomains?.length ? { includeDomains: request.preferredDomains } : {}),
        scrapeOptions: { formats: ["markdown"], onlyMainContent: true },
      }),
    });
    if (!response.ok) throw new Error(`RETRIEVAL_FAILURE firecrawl ${response.status}`);
    const body = await readBoundedJsonBody<FirecrawlSearchResponse>(response.body, 5_000_000);
    const rows = Array.isArray(body.data) ? body.data : body.data?.web ?? [];
    const candidates = rows
      .flatMap((row) => {
        const url = publicRetrievalUrl(row.url);
        return url && !isRestrictedSchemeUrl(url) ? [{ ...row, url }] : [];
      })
      .sort((a, b) => RANK[sourceAuthority(b.url, request.purpose)] - RANK[sourceAuthority(a.url, request.purpose)])
      .slice(0, maximum);
    if (request.purpose === "official_rule" && !candidates.some((row) => sourceAuthority(row.url, request.purpose) === "primary")) {
      throw new Error("RETRIEVAL_FAILURE no authoritative source");
    }
    const retrievedAt = new Date().toISOString();
    return candidates.map((row, index): Evidence => {
      const authority = sourceAuthority(row.url, request.purpose);
      return {
        id: `retrieval_fc_${index}_${crypto.randomUUID()}`,
        informationClass: "VERIFIED_EXTERNAL",
        source: authority === "primary" ? "official_source" : "retrieval",
        authority,
        value: { title: row.title ?? row.url, content: (row.markdown || row.description || "").slice(0, MAX_CONTENT), engine: "firecrawl" },
        provenance: { url: row.url, retrievedAt },
        verification: authority === "primary" || authority === "secondary" ? "verified" : "probable",
      };
    });
  }
}

/**
 * Runs every configured engine in parallel and merges by URL, keeping the
 * higher-authority copy. One engine failing is tolerated; all failing throws
 * the first error, so "no reliable source" is still reported, never guessed.
 */
export class CombinedRetrievalService implements RetrievalService {
  constructor(readonly services: readonly RetrievalService[]) {}

  async retrieve(request: RetrievalRequest): Promise<Evidence[]> {
    const settled = await Promise.allSettled(this.services.map((service) => service.retrieve(request)));
    const ok = settled.flatMap((item) => item.status === "fulfilled" ? [item.value] : []);
    if (!ok.length) {
      const first = settled.find((item): item is PromiseRejectedResult => item.status === "rejected");
      throw first?.reason instanceof Error ? first.reason : new Error("RETRIEVAL_FAILURE");
    }
    const byUrl = new Map<string, Evidence>();
    for (const item of ok.flat()) {
      if (item.provenance.url && isRestrictedSchemeUrl(item.provenance.url)) continue;
      const key = (item.provenance.url ?? item.id).replace(/\/$/, "").toLowerCase();
      const existing = byUrl.get(key);
      if (!existing || RANK[item.authority] > RANK[existing.authority]) byUrl.set(key, item);
    }
    const maximum = Math.max(1, Math.min(20, request.maxSources ?? 5));
    return [...byUrl.values()].sort((a, b) => RANK[b.authority] - RANK[a.authority]).slice(0, maximum);
  }
}
