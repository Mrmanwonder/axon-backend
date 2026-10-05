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
 * One engine at a time (owner, 6 Oct 2026). The first service answers; the
 * next is tried only when it throws or returns no usable evidence. A later
 * engine never runs alongside an earlier one. If every engine fails, the
 * first error is reported, so "no reliable source" is still said, never guessed.
 */
export class FallbackRetrievalService implements RetrievalService {
  constructor(readonly services: readonly RetrievalService[]) {}

  async retrieve(request: RetrievalRequest): Promise<Evidence[]> {
    let firstError: unknown = null;
    for (const service of this.services) {
      try {
        const evidence = (await service.retrieve(request))
          .filter((item) => !item.provenance.url || !isRestrictedSchemeUrl(item.provenance.url));
        if (evidence.length) return evidence;
      } catch (error) {
        firstError ??= error;
      }
    }
    if (firstError) throw firstError instanceof Error ? firstError : new Error("RETRIEVAL_FAILURE");
    return [];
  }
}
