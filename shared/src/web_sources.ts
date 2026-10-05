// Firecrawl as a second public-web retriever beside Tavily, plus the source
// filter both share.
//
// Same privacy boundary as tavily.ts: the query is server-authored public
// academic context, never student text, and only public HTTPS pages are ever
// fetched. Results from either engine pass through `isRestrictedSchemeUrl`
// before a model sees them.
//
// Why the filter exists (hard rule 2): Cambridge and Pearson mark schemes may
// not be reproduced or used, and mirrors index them by predictable filenames
// (e.g. 9231_w25_ms_11.pdf). A web search seeded with question text can land on
// one. Official CBSE schemes are the only permitted Tier-2 source and reach the
// model only through the reviewed registry, so no mark-scheme URL is accepted
// from open-web search at all.

import type { Env } from "./env.js";

const FIRECRAWL_API = "https://api.firecrawl.dev/v2";
const FIRECRAWL_TIMEOUT_MS = 15_000;

/** Past-paper mirrors whose purpose is redistributing board mark schemes. */
// Revision sites that also carry notes (PMT, ZNotes, Save My Exams) are not
// blocked by host; their scheme files are caught by the path patterns below.
const SCHEME_MIRROR_HOSTS = [
  "bestexamhelp.com", "papacambridge.com", "xtremepapers.com", "gceguide.com", "gceguide.cc",
  "pastpapers.co", "dynamicpapers.com", "freeexampapers.com",
] as const;

// Cambridge naming: <syllabus>_<series><yy>_ms_<component>.pdf; Pearson: "mark scheme"/"ms" files.
const SCHEME_PATH_PATTERNS = [
  /(?:^|[\/_.-])\d{4}_[msw]\d{2}_ms_\d{1,2}\b/i,
  /mark[\s_%20-]*schemes?/i,
  /(?:^|[\/_-])ms(?:[_-]\d{1,2})?\.pdf$/i,
  /[\/_-]markscheme/i,
];

function hostMatches(host: string, domain: string): boolean {
  return host === domain || host.endsWith(`.${domain}`);
}

/**
 * True when a URL must not reach a model: a known scheme mirror, or any
 * mark-scheme-shaped document. CBSE's own first-party host is not exempt here
 * either — CBSE schemes enter only through the reviewed scheme registry.
 */
export function isRestrictedSchemeUrl(raw: unknown): boolean {
  if (typeof raw !== "string") return true;
  let url: URL;
  try { url = new URL(raw); } catch { return true; }
  const host = url.hostname.toLowerCase().replace(/^www\./, "");
  if (SCHEME_MIRROR_HOSTS.some((domain) => hostMatches(host, domain))) return true;
  const path = decodeURIComponent(url.pathname);
  return SCHEME_PATH_PATTERNS.some((pattern) => pattern.test(path));
}

export interface WebResult { title: string; url: string; content: string }

type Json = Record<string, unknown>;

function clean(value: unknown, limit: number): string {
  return typeof value === "string" ? value.replace(/\s+/g, " ").trim().slice(0, limit) : "";
}

async function firecrawl(env: Env, path: "/search" | "/scrape", body: Json): Promise<Json> {
  if (!env.FIRECRAWL_API_KEY) return { error: "Firecrawl is not configured for this worker." };
  let response: Response;
  try {
    response = await fetch(FIRECRAWL_API + path, {
      method: "POST",
      headers: { Authorization: `Bearer ${env.FIRECRAWL_API_KEY}`, "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(FIRECRAWL_TIMEOUT_MS),
    });
  } catch (cause) {
    return { error: `Firecrawl request failed: ${cause instanceof Error ? cause.message : String(cause)}` };
  }
  if (!response.ok) return { error: `Firecrawl returned HTTP ${response.status}` };
  try {
    const parsed = await response.json();
    return parsed && typeof parsed === "object" ? parsed as Json : { error: "Firecrawl returned an invalid response." };
  } catch {
    return { error: "Firecrawl returned non-JSON data." };
  }
}

const TBS: Record<string, string> = { day: "qdr:d", week: "qdr:w", month: "qdr:m", year: "qdr:y" };

/** Firecrawl web search. Returns [] when unconfigured or failing; never throws. */
export async function firecrawlSearch(
  env: Env,
  query: string,
  opts: { limit: number; timeRange?: string; publicUrl: (raw: unknown) => string | null },
): Promise<WebResult[]> {
  const raw = await firecrawl(env, "/search", {
    query: query.slice(0, 500),
    limit: Math.max(1, Math.min(10, opts.limit)),
    sources: ["web"],
    ...(opts.timeRange && TBS[opts.timeRange] ? { tbs: TBS[opts.timeRange] } : {}),
  });
  if (raw.error) return [];
  const data = raw.data && typeof raw.data === "object" ? raw.data as Json : {};
  const rows = Array.isArray(data.web) ? data.web : Array.isArray(raw.data) ? raw.data as unknown[] : [];
  return rows.flatMap((item) => {
    const row = item && typeof item === "object" ? item as Json : {};
    const url = opts.publicUrl(row.url);
    if (!url || isRestrictedSchemeUrl(url)) return [];
    return [{ title: clean(row.title, 240), url, content: clean(row.description, 1_800) }];
  });
}

/** Firecrawl single-page scrape to markdown. Returns null on any failure. */
export async function firecrawlScrape(env: Env, url: string, limit = 4_000): Promise<string | null> {
  if (isRestrictedSchemeUrl(url)) return null;
  const raw = await firecrawl(env, "/scrape", { url, formats: ["markdown"], onlyMainContent: true, timeout: FIRECRAWL_TIMEOUT_MS });
  if (raw.error) return null;
  const data = raw.data && typeof raw.data === "object" ? raw.data as Json : {};
  const text = clean(data.markdown, limit);
  return text || null;
}

/**
 * Merge two engines' results: dedupe by URL (first engine wins on ties),
 * interleave so neither engine crowds the other out, cap at `limit`.
 */
export function mergeWebResults(primary: WebResult[], secondary: WebResult[], limit: number): WebResult[] {
  const out: WebResult[] = [];
  const seen = new Set<string>();
  const key = (url: string) => url.replace(/\/$/, "").toLowerCase();
  const max = Math.max(primary.length, secondary.length);
  for (let i = 0; i < max && out.length < limit; i++) {
    for (const row of [primary[i], secondary[i]]) {
      if (!row || seen.has(key(row.url)) || out.length >= limit) continue;
      seen.add(key(row.url));
      out.push(row);
    }
  }
  return out;
}
