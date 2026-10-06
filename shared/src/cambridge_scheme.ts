// Cambridge mark-scheme lookup for checking a student's own paper.
//
// Owner decision (6 Oct 2026): Axon may read the published Cambridge mark
// scheme for the exact paper a student scanned, to check that student's work.
// The scheme is used only as private model input:
//   - found by exact identity only: the printed paper code becomes one filename
//     (9231/11, Oct/Nov 2025 -> 9231_w25_ms_11.pdf); nothing else is accepted;
//   - addressed directly, not searched for: the filename plus the subject
//     folder gives the bestexamhelp URL (bestexamhelp_subjects.ts); search is
//     only the fallback when that site does not carry the file;
//   - verified inside the document before use (code/component and session must
//     appear on the PDF itself);
//   - never written to Axon's database, R2 or logs; only the filename is kept
//     as a reference;
//   - never quoted to the student: the prompt paraphrases, and the validator
//     rejects output that copies scheme lines.
// Open-web search still filters scheme mirrors (web_sources.ts); this is the
// one deliberate, exact path.

import type { Env } from "./env.js";
import { publicWebUrl } from "./tavily.js";
import { BESTEXAMHELP_LEVELS, BESTEXAMHELP_SUBJECTS, type BestExamHelpLevel } from "./bestexamhelp_subjects.js";

const FIRECRAWL_API = "https://api.firecrawl.dev/v2";
const TAVILY_API = "https://api.tavily.com";
const SEARCH_TIMEOUT_MS = 15_000;
const SCRAPE_TIMEOUT_MS = 60_000;
/** The user-named mirror is tried first; it serves the raw PDF. */
const PREFERRED_HOST = "bestexamhelp.com";

export interface CambridgeIdentity {
  subject_code: string | null;
  exam_year: number | null;
  session: string | null;
  paper_code: string | null;
  component_code: string | null;
  variant: string | null;
  confidence?: string;
}

export interface SchemeRef {
  code: string;        // 9231
  component: string;   // 11
  series: "m" | "s" | "w";
  year: number;        // 2025
  filename: string;    // 9231_w25_ms_11.pdf
  label: string;       // 9231/11/O/N/25
}

function digits(value: unknown): string {
  return typeof value === "string" || typeof value === "number" ? String(value).replace(/\D+/g, "") : "";
}

/** Cambridge series letter from the printed session wording. */
export function seriesLetter(session: unknown): "m" | "s" | "w" | null {
  if (typeof session !== "string") return null;
  const s = session.toLowerCase();
  if (/\b(oct|nov|o\/n)\b|october|november/.test(s)) return "w";
  if (/\b(may|jun|m\/j)\b|june/.test(s)) return "s";
  if (/\b(feb|mar|f\/m)\b|february|march/.test(s)) return "m";
  return null;
}

/**
 * The one scheme filename this printed header points to, or null if any part
 * is missing or ambiguous. Never guesses a component or series.
 */
/** A full printed reference such as "9231/11/O/N/25" (the footer of every Cambridge page). */
export function refFromPrintedReference(text: unknown): SchemeRef | null {
  if (typeof text !== "string") return null;
  const m = text.toUpperCase().replace(/\s+/g, "").match(/(\d{4})\/(\d{2})\/(O\/N|M\/J|F\/M)\/(\d{2})(?!\d)/);
  if (!m) return null;
  const [, code, component, sessionCode, yy] = m;
  const series = sessionCode === "O/N" ? "w" : sessionCode === "M/J" ? "s" : "m";
  const year = 2000 + Number(yy);
  return { code: code!, component: component!, series, year, filename: `${code}_${series}${yy}_ms_${component}.pdf`, label: `${code}/${component}/${sessionCode}/${yy}` };
}

export function cambridgeSchemeRef(identity: CambridgeIdentity | null | undefined): SchemeRef | null {
  if (!identity) return null;
  // The footer reference is unambiguous on its own: code, component, series
  // and year in one printed string.
  for (const field of [identity.paper_code, identity.component_code, identity.session]) {
    const printed = refFromPrintedReference(field);
    if (printed && (!identity.subject_code || digits(identity.subject_code) === printed.code)) return printed;
  }
  if (identity.confidence === "low") return null;
  const code = digits(identity.subject_code);
  if (!/^\d{4}$/.test(code)) return null;

  // Component: "11", or "1"+"1" (paper + variant), or "9231/11" in paper_code.
  let component = "";
  for (const raw of [identity.component_code, identity.paper_code]) {
    const d = digits(raw);
    if (d.length === 6 && d.startsWith(code)) { component = d.slice(4); break; }
    if (d.length === 2) { component = d; break; }
  }
  if (!component) {
    const paper = digits(identity.paper_code ?? identity.component_code);
    const variant = digits(identity.variant);
    if (paper.length === 1 && variant.length === 1) component = paper + variant;
  }
  if (!/^\d{2}$/.test(component)) return null;

  const series = seriesLetter(identity.session);
  const year = Number(identity.exam_year);
  if (!series || !Number.isInteger(year) || year < 2000 || year > 2100) return null;
  const yy = String(year % 100).padStart(2, "0");
  const sessionLabel = series === "w" ? "O/N" : series === "s" ? "M/J" : "F/M";
  return { code, component, series, year, filename: `${code}_${series}${yy}_ms_${component}.pdf`, label: `${code}/${component}/${sessionLabel}/${yy}` };
}

function endsWithFile(url: string, filename: string): boolean {
  try { return decodeURIComponent(new URL(url).pathname).toLowerCase().endsWith(`/${filename.toLowerCase()}`); } catch { return false; }
}

async function postJson(url: string, key: string, body: unknown, timeoutMs: number): Promise<any | null> {
  try {
    const res = await fetch(url, {
      method: "POST",
      headers: { Authorization: `Bearer ${key}`, "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(timeoutMs),
    });
    return res.ok ? await res.json() : null;
  } catch {
    return null;
  }
}

function candidatesFrom(rawUrls: unknown[], ref: SchemeRef): string[] {
  const urls = rawUrls.map((u) => publicWebUrl(u)).filter((u): u is string => !!u);
  const exact = urls.filter((u) => endsWithFile(u, ref.filename));
  const derived: string[] = [];
  for (const u of urls) {
    try {
      const slug = new URL(u).pathname.split("/").find((seg) => seg.endsWith(`-${ref.code}`) && /^[a-z-]+-\d{4}$/.test(seg));
      if (slug) {
        const level = /^0\d{3}$/.test(ref.code) ? "cambridge-igcse" : "cambridge-international-a-level";
        derived.push(`https://${PREFERRED_HOST}/exam/${level}/${slug}/${ref.year}/${ref.filename}`);
        break;
      }
    } catch { /* ignore */ }
  }
  const host = (u: string) => { try { return new URL(u).hostname.replace(/^www\./, ""); } catch { return ""; } };
  const ordered = [
    ...exact.filter((u) => host(u) === PREFERRED_HOST),
    ...derived,
    ...exact.filter((u) => host(u) !== PREFERRED_HOST),
  ];
  return [...new Set(ordered)].slice(0, 4);
}

// ── Direct address (owner, 6 Oct 2026) ──────────────────────────────────────
// bestexamhelp's paths are a pure function of the paper code once the subject
// folder is known, so the common case needs no search at all: compute the URL,
// confirm it with a free HEAD from the Worker, read it once.

const BEH = `https://${PREFERRED_HOST}/exam`;
const PROBE_TIMEOUT_MS = 5_000;
const INDEX_MAX_AGE_MS = 30 * 24 * 3600 * 1000;

/** The direct URL for this scheme, given the subject folder ("<level>/<slug>-<code>"). */
export function bestExamHelpUrl(ref: SchemeRef, folder: string): string {
  return `${BEH}/${folder}/${ref.year}/${ref.filename}`;
}

/** Index pages to read for a code missing from the table, likeliest level first. */
function levelsFor(code: string): BestExamHelpLevel[] {
  const first: BestExamHelpLevel = code.startsWith("0") ? "cambridge-igcse" : code.startsWith("9") ? "cambridge-international-a-level" : "cambridge-o-level";
  return [first, ...BESTEXAMHELP_LEVELS.filter((l) => l !== first)];
}

/**
 * The subject folder for a code: the built-in table, else the site's own
 * subject index (one Firecrawl call per level, served from Firecrawl's cache
 * for 30 days). Null when the site does not carry the subject.
 */
export async function bestExamHelpFolder(env: Env, code: string): Promise<string | null> {
  const known = BESTEXAMHELP_SUBJECTS[code];
  if (known) return known;
  if (!env.FIRECRAWL_API_KEY) return null;
  const pattern = new RegExp(`/exam/(cambridge-[a-z-]+/[a-z0-9-]+-${code})/(?:index\\.php)?$`);
  for (const level of levelsFor(code)) {
    const res = await postJson(`${FIRECRAWL_API}/scrape`, env.FIRECRAWL_API_KEY, {
      url: `${BEH}/${level}/`, formats: ["links"], maxAge: INDEX_MAX_AGE_MS, timeout: SEARCH_TIMEOUT_MS,
    }, SEARCH_TIMEOUT_MS + 5_000);
    for (const link of (res?.data?.links ?? []) as unknown[]) {
      const m = typeof link === "string" ? link.match(pattern) : null;
      if (m) return m[1]!;
    }
  }
  return null;
}

export type ProbeResult = "found" | "missing" | "unknown";

/**
 * HEAD the URL from the Worker itself: free, no provider credits. 404/410
 * means the file is not there; anything else that is not a PDF 200 (a block,
 * a timeout) is "unknown", and the read step decides.
 */
export async function probeUrl(url: string): Promise<ProbeResult> {
  try {
    const res = await fetch(url, { method: "HEAD", redirect: "follow", signal: AbortSignal.timeout(PROBE_TIMEOUT_MS) });
    if (res.status === 404 || res.status === 410) return "missing";
    const type = res.headers.get("content-type") ?? "";
    if (res.ok && /pdf|octet-stream/i.test(type)) return "found";
    return "unknown";
  } catch {
    return "unknown";
  }
}

/**
 * Candidate URLs for exactly this file, best first.
 *
 * 1. The computed bestexamhelp address, unless a HEAD says it is not there.
 *    No search is made when it is (or may be) there.
 * 2. Only when bestexamhelp does not carry it: one search for the quoted
 *    filename, one engine at a time (Tavily, then Firecrawl).
 */
export async function locateScheme(env: Env, ref: SchemeRef): Promise<string[]> {
  const folder = await bestExamHelpFolder(env, ref.code);
  if (!folder) return searchScheme(env, ref);
  const url = bestExamHelpUrl(ref, folder);
  if (await probeUrl(url) !== "missing") return [url];
  return searchScheme(env, ref, [url]);
}

/** Last resort: find a copy of the exact filename on any host. */
export async function searchScheme(env: Env, ref: SchemeRef, exclude: string[] = []): Promise<string[]> {
  const query = `"${ref.filename}"`;
  const keep = (urls: string[]) => urls.filter((u) => !exclude.includes(u));
  if (env.TAVILY_API_KEY) {
    const tv = await postJson(`${TAVILY_API}/search`, env.TAVILY_API_KEY, { query, max_results: 10, search_depth: "basic", include_raw_content: false }, SEARCH_TIMEOUT_MS);
    const found = keep(candidatesFrom((tv?.results ?? []).map((r: any) => r?.url), ref));
    if (found.length) return found;
  }
  if (env.FIRECRAWL_API_KEY) {
    const fc = await postJson(`${FIRECRAWL_API}/search`, env.FIRECRAWL_API_KEY, { query, limit: 10, sources: ["web"] }, SEARCH_TIMEOUT_MS);
    return keep(candidatesFrom((fc?.data?.web ?? []).map((r: any) => r?.url), ref));
  }
  return [];
}

/** The PDF's own text must name this exact paper before it is trusted. */
export function documentMatches(markdown: string, ref: SchemeRef): boolean {
  const text = markdown.replace(/\s+/g, " ");
  const paper = new RegExp(`${ref.code}\\s*/\\s*${ref.component}\\b`);
  const year = String(ref.year);
  const session = ref.series === "w" ? /October\s*\/?\s*November/i : ref.series === "s" ? /May\s*\/?\s*June/i : /February\s*\/?\s*March/i;
  const head = text.slice(0, 4000);
  return paper.test(head) && head.includes(year) && session.test(head) && /mark scheme/i.test(head);
}

export interface FetchedScheme { ref: SchemeRef; markdown: string; sourceHost: string }

/**
 * Locate and read the scheme for this exact paper, or null. Tries at most four
 * candidates; the first whose PDF text names this paper wins. Firecrawl's own
 * cache (maxAge) serves repeat reads, so Axon keeps no copy.
 */
export async function fetchCambridgeScheme(env: Env, ref: SchemeRef, candidates?: string[]): Promise<FetchedScheme | null> {
  if (!env.FIRECRAWL_API_KEY && !env.TAVILY_API_KEY) return null;
  const tried: string[] = [];
  const attempt = async (urls: string[]) => {
    for (const url of urls) {
      if (tried.includes(url)) continue;
      tried.push(url);
      const markdown = await readSchemePdf(env, url);
      if (markdown && documentMatches(markdown, ref)) return { ref, markdown, sourceHost: new URL(url).hostname };
    }
    return null;
  };
  const direct = await attempt(candidates ?? await locateScheme(env, ref));
  if (direct) return direct;
  // The direct address could not be read or did not name this paper: search
  // once, skipping what was already tried. A search hit is never trusted
  // without the same in-document check.
  if (!tried.some((u) => !isBestExamHelp(u))) return attempt(await searchScheme(env, ref, tried));
  return null;
}

function isBestExamHelp(url: string): boolean {
  try { return new URL(url).hostname.replace(/^www\./, "") === PREFERRED_HOST; } catch { return false; }
}

/**
 * Read one scheme PDF as text. One engine at a time: Firecrawl first here,
 * because its PDF parser keeps the scheme's table rows (the per-question
 * split depends on them); Tavily extract only if Firecrawl cannot read it.
 */
async function readSchemePdf(env: Env, url: string): Promise<string | null> {
  if (env.FIRECRAWL_API_KEY) {
    const res = await postJson(`${FIRECRAWL_API}/scrape`, env.FIRECRAWL_API_KEY, {
      url, formats: ["markdown"], parsers: ["pdf"], onlyMainContent: false,
      maxAge: 7 * 24 * 3600 * 1000, timeout: SCRAPE_TIMEOUT_MS,
    }, SCRAPE_TIMEOUT_MS + 5_000);
    const markdown: unknown = res?.data?.markdown;
    const type = String(res?.data?.metadata?.contentType ?? "");
    if (typeof markdown === "string" && markdown.trim() && (!type || type.includes("pdf"))) return markdown;
  }
  if (env.TAVILY_API_KEY) {
    const res = await postJson(`${TAVILY_API}/extract`, env.TAVILY_API_KEY, { urls: [url], extract_depth: "advanced", format: "markdown" }, SCRAPE_TIMEOUT_MS);
    const text: unknown = res?.results?.[0]?.raw_content;
    if (typeof text === "string" && text.trim()) return text;
  }
  return null;
}

/** "3(b)(ii)" -> 3 ; "Q12" -> 12 ; null when no leading question number. */
export function topLevelNumber(label: unknown): number | null {
  if (typeof label !== "string") return null;
  const m = label.trim().match(/^(?:q(?:uestion)?\s*)?(\d{1,2})/i);
  return m ? Number(m[1]) : null;
}

/**
 * Split scheme text into top-level questions. A row starts a question when its
 * first cell (or line) is a question label: "3", "3(a)", "3(b)(ii)". The
 * preamble (generic marking principles) is dropped. Returns a map from the
 * question number to its text.
 */
/**
 * Index of the first answer-table header ("Question | Answer | Marks"). A
 * Cambridge scheme opens with numbered generic marking principles laid out as
 * table rows "| 1 |", "| 2 |"... which look exactly like question rows, so the
 * question table is only read from its own header onwards.
 */
export function firstAnswerTable(lines: string[]): number {
  return lines.findIndex((l) => /^\W*question\W+answer\W+marks/i.test(l.replace(/\*\*/g, "")));
}

export function schemeSections(markdown: string): Map<number, string> {
  const all = markdown.split(/\r?\n/);
  const header = firstAnswerTable(all);
  const lines = header >= 0 ? all.slice(header) : all;
  const sections = new Map<number, string[]>();
  let current: number | null = null;
  let highest = 0;
  const startOf = (line: string): number | null => {
    const cell = line.replace(/^\s*\|?\s*/, "").replace(/\*\*/g, "");
    const m = cell.match(/^(\d{1,2})(?:\s*\(\s*[a-z]{1,4}\s*\))*\s*(?:\||$|\s{2,})/i)
      ?? cell.match(/^(\d{1,2})(?:\s*\(\s*[a-z]{1,4}\s*\))+\s/i);
    return m ? Number(m[1]) : null;
  };
  for (const line of lines) {
    const n = startOf(line);
    // A new question must be the current one or the next few; page numbers and
    // stray integers further away are content, not structure.
    if (n !== null && n >= 1 && (current === null ? n <= 2 : n === current || (n > current && n <= current + 2))) {
      current = n;
      highest = Math.max(highest, n);
    }
    if (current !== null) {
      const list = sections.get(current) ?? [];
      list.push(line);
      sections.set(current, list);
    }
  }
  return new Map([...sections].map(([n, rows]) => [n, rows.join("\n").trim()]));
}

/** The scheme text for one region's top-level question, capped, or null. */
export function sectionFor(sections: Map<number, string>, label: unknown, max = 6_000): string | null {
  const n = topLevelNumber(label);
  if (n === null) return null;
  const text = sections.get(n);
  return text ? text.slice(0, max) : null;
}

/**
 * Does a printed reference agree with the scheme we are about to use? The
 * code/component must match; any session or year printed must match too.
 */
export function headerMatches(reference: string | null, ref: SchemeRef): boolean {
  if (!reference) return false;
  const compact = reference.toUpperCase().replace(/\s+/g, "");
  if (!compact.includes(`${ref.code}/${ref.component}`)) return false;
  const full = compact.match(/\d{4}\/\d{2}\/(O\/N|M\/J|F\/M)\/(\d{2})/);
  if (full) {
    const expected = ref.series === "w" ? "O/N" : ref.series === "s" ? "M/J" : "F/M";
    if (full[1] !== expected || full[2] !== String(ref.year % 100).padStart(2, "0")) return false;
  }
  const year = reference.match(/\b(20\d{2})\b/);
  if (year && Number(year[1]) !== ref.year) return false;
  const printedSeries = seriesLetter(reference);
  if (printedSeries && printedSeries !== ref.series) return false;
  return true;
}

/** The scheme's opening notes (mark types, abbreviations), capped. */
export function schemePreamble(markdown: string, max = 3_000): string | null {
  const lines = markdown.split(/\r?\n/);
  const header = firstAnswerTable(lines);
  const head = (header > 0 ? lines.slice(0, header) : lines.slice(0, 120)).join("\n");
  const at = head.search(/abbreviation|mark scheme notes|types of mark|\bM marks?\b/i);
  if (at < 0) return null;
  const notes = head.slice(at);
  return notes.length > 40 ? notes.slice(0, max) : null;
}
