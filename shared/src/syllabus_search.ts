/**
 * Server side of the scoped syllabus search (AXO-226).
 *
 * The database decides the scope (`resolve_syllabus_retrieval_scope`): the
 * paper's verified, approved, version-pinned syllabus or a reason why there is
 * none. This module only chooses search terms and reads the shortlist. The
 * model never supplies a paper, document, hash or SQL; at most it suggests a
 * few refinement terms, which are filtered here before they reach the database.
 */
import type { SupabaseClient } from "@supabase/supabase-js";
import { mustRpc } from "./db.js";

export type Scope =
  | { status: "available"; document_id: string; source_sha256: string; version_label?: string; exam_year?: number }
  | { status: "unavailable"; reason: string };

export interface Candidate {
  objective_id: string;
  code: string;
  title: string;
  objective_text: string;
  source_page: number;
  text_truncated: boolean;
  parent: { id: string; code: string; title: string } | null;
}

export type SearchResult =
  | { status: "available"; candidates: Candidate[] }
  | { status: "unavailable"; reason: string; candidates: [] };

/** Words that say what to do, not what is assessed. Kept out of searches. */
const STOP = new Set(`
a an and are as at be been but by can could did do does each for from had has have how if in into is it its
may might more most must no not of on one or other our out over so some such than that the their them then
there these they this those through to two up use used using was we were what when where which while who why
will with would you your
answer answers calculate calculated calculation complete describe determine diagram draw estimate evaluate
explain figure find give given hence identify mark marks name note page question questions show sketch space
state suggest table total value values work write`.trim().split(/\s+/));

const TERM = /[\p{L}\p{N}][\p{L}\p{N}.\-^²³]*/gu;
const MAX_TERMS = 12;
const MAX_TERM_LENGTH = 64;
const MAX_BYTES = 480;

function clean(word: string): string | null {
  const w = word.toLowerCase().replace(/[.\-^]+$/, "");
  if (w.length > MAX_TERM_LENGTH) return null;
  if (STOP.has(w)) return null;
  // Short words only when they carry a digit (e.g. "h2o", "2a", "v2").
  if (w.length < 4 && !/\d/.test(w)) return null;
  if (/^\d+$/.test(w)) return null; // a bare number matches nothing useful
  return w;
}

function bounded(terms: Iterable<string>): string[] {
  const out: string[] = [];
  let bytes = 0;
  for (const t of terms) {
    if (out.includes(t)) continue;
    const b = new TextEncoder().encode(t).length + 1;
    if (out.length >= MAX_TERMS || bytes + b > MAX_BYTES) break;
    out.push(t);
    bytes += b;
  }
  return out;
}

/**
 * Search terms from the printed question only (never the student's answer: a
 * wrong method must not decide what the question tests). The simple dictionary
 * does not stem, so a plural also brings its singular.
 */
export function questionTerms(questionText: string): string[] {
  const words: string[] = [];
  for (const m of questionText.normalize("NFKC").matchAll(TERM)) {
    const w = clean(m[0]);
    if (!w) continue;
    words.push(w);
    if (w.length > 4 && w.endsWith("s") && !w.endsWith("ss")) words.push(w.slice(0, -1));
  }
  return bounded(words);
}

/** Refinement terms the model suggested, reduced to plain search words. */
export function refinementTerms(raw: unknown): string[] {
  if (!Array.isArray(raw)) return [];
  const words: string[] = [];
  for (const item of raw.slice(0, 6)) {
    if (typeof item !== "string") continue;
    const phrase = [...item.normalize("NFKC").matchAll(TERM)].map((m) => clean(m[0])).filter((w): w is string => !!w).slice(0, 4).join(" ");
    if (phrase) words.push(phrase.slice(0, MAX_TERM_LENGTH));
  }
  return bounded(words).slice(0, 6);
}

export async function resolveScope(sb: SupabaseClient, paperId: string): Promise<Scope> {
  const scope = await mustRpc(sb.rpc("resolve_syllabus_retrieval_scope", { p_paper_id: paperId }), "resolve_syllabus_retrieval_scope") as any;
  if (scope?.status === "available" && typeof scope.document_id === "string" && typeof scope.source_sha256 === "string") return scope;
  return { status: "unavailable", reason: typeof scope?.reason === "string" ? scope.reason : "missing" };
}

export async function searchAny(sb: SupabaseClient, paperId: string, scope: Extract<Scope, { status: "available" }>, terms: string[], limit = 12): Promise<SearchResult> {
  if (!terms.length) return { status: "available", candidates: [] };
  const out = await mustRpc(sb.rpc("search_syllabus_topics_any", {
    p_paper_id: paperId,
    p_document_id: scope.document_id,
    p_source_sha256: scope.source_sha256,
    p_terms: terms,
    p_limit: limit,
  }), "search_syllabus_topics_any") as any;
  if (out?.status !== "available") return { status: "unavailable", reason: typeof out?.reason === "string" ? out.reason : "missing", candidates: [] };
  return { status: "available", candidates: Array.isArray(out.candidates) ? out.candidates as Candidate[] : [] };
}
