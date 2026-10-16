/**
 * The AXO-122 placement walk, mirrored from Axon-Site `src/questionCount.js`
 * (and its SQL twin `public.question_count_contract`). The review screen groups
 * and counts parts with that walk, so the backend judges numbering and decides
 * who is asked with the same one: a part the screen can place is not "badly
 * numbered" here, and a part the screen shows as unassigned is asked about.
 *
 * Council D1 (7 Oct 2026): numbering is judged on placed labels in page order,
 * not on raw labels in the order structure happened to finish.
 *
 * Keep `parseQuestionLabel` and `placeRegions` byte-for-byte equivalent to the
 * site's; the fixture test in __tests__/placement.test.ts reads the site's
 * contract cases.
 */

import type { SignalValue } from "./confidence.js";

export interface PlacementInput {
  order_index?: number;
  label: string | null;
  page: number | null;
  y: number | null;
  /** True when the region carries a mark, an answer or question text. */
  evidence: boolean;
}

export interface Placed<R extends PlacementInput = PlacementInput> {
  region: R;
  q: number | null;
  part: string | null;
  counted: boolean;
  unassigned: boolean;
  inferred: boolean;
}

/** Mirrors public.parse_question_label. Returns { q, part }; either may be null. */
export function parseQuestionLabel(label: string | null | undefined): { q: number | null; part: string | null } {
  const s = String(label ?? "").trim().toLowerCase();
  if (!s) return { q: null, part: null };

  let m = s.match(/^\(\s*([ivx]+)\s*\)$|^([ivx]{2,})[.)]?$/);
  if (m) return { q: null, part: `(${m[1] ?? m[2]})` };

  m = s.match(/^\(?\s*(?:(\d{1,4})\s*[.)]?\s*)?\(?\s*([a-z])\s*\)?\s*(?:[.)]?\s*\(?\s*([ivx]+)\s*\)?)?\s*$/);
  if (m) return { q: m[1] ? Number(m[1]) : null, part: m[2] + (m[3] ? `(${m[3]})` : "") };

  m = s.match(/^\(?\s*(\d{1,4})\s*[.)]?\s*$/);
  if (m) return { q: Number(m[1]), part: null };

  return { q: null, part: null };
}

const nullsLast = (x: number | null, y: number | null) => (x == null ? (y == null ? 0 : 1) : y == null ? -1 : x - y);

/** Page, then top of the region on that page, then the stored index. */
export function byReadingOrder(a: PlacementInput, b: PlacementInput): number {
  return nullsLast(a.page, b.page) || nullsLast(a.y, b.y) || (a.order_index ?? 0) - (b.order_index ?? 0);
}

/** The one walk over a paper's regions, in reading order. See the site's questionCount.js. */
export function placeRegions<R extends PlacementInput>(regions: R[]): Placed<R>[] {
  const used = new Set<string>();
  let prevQ: number | null = null;
  let prevPage: number | null = null;
  const placed: Placed<R>[] = [];

  for (const region of [...regions].sort(byReadingOrder)) {
    const { q, part } = parseQuestionLabel(region.label);
    const page = region.page ?? null;
    const entry: Placed<R> = { region, q: null, part, counted: true, unassigned: false, inferred: false };

    if (q === null && part === null) {
      entry.counted = !!region.evidence;
      entry.unassigned = !!region.evidence;
      prevQ = null;
      prevPage = null;
    } else if (q !== null) {
      entry.q = q;
      if (part !== null) used.add(`${q}:${part}`);
      prevQ = q;
      prevPage = page;
    } else if (prevQ !== null && prevPage !== null && page !== null
        && page - prevPage >= 0 && page - prevPage <= 1
        && !used.has(`${prevQ}:${part}`)) {
      used.add(`${prevQ}:${part}`);
      entry.q = prevQ;
      entry.inferred = true;
      prevPage = page;
    } else {
      entry.unassigned = true;
      prevQ = null;
      prevPage = null;
    }
    placed.push(entry);
  }
  return placed;
}

/** "3(c)", "2(d)(i)", "4" — the label the student sees, or null when unplaced. */
export function placedLabelText(q: number | null, part: string | null): string | null {
  if (q === null) return null;
  if (!part) return String(q);
  return `${q}${part.startsWith("(") ? part : `(${part[0]})${part.slice(1)}`}`;
}

export interface RegionPlacement {
  /** Numbering, judged on placed labels in page order. */
  structural: SignalValue;
  /** Ask-rule (c): a part the walk cannot place, or that still collides once placed. */
  unplaceable: boolean;
  /** False for an unlabelled region with no evidence: not a part, never asked. */
  counted: boolean;
  /** The placed label ("3(c)") or null. */
  placedLabel: string | null;
  /** Position in reading order. */
  readingIndex: number;
}

/**
 * Placement verdicts for every region, returned in the INPUT order.
 *
 *   - unassigned (no parent can be determined)          → structural false, unplaceable
 *   - two counted parts placed on the same question+part → structural false, unplaceable
 *   - placed, but its question number jumps (not the same as, or one after,
 *     the previous placed question)                      → structural false, NOT unplaceable:
 *     the part is placed and shown where it was read; a skipped question or a
 *     page that did not read is not something the student can fix on this part.
 *   - an unlabelled region with no evidence              → structural unknown, not counted
 *   - otherwise                                          → structural true
 */
export function placementVerdicts(regions: PlacementInput[]): RegionPlacement[] {
  const indexed = regions.map((r, i) => ({ ...r, _i: i }));
  const placed = placeRegions(indexed);
  const keyCount = new Map<string, number>();
  for (const e of placed) {
    if (!e.counted || e.q === null) continue;
    const key = `${e.q}:${e.part ?? ""}`;
    keyCount.set(key, (keyCount.get(key) ?? 0) + 1);
  }
  const out: RegionPlacement[] = new Array(regions.length);
  let prevQ: number | null = null;
  placed.forEach((e, readingIndex) => {
    const i = e.region._i;
    const placedLabel = placedLabelText(e.q, e.part);
    if (!e.counted) {
      out[i] = { structural: "unknown", unplaceable: false, counted: false, placedLabel, readingIndex };
      return;
    }
    if (e.unassigned || e.q === null) {
      out[i] = { structural: false, unplaceable: true, counted: true, placedLabel, readingIndex };
      return;
    }
    const collides = (keyCount.get(`${e.q}:${e.part ?? ""}`) ?? 0) > 1;
    const inSequence = prevQ === null || e.q === prevQ || e.q === prevQ + 1;
    prevQ = e.q;
    out[i] = { structural: !collides && inSequence, unplaceable: collides, counted: true, placedLabel, readingIndex };
  });
  return out;
}

/** Placement input from a question_region row, the way the site builds it. */
export function placementInput(row: {
  order_index?: number | null;
  question_label?: string | null;
  page_spans?: unknown;
  marks_awarded?: unknown;
  marks_available?: unknown;
  student_answer?: unknown;
  question_text?: unknown;
}): PlacementInput {
  const spans = Array.isArray(row.page_spans) ? row.page_spans as Array<{ page?: unknown; box?: { y?: unknown } }> : [];
  const first = spans[0];
  const page = first && first.page != null && Number.isFinite(Number(first.page)) ? Number(first.page) : null;
  const y = first?.box && first.box.y != null && Number.isFinite(Number(first.box.y)) ? Number(first.box.y) : null;
  return {
    order_index: row.order_index ?? undefined,
    label: row.question_label ?? null,
    page,
    y,
    evidence: row.marks_awarded != null || row.marks_available != null || !!row.student_answer || !!row.question_text,
  };
}
