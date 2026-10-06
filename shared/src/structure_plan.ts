/**
 * What one page of the structure stage writes, decided without the database.
 *
 * AXO-116. Production run 89c8d7a9 lost 6 of its 14 pages, and every run of
 * the same paper lost the same six. The model read every page; the writes did
 * not land, and the student was told "We could not read this page well enough
 * to find the questions on it", which was not true. Two defects, both rooted in
 * the worker assuming it sees pages in order. It does not: queue delivery order
 * is not page order, and that run was structured 8, 12, 7, 5, 1, 3, …
 *
 * 1. Bare part labels. A page whose questions are printed `(a)`, `(b)` stores
 *    the labels `a` and `b`. The parent number lives on an earlier page the
 *    worker may not have seen yet, so it cannot be qualified here. The next
 *    page with an `(a)` collided with the run's label-uniqueness index
 *    (SQLSTATE 23505), the whole page insert was rejected, and the page was
 *    recorded as unreadable. Bare labels are now outside that index (they are
 *    resolved in page order downstream, AXO-122), and a qualified label that
 *    genuinely repeats is kept as a region with its label withheld and flagged
 *    for review — never a dropped page.
 *
 * 2. Continuations. `continues_from_previous` was stitched onto "the region
 *    with the highest order_index", i.e. whichever page happened to finish
 *    last. Production holds regions whose spans run [12, 3] and [11, 3]: page
 *    3's tail attached to a page-12 question, with page 3's teacher marks on
 *    it. A continuation is now written page-locally, flagged, and assembled
 *    once, in page order, after every page is structured
 *    (`private.assemble_structure`, called from `advance_after_structure`).
 *
 * Pure so both decisions are testable; the worker only does the I/O.
 */

import { takeBox } from "./contract.js";
import { ModelError } from "./model-client.js";
import type { StructureRegion } from "./prompts/structure.v1.js";

/** The key the run-level uniqueness index compares, or null when the index
    does not apply to this label. Mirrors `question_region_one_label_per_run`:
    only labels carrying a question number are unique within a run. A bare
    part (`a`, `(ii)`) legitimately recurs under every question. */
export function uniqueLabelKey(label: string | null | undefined): string | null {
  if (typeof label !== "string" || !/[0-9]/.test(label)) return null;
  const key = label.replace(/[^A-Za-z0-9]/g, "").toLowerCase();
  return key.length ? key : null;
}

/**
 * True when a "question number" is really the printed page number: a bare
 * number in the header or footer band, near the horizontal centre. Question
 * numbers are printed in the left margin, never centred in the header. When
 * the scan cut that margin off, the reader took the page number instead
 * (owner, 6 Oct 2026); the label is withheld and the region goes to review.
 */
export function looksLikePageNumber(
  label: string | null | undefined,
  box: { x: number; y: number; w: number; h: number } | null,
  width: number,
  height: number,
): boolean {
  if (!box || typeof label !== "string" || !/^\s*\d{1,3}\s*\.?\s*$/.test(label)) return false;
  const cx = (box.x + box.w / 2) / width;
  const top = box.y / height;
  const bottom = (box.y + box.h) / height;
  return cx > 0.35 && cx < 0.65 && (bottom < 0.08 || top > 0.92);
}

export interface PlannedRegion {
  order_index: number;
  span: { page: number; box: { x: number; y: number; w: number; h: number } };
  row: Record<string, unknown>;
  /** Why the label was withheld, when it was. */
  withheld_label?: string;
}

export interface PagePlanInput {
  regions: StructureRegion[];
  page: number;
  width: number;
  height: number;
  runId: string;
  paperId: string;
  studentId: string;
  /** First free order_index in the run. */
  nextIndex: number;
  /** `uniqueLabelKey` of every label already stored for this run on OTHER pages. */
  takenLabels: Set<string>;
}

export function planPage(input: PagePlanInput): PlannedRegion[] {
  const { page, width, height } = input;
  const taken = new Set(input.takenLabels);
  const out: PlannedRegion[] = [];
  let next = input.nextIndex;

  for (const [i, region] of input.regions.entries()) {
    const box = takeBox(region.box, page, width, height);
    if (!box) continue;
    const span = { page, box: { x: box.x, y: box.y, w: box.w, h: box.h } };

    // Only the top band of a page can be the tail of an earlier question.
    const continuation = i === 0 && region.continues_from_previous === true;

    const numberBox = continuation ? null : takeBox(region.number_box, page, width, height);
    let label: string | null = numberBox ? region.candidate_number : null;
    let withheld: string | undefined;

    if (looksLikePageNumber(label, numberBox, width, height)) {
      withheld = label ?? undefined;
      label = null;
    }

    const key = uniqueLabelKey(label);
    if (key && taken.has(key)) {
      // The same numbered part read twice in one run. One of the two is wrong
      // and nothing on this page says which, so the label is withheld and the
      // region goes to review. The region itself — its box, its marks — stays.
      withheld = label ?? undefined;
      label = null;
    } else if (key) {
      taken.add(key);
    }

    out.push({
      order_index: next,
      span,
      withheld_label: withheld,
      row: {
        run_id: input.runId,
        paper_id: input.paperId,
        student_id: input.studentId,
        order_index: next,
        page_spans: [span],
        question_label: label,
        question_label_box: label ? numberBox : null,
        confidence_tier: "unsure",
        continues_from_previous: continuation,
        ...(withheld ? { needs_review: true } : {}),
      },
    });
    next += 1;
  }
  return out;
}

/** The regions a previous attempt at this page wrote, which a retry must
    remove before writing its own. Since assembly moved to the run level, a
    page only ever writes regions whose first span is its own page. */
export function regionsWrittenByPage(
  rows: Array<{ id: string; page_spans: unknown }>,
  page: number,
): string[] {
  return rows
    .filter((r) => {
      const spans = r.page_spans as Array<{ page?: unknown }> | null;
      return Array.isArray(spans) && Number(spans[0]?.page) === page;
    })
    .map((r) => r.id);
}

/**
 * What the student is told about a page this stage gave up on.
 *
 * "We could not read this page" is a claim about the page, and it is only true
 * when the model is what failed. Run 89c8d7a9 showed it on six pages the model
 * had read perfectly well: the database rejected the write. Telling a student
 * their scan was unreadable when it was not sends them to re-photograph a good
 * page — the wrong fix, stated with confidence (hard rule 4).
 */
export function structureFailureReason(error: unknown): string {
  if (error instanceof ModelError) {
    return "We could not read this page well enough to find the questions on it.";
  }
  return "We read this page but could not save the questions on it. Your page is kept — try this paper again.";
}
