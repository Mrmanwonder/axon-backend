// How big a page actually is, and what to do when nobody wrote it down.
//
// ── the defect this replaces ───────────────────────────────────────────────
//
// Structure and content both convert the model's boxes off a 0-1000 normalised
// grid into page pixels, and both did it like this:
//
//     const width  = meta.width  ?? 2400;
//     const height = meta.height ?? 3200;
//
// `conditioning_meta` has never carried `width` or `height` — not on one page in
// production, all 80 of them — so that fallback was not a fallback, it was the
// only branch. Every box on every page was scaled against a 2400x3200 page
// whatever shape the page really was, and a 4:3 page came out stretched by a
// third in one axis. Those boxes are what the crop stage cuts against, which is
// why this is fixed before anything is cropped with them.
//
// The client now writes both (src/scan/conditioning.js). This is what the
// workers read them with, and what they do about the pages written before that.
//
// ── on not guessing ────────────────────────────────────────────────────────
//
// CLAUDE.md rule 4: never fill a gap with a plausible guess. 2400x3200 was a
// plausible guess and it was wrong on every page. So the order here is:
//
//   1. The dimensions, if they were recorded.
//   2. Derived, if enough was recorded to derive them honestly — `source_size`
//      gives the aspect the page was conditioned from and `quality_signals`
//      gives the long edge that came out, and the two together determine the
//      page exactly. This recovers the pages that went through `conditionPage`
//      before it wrote the dimensions down.
//   3. Nothing. Not a default. The caller decides what an unplaceable page
//      means for its stage, and says so out loud.

export interface PageDimensions {
  width: number;
  height: number;
  /** "recorded" | "derived". Carried so a downstream oddity can be traced to
      which of the two produced the numbers, rather than guessed at later. */
  source: "recorded" | "derived";
}

interface DimensionInputs {
  conditioning_meta?: unknown;
  quality_signals?: unknown;
}

function positive(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) && value > 0 ? Math.round(value) : null;
}

/**
 * The page's pixel dimensions, or null when they cannot be established.
 *
 * Null is a real answer and callers must handle it. It means "we cannot place
 * anything on this page", which is a page-level unreadable — not a reason to
 * substitute a shape and carry on placing boxes against it.
 */
export function pageDimensions(page: DimensionInputs): PageDimensions | null {
  const meta = (page.conditioning_meta ?? {}) as Record<string, unknown>;

  const width = positive(meta.width);
  const height = positive(meta.height);
  if (width && height) return { width, height, source: "recorded" };

  // Conditioning scales the source by a single factor and never changes its
  // aspect (targetSize caps both axes by the same ratio), so the source's shape
  // and the conditioned long edge fully determine the conditioned page.
  const source = (meta.source_size ?? null) as Record<string, unknown> | null;
  const signals = (page.quality_signals ?? {}) as Record<string, unknown>;
  const sourceW = positive(source?.width);
  const sourceH = positive(source?.height);
  const longEdge = positive(signals.long_edge);

  if (sourceW && sourceH && longEdge) {
    const scale = longEdge / Math.max(sourceW, sourceH);
    return {
      width: Math.max(1, Math.round(sourceW * scale)),
      height: Math.max(1, Math.round(sourceH * scale)),
      source: "derived",
    };
  }

  return null;
}

/** Said to the student when a page's dimensions cannot be established. States
    the consequence and what to do; does not mention conditioning_meta. */
export const UNPLACEABLE_PAGE_REASON =
  "We could not work out the size of this page, so we cannot say where anything on it sits. Please add it again.";
