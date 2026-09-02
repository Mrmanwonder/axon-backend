import type { PixelBox } from "./contract.js";

export type MarkShape = "glyph" | "crossing" | "enclosure" | "stroke" | "unknown";
export type MarkClass = "tick" | "cross" | "circle" | "underline" | "marginal_number" | "comment" | "unknown";

export interface RawMark {
  page: number;
  box: PixelBox;
  shape: MarkShape;
  metrics?: { quadrants?: [number, number, number, number]; [key: string]: unknown };
}

export interface AttributedMark {
  page_number: number;
  box: PixelBox;
  shape: MarkShape;
  mark_class: MarkClass;
  value: number | null;
  region_index: number | null;
  metrics: Record<string, unknown>;
}

export interface RegionSpan {
  page: number;
  box: PixelBox;
}

export interface AttributionRegion {
  order_index: number;
  label: string | null;
  spans: RegionSpan[];
}

export interface MarginBand {
  x0: number;
  x1: number;
}

const centre = (b: PixelBox) => ({ x: b.x + b.w / 2, y: b.y + b.h / 2 });

/**
 * Coarse ink-shape → mark-class mapping. `inMarginBand` disambiguates a
 * glyph as most likely a marginal mark number rather than something else
 * small and round.
 */
export function classifyMark(mark: RawMark, inMarginBand: boolean): MarkClass {
  const q = mark.metrics?.quadrants ?? [0, 0, 0, 0];
  switch (mark.shape) {
    case "glyph":
      return inMarginBand ? "marginal_number" : "unknown";
    case "crossing":
      if (q.every((v) => v >= 0.15)) return "cross";
      if (q[0] < 0.1) return "tick";
      return "unknown";
    case "enclosure":
      return "circle";
    case "stroke":
      return "underline";
    default:
      return "unknown";
  }
}

/**
 * Groups small marks that sit in a row into a single "comment" — a run of
 * a teacher's handwritten note gets picked up as several glyph-shaped marks
 * individually, and this stitches them back into one attributable object
 * rather than four unrelated marginal numbers.
 */
export function groupComments(marks: RawMark[], pageWidth: number): RawMark[][] {
  const candidates = marks.filter((m) => m.shape === "glyph" || m.shape === "unknown");
  const groups: RawMark[][] = [];
  const used = new Set<RawMark>();
  for (const seed of candidates) {
    if (used.has(seed)) continue;
    const rowHeight = Math.max(seed.box.h, 8);
    const row = candidates.filter(
      (m) => !used.has(m) && m.page === seed.page && Math.abs(centre(m.box).y - centre(seed.box).y) < rowHeight * 0.8
    );
    if (row.length < 4) continue;
    row.sort((a, b) => a.box.x - b.box.x);
    const gaps = row.slice(1).map((m, i) => m.box.x - (row[i].box.x + row[i].box.w));
    const median = gaps.slice().sort((a, b) => a - b)[Math.floor(gaps.length / 2)];
    if (median > rowHeight * 2.5) continue;
    if (row[row.length - 1].box.x + row[row.length - 1].box.w - row[0].box.x < pageWidth * 0.08) continue;
    row.forEach((m) => used.add(m));
    groups.push(row);
  }
  return groups;
}

function assignToRegion(mark: { page: number; box: PixelBox }, regions: AttributionRegion[]): number | null {
  const c = centre(mark.box);
  const onPage = regions
    .map((r, i) => ({ index: i, span: r.spans.find((s) => s.page === mark.page) }))
    .filter((r): r is { index: number; span: RegionSpan } => !!r.span);
  if (!onPage.length) return null;
  const inside = onPage.filter(
    ({ span }) => c.x >= span.box.x && c.x <= span.box.x + span.box.w && c.y >= span.box.y && c.y <= span.box.y + span.box.h
  );
  if (inside.length === 1) return inside[0].index;
  if (inside.length > 1) {
    // Ambiguous: the mark sits inside more than one region's box (regions
    // can overlap slightly at boundaries). Prefer the smaller region — it's
    // the tighter, more specific claim.
    inside.sort((a, b) => a.span.box.w * a.span.box.h - b.span.box.w * b.span.box.h);
    return inside[0].index;
  }
  let best: number | null = null;
  let bestDistance = Infinity;
  for (const { index, span } of onPage) {
    const top = span.box.y;
    const bottom = span.box.y + span.box.h;
    const distance = c.y < top ? top - c.y : c.y > bottom ? c.y - bottom : 0;
    if (distance < bestDistance) {
      bestDistance = distance;
      best = index;
    }
  }
  return best;
}

export interface AttributeOptions {
  regions: AttributionRegion[];
  marks: RawMark[];
  marginBands: Map<number, MarginBand | null>;
  pageWidths: Map<number, number>;
}

/** Assigns every detected ink mark to the question region it belongs to, grouping stray marks into comments first. */
export function attribute(opts: AttributeOptions): AttributedMark[] {
  const { regions, marks, marginBands, pageWidths } = opts;
  const commentMembers = new Set<RawMark>();
  const out: AttributedMark[] = [];

  for (const [page, width] of pageWidths) {
    const pageMarks = marks.filter((m) => m.page === page);
    for (const group of groupComments(pageMarks, width)) {
      group.forEach((m) => commentMembers.add(m));
      const x = Math.min(...group.map((m) => m.box.x));
      const y = Math.min(...group.map((m) => m.box.y));
      const box: PixelBox = {
        page,
        x,
        y,
        w: Math.max(...group.map((m) => m.box.x + m.box.w)) - x,
        h: Math.max(...group.map((m) => m.box.y + m.box.h)) - y,
      };
      out.push({
        page_number: page,
        box,
        shape: "unknown",
        mark_class: "comment",
        value: null,
        region_index: assignToRegion({ page, box }, regions),
        metrics: { grouped_from: group.length },
      });
    }
  }

  for (const mark of marks) {
    if (commentMembers.has(mark)) continue;
    const band = marginBands.get(mark.page) ?? null;
    const c = centre(mark.box);
    const inBand = !!band && c.x >= band.x0 && c.x <= band.x1;
    out.push({
      page_number: mark.page,
      box: mark.box,
      shape: mark.shape,
      mark_class: classifyMark(mark, inBand),
      value: null,
      region_index: assignToRegion(mark, regions),
      metrics: { ...mark.metrics, in_margin_band: inBand },
    });
  }

  return out;
}
