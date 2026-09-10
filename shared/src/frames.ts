/**
 * A model box means nothing without the image it was drawn on.
 *
 * Re-audit P0-H. The content worker can send the model a cropped band of a
 * page instead of the whole page, and the prompt asks for boxes on "the images
 * given to you" — so the model answers in the crop's own 0–1000 grid. The
 * worker then converted that with:
 *
 *     takeBox(v.box, page, fullPageWidth, fullPageHeight)
 *
 * which is the transform for a box drawn on the whole page. The crop's local
 * y=200 became 20% down the entire sheet rather than 20% down the band, and
 * the band's own offset was never added at all. Tapping an extracted value
 * would highlight pixels the model never looked at — which is the provenance
 * claim the product is built on, quietly wrong.
 *
 * Today no region has a crop_key (checked against production: 0 of 64), so
 * this is a landmine rather than a live fire. It arms itself the moment the
 * crop stage is switched on, which is exactly when nobody would be looking for
 * a coordinate bug.
 *
 * The fix is to stop passing bare numbers around. Every image handed to a
 * model carries a frame describing what it is a picture of, and one function
 * maps a box back to the page. A rectangle without a frame is not provenance.
 */

import type { PixelBox } from "./contract.js";

/** A rectangle in real page pixels. */
export interface PixelBand {
  x: number;
  y: number;
  w: number;
  h: number;
}

/**
 * What a particular image handed to the model actually shows.
 *
 * `cropmask` is the same geometry as `crop` — it is the same band with the
 * teacher's ink isolated — and is kept as a distinct kind only so a caller
 * cannot silently pass one where the other was meant.
 */
export type ModelFrame =
  | { kind: "page"; pageNumber: number; pageWidth: number; pageHeight: number }
  | {
      kind: "crop" | "cropmask";
      pageNumber: number;
      pageWidth: number;
      pageHeight: number;
      /** Where this crop sits on its page, in page pixels. */
      band: PixelBand;
    };

const NORMALISED_MAX = 1000;
/** The model overshoots the grid by a pixel or two; more than that is a miss. */
const OVERSHOOT_TOLERANCE = 2;

interface RawBox {
  x?: unknown;
  y?: unknown;
  w?: unknown;
  h?: unknown;
}

/**
 * A box the model could actually have drawn.
 *
 * Lower bounds are checked here and were not checked before: `takeBox` tested
 * only the upper edges, so a negative x or y passed straight through and was
 * multiplied into a negative pixel coordinate. That is not a box, and storing
 * it as provenance means highlighting off the top-left of the page.
 */
function normalised(raw: unknown): { x: number; y: number; w: number; h: number } | null {
  const b = (raw ?? {}) as RawBox;
  const nums = [b.x, b.y, b.w, b.h];
  if (!nums.every((n) => typeof n === "number" && Number.isFinite(n))) return null;
  const x = b.x as number, y = b.y as number, w = b.w as number, h = b.h as number;
  if (w <= 0 || h <= 0) return null;
  if (x < 0 || y < 0) return null;
  if (x > NORMALISED_MAX || y > NORMALISED_MAX) return null;
  if (x + w > NORMALISED_MAX + OVERSHOOT_TOLERANCE) return null;
  if (y + h > NORMALISED_MAX + OVERSHOOT_TOLERANCE) return null;
  return { x, y, w, h };
}

/**
 * Map a box the model drew on `frame` back to pixels on its page.
 *
 * The only place this conversion happens. Worker code should never see a raw
 * model box: it gets a page-space PixelBox or null.
 */
export function mapModelBoxToPage(frame: ModelFrame, raw: unknown): PixelBox | null {
  const b = normalised(raw);
  if (!b) return null;

  if (frame.kind === "page") {
    return {
      page: frame.pageNumber,
      x: Math.round((b.x / NORMALISED_MAX) * frame.pageWidth),
      y: Math.round((b.y / NORMALISED_MAX) * frame.pageHeight),
      w: Math.max(1, Math.round((b.w / NORMALISED_MAX) * frame.pageWidth)),
      h: Math.max(1, Math.round((b.h / NORMALISED_MAX) * frame.pageHeight)),
    };
  }

  // The crop's grid spans the band, not the page: scale by the band's size and
  // then translate by where the band sits. Both halves matter — scaling alone
  // was the bug, and translating alone would be just as wrong.
  const { band } = frame;
  return {
    page: frame.pageNumber,
    x: Math.round(band.x + (b.x / NORMALISED_MAX) * band.w),
    y: Math.round(band.y + (b.y / NORMALISED_MAX) * band.h),
    w: Math.max(1, Math.round((b.w / NORMALISED_MAX) * band.w)),
    h: Math.max(1, Math.round((b.h / NORMALISED_MAX) * band.h)),
  };
}

/**
 * The frame a model's `page_index` refers to.
 *
 * Out of range returns null rather than clamping to the first or last image.
 * Clamping is how a field claimed to be on page 5 ends up recorded on page 1 —
 * a wrong answer that looks like a right one. The previous code did
 * `Math.min(page_index, spans.length - 1)`, which is exactly that clamp, and
 * then read every field's dimensions off the FIRST page regardless, so a
 * booklet whose pages differ in size mapped later pages through the wrong
 * height.
 */
export function frameForIndex(frames: ModelFrame[], pageIndex: unknown): ModelFrame | null {
  if (!frames.length) return null;
  const i = pageIndex ?? 0;
  if (typeof i !== "number" || !Number.isInteger(i)) return null;
  if (i < 0 || i >= frames.length) return null;
  return frames[i];
}
