// The geometry half of the crop stage (AXON_FIX_BRIEF.md §8).
//
// Kept apart from the WASM codec plumbing on purpose: this is where every
// decision that could put a teacher's mark outside the frame lives, and it is
// pure, so it can be tested in Node against real numbers rather than reasoned
// about inside a Worker.
//
// ── which image the crops come from ────────────────────────────────────────
//
// §8.2 says to prefer the original in `axon-originals` over the conditioned
// derivative. Cropping the original is not possible as written, and the reason
// is worth stating rather than quietly working around: `question_region.
// page_spans` boxes are in **conditioned-page pixel space**. Structure looks at
// the conditioned page and `takeBox` scales the model's 0-1000 grid against
// `conditioning_meta.width`/`height`. The original is unwarped and at a
// different scale, so those same boxes cut out of it would land somewhere else
// on the paper — a crop that misses the question entirely, which is a worse
// failure than the one cropping is meant to fix.
//
// The concern behind the instruction was resolution: "cropping a 1000px page is
// worthless". WP3's capture floor is what answers that. Every new page is
// 2400px on its long edge, so a question band cut from it is at full
// conditioned resolution — which is the most resolution that exists in the
// coordinate space the boxes are expressed in.
//
// The original is still stored and still recoverable; it is simply not what
// these particular coordinates refer to.
//
// ── why crops are never resampled ──────────────────────────────────────────
//
// A crop is cheaper than a page because it contains less of the page, not
// because it has been shrunk. Downscaling a crop to some target long edge would
// hand the model *less* detail for the region than the full page did, which
// inverts the entire point. So the pixels are copied at their native scale and
// nothing is resampled.

export interface Box {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface PageSpan {
  page: number;
  box?: Box | null;
}

export interface RgbaImage {
  data: Uint8Array | Uint8ClampedArray;
  width: number;
  height: number;
}

/** Fraction of the page's own height added above and below the region. */
export const CROP_PAD_PAGE_H = 0.025;
/** Fraction of the region's height added above and below, whichever is larger. */
export const CROP_PAD_BOX_H = 0.15;
/**
 * How far in from the page edge a crop may start horizontally.
 *
 * Zero: crops span the **full page width**, always.
 *
 * §8.2 asks for padding wide enough that a mark in the margin survives,
 * "especially horizontally", and full width is the widest that padding can be.
 * It is also the only value that makes clipping a marginal number structurally
 * impossible rather than a matter of whether the chosen fraction happened to be
 * enough on this page — and §8.5 is explicit that a crop which clips the mark is
 * worse than no crop at all. The teacher's total for a question lives in the
 * margin, outside the region box, at a horizontal position nothing in the
 * pipeline records: `margin_band` is computed on the device and never submitted.
 *
 * It costs little. A page holds five to ten questions stacked vertically, so a
 * full-width band is still four to five times fewer pixels than the page, which
 * is where the latency win actually comes from.
 */
export const CROP_INSET_X = 0;

/**
 * The band of the page one region should be cut from, or null if there is
 * nothing sensible to cut.
 *
 * Null rather than a best-effort rectangle in three cases, each of which means
 * the full-page path is the correct answer and not a degradation:
 *
 * · **The region spans more than one page.** Half an answer is worse than the
 *   whole paper; a crop that silently dropped the continuation would produce a
 *   confident reading of an answer the model never saw the end of.
 * · **No usable box.** Nothing to cut against.
 * · **The band is degenerate or covers essentially the whole page.** Cutting a
 *   copy of the page and paying to store it buys nothing.
 */
export function bandForRegion(
  spans: PageSpan[],
  pageNumber: number,
  pageWidth: number,
  pageHeight: number,
): Box | null {
  if (!Array.isArray(spans) || spans.length !== 1) return null;
  const span = spans[0];
  if (!span || span.page !== pageNumber) return null;

  const box = span.box;
  if (!box) return null;
  const { x, y, w, h } = box;
  if (![x, y, w, h].every((n) => typeof n === "number" && Number.isFinite(n))) return null;
  if (w <= 0 || h <= 0) return null;

  const pad = Math.max(pageHeight * CROP_PAD_PAGE_H, h * CROP_PAD_BOX_H);
  const top = Math.max(0, Math.round(y - pad));
  const bottom = Math.min(pageHeight, Math.round(y + h + pad));

  const left = Math.max(0, Math.round(CROP_INSET_X * pageWidth));
  const right = Math.min(pageWidth, Math.round(pageWidth - CROP_INSET_X * pageWidth));

  const band = { x: left, y: top, w: right - left, h: bottom - top };
  if (band.w < 8 || band.h < 8) return null;

  // A band that is already the page is not a crop. 0.9 rather than 1.0 because
  // a region covering nine tenths of the page's height is, for every purpose
  // here, the page: the tokens saved would not pay for the second object.
  if (band.h >= pageHeight * 0.9) return null;

  return band;
}

/**
 * Copy one rectangle out of an RGBA buffer.
 *
 * No resampling — see the note at the top of this file. Row-at-a-time via
 * `subarray`/`set` rather than a per-pixel loop, because a full-width band off a
 * 2400px page is a few million pixels and this runs inside a Worker's CPU
 * budget with a WebP encode still to come.
 */
export function cutRegion(image: RgbaImage, band: Box): RgbaImage {
  const { width, height } = image;
  const x = Math.max(0, Math.min(width - 1, band.x));
  const y = Math.max(0, Math.min(height - 1, band.y));
  const w = Math.max(1, Math.min(width - x, band.w));
  const h = Math.max(1, Math.min(height - y, band.h));

  const out = new Uint8ClampedArray(w * h * 4);
  const src = image.data;
  for (let row = 0; row < h; row++) {
    const from = ((y + row) * width + x) * 4;
    out.set(src.subarray(from, from + w * 4), row * w * 4);
  }
  return { data: out, width: w, height: h };
}

/** Magic-byte sniff. The page is WebP or JPEG depending on what the student's
    browser could encode, and the mask is always PNG — but which is which is a
    fact about the bytes, not something to infer from a column. */
export function imageFormat(bytes: Uint8Array): "webp" | "jpeg" | "png" | null {
  if (bytes.length < 12) return null;
  if (bytes[0] === 0xff && bytes[1] === 0xd8 && bytes[2] === 0xff) return "jpeg";
  if (bytes[0] === 0x89 && bytes[1] === 0x50 && bytes[2] === 0x4e && bytes[3] === 0x47) return "png";
  if (
    bytes[0] === 0x52 && bytes[1] === 0x49 && bytes[2] === 0x46 && bytes[3] === 0x46 &&
    bytes[8] === 0x57 && bytes[9] === 0x45 && bytes[10] === 0x42 && bytes[11] === 0x50
  ) return "webp";
  return null;
}

function saneDimensions(width: number, height: number): { width: number; height: number } | null {
  return Number.isSafeInteger(width) && Number.isSafeInteger(height) && width > 0 && height > 0
    ? { width, height }
    : null;
}

function u24le(bytes: Uint8Array, offset: number): number {
  return bytes[offset] | (bytes[offset + 1] << 8) | (bytes[offset + 2] << 16);
}

function u32le(bytes: Uint8Array, offset: number): number {
  return (bytes[offset] | (bytes[offset + 1] << 8) | (bytes[offset + 2] << 16) | (bytes[offset + 3] << 24)) >>> 0;
}

function webpDimensions(bytes: Uint8Array): { width: number; height: number } | null {
  let offset = 12;
  while (offset + 8 <= bytes.length) {
    const chunk = String.fromCharCode(bytes[offset], bytes[offset + 1], bytes[offset + 2], bytes[offset + 3]);
    const size = u32le(bytes, offset + 4);
    const payload = offset + 8;
    if (payload + size > bytes.length) return null;

    if (chunk === "VP8X" && size >= 10) {
      return saneDimensions(1 + u24le(bytes, payload + 4), 1 + u24le(bytes, payload + 7));
    }
    if (chunk === "VP8L" && size >= 5 && bytes[payload] === 0x2f) {
      const b0 = bytes[payload + 1];
      const b1 = bytes[payload + 2];
      const b2 = bytes[payload + 3];
      const b3 = bytes[payload + 4];
      const width = 1 + b0 + ((b1 & 0x3f) << 8);
      const height = 1 + (b1 >> 6) + (b2 << 2) + ((b3 & 0x0f) << 10);
      return saneDimensions(width, height);
    }
    if (
      chunk === "VP8 " && size >= 10 &&
      bytes[payload + 3] === 0x9d && bytes[payload + 4] === 0x01 && bytes[payload + 5] === 0x2a
    ) {
      const width = (bytes[payload + 6] | (bytes[payload + 7] << 8)) & 0x3fff;
      const height = (bytes[payload + 8] | (bytes[payload + 9] << 8)) & 0x3fff;
      return saneDimensions(width, height);
    }

    offset = payload + size + (size & 1);
  }
  return null;
}

function jpegDimensions(bytes: Uint8Array): { width: number; height: number } | null {
  let offset = 2;
  while (offset + 3 < bytes.length) {
    while (offset < bytes.length && bytes[offset] !== 0xff) offset++;
    while (offset < bytes.length && bytes[offset] === 0xff) offset++;
    if (offset >= bytes.length) return null;

    const marker = bytes[offset++];
    if (marker === 0xd8 || marker === 0xd9 || marker === 0x01 || (marker >= 0xd0 && marker <= 0xd7)) continue;
    if (marker === 0xda || offset + 1 >= bytes.length) return null;

    const length = (bytes[offset] << 8) | bytes[offset + 1];
    if (length < 2 || offset + length > bytes.length) return null;

    const isSof = marker >= 0xc0 && marker <= 0xcf && marker !== 0xc4 && marker !== 0xc8 && marker !== 0xcc;
    if (isSof) {
      if (length < 7) return null;
      const height = (bytes[offset + 3] << 8) | bytes[offset + 4];
      const width = (bytes[offset + 5] << 8) | bytes[offset + 6];
      return saneDimensions(width, height);
    }
    offset += length;
  }
  return null;
}

/**
 * Read dimensions from the encoded container without allocating the decoded
 * pixel buffer. This is a security boundary for the crop worker: a forged image
 * with enormous dimensions must be rejected before its codec can expand it.
 */
export function imageDimensions(bytes: Uint8Array): { width: number; height: number } | null {
  const format = imageFormat(bytes);
  if (format === "png") {
    if (
      bytes.length < 24 ||
      bytes[12] !== 0x49 || bytes[13] !== 0x48 || bytes[14] !== 0x44 || bytes[15] !== 0x52
    ) return null;
    const width = ((bytes[16] << 24) | (bytes[17] << 16) | (bytes[18] << 8) | bytes[19]) >>> 0;
    const height = ((bytes[20] << 24) | (bytes[21] << 16) | (bytes[22] << 8) | bytes[23]) >>> 0;
    return saneDimensions(width, height);
  }
  if (format === "webp") return webpDimensions(bytes);
  if (format === "jpeg") return jpegDimensions(bytes);
  return null;
}
