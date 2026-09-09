// Small shared contract pieces used across the pipeline's HTTP and model
// boundaries.

/** Bumped whenever the shape submit_paper() writes changes. Passed through to it so old and new clients stay distinguishable in the data. */
export const PIPELINE_VERSION = "1.0.0";

export interface RawBox {
  x?: unknown;
  y?: unknown;
  w?: unknown;
  h?: unknown;
  page?: unknown;
}

export interface PixelBox {
  page: number;
  x: number;
  y: number;
  w: number;
  h: number;
}

function hasProvenance(box: unknown, page: number): box is Required<RawBox> {
  const b = box as RawBox | null | undefined;
  return (
    !!b &&
    [b.x, b.y, b.w, b.h].every((n) => typeof n === "number" && Number.isFinite(n)) &&
    (b.w as number) > 0 &&
    (b.h as number) > 0 &&
    // Lower bounds, which were missing. Only the far edges were checked, so a
    // negative x or y passed straight through and was multiplied into a
    // negative pixel coordinate — a highlight off the top-left of the page,
    // stored as if it were provenance. The model does occasionally return one.
    (b.x as number) >= 0 &&
    (b.y as number) >= 0 &&
    ((b.page as number) ?? page) > 0
  );
}

/**
 * Converts a model-returned box on the 0-1000 normalised grid into pixel
 * coordinates on the actual page image, rejecting anything that isn't a
 * real, in-bounds box. A box the model couldn't ground in the image comes
 * back null rather than a best guess — see CLAUDE.md rule 4.
 */
export function takeBox(raw: unknown, page: number, pageW: number, pageH: number): PixelBox | null {
  if (!hasProvenance(raw, page)) return null;
  const b = raw as Required<RawBox> & { x: number; y: number; w: number; h: number; page: number };
  if (b.x > 1000 || b.y > 1000 || b.x + b.w > 1002 || b.y + b.h > 1002) return null;
  return {
    page: b.page ?? page,
    x: Math.round((b.x / 1000) * pageW),
    y: Math.round((b.y / 1000) * pageH),
    w: Math.max(1, Math.round((b.w / 1000) * pageW)),
    h: Math.max(1, Math.round((b.h / 1000) * pageH)),
  };
}
