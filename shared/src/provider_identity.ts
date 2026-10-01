/**
 * Curriculum provider identity from stored student columns.
 *
 * AXO-94 invariant: missing or unrecognised identity is unknown (null). It must
 * never become Cambridge. Only the explicit legacy Cambridge board values map to
 * "cambridge"; a programme key resolves by its own prefix.
 */
export type ProviderKey = "cambridge" | "cbse" | "ib";

const LEGACY_CAMBRIDGE_BOARDS: ReadonlySet<string> = new Set(["CAIE", "IGCSE", "AS_A_LEVEL"]);

export function providerKeyForBoard(board: unknown): ProviderKey | null {
  if (board === "CBSE") return "cbse";
  if (board === "IBDP") return "ib";
  if (typeof board === "string" && LEGACY_CAMBRIDGE_BOARDS.has(board)) return "cambridge";
  return null;
}

export function providerKeyForProgramme(key: unknown): ProviderKey | null {
  if (typeof key !== "string") return null;
  if (key.startsWith("cambridge_")) return "cambridge";
  if (key.startsWith("cbse_")) return "cbse";
  if (key === "ibdp") return "ib";
  return null;
}
