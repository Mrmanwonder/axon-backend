import type { Box } from "../../schemas";
import type { PageQualityClass } from "../quality";

export type RecognitionPath = "PRINTED_OCR" | "HANDWRITING_ENSEMBLE" | "INDEPENDENT_READER" | "RESCAN";
export interface RegionReadRequest { pageId: string; region: Box; layer: "PRINTED" | "STUDENT" | "TEACHER" | "UNKNOWN"; quality: PageQualityClass; subject?: string; contextRegionIds: string[] }
export interface RegionRead { value: string | null; alternatives: string[]; status: "read" | "ambiguous" | "unreadable"; region: Box; readerIds: string[] }

export function recognitionPath(request: RegionReadRequest): RecognitionPath {
  if (request.quality === "UNREADABLE") return "RESCAN";
  if (request.quality === "AMBIGUOUS") return "INDEPENDENT_READER";
  if (request.layer === "PRINTED") return "PRINTED_OCR";
  return "HANDWRITING_ENSEMBLE";
}

export function reconcileReads(reads: readonly RegionRead[]): RegionRead {
  if (reads.length === 0) throw new Error("At least one read is required");
  const seenReaders = new Set<string>();
  const independent = reads.filter((read) => {
    const fresh = read.readerIds.some((readerId) => !seenReaders.has(readerId));
    for (const readerId of read.readerIds) seenReaders.add(readerId);
    return fresh;
  });
  const usable = independent.filter((read) => read.status === "read" && read.value !== null);
  const counts = new Map<string, number>();
  for (const read of usable) counts.set(read.value!, (counts.get(read.value!) ?? 0) + 1);
  const ranked = [...counts.entries()].sort((a, b) => b[1] - a[1]);
  const first = reads[0];
  if (ranked[0] && ranked[0][1] >= 2 && (!ranked[1] || ranked[0][1] > ranked[1][1])) {
    return { value: ranked[0][0], alternatives: ranked.slice(1).map(([value]) => value), status: "read", region: first.region, readerIds: [...seenReaders] };
  }
  return { value: null, alternatives: [...new Set(independent.flatMap((read) => read.value ? [read.value, ...read.alternatives] : read.alternatives))], status: independent.every((read) => read.status === "unreadable") ? "unreadable" : "ambiguous", region: first.region, readerIds: [...seenReaders] };
}
