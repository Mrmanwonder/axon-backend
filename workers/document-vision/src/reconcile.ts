import type { RgbaImage } from "@mastery/shared/crop.js";
import { inkSignals, measureRegionFeatures, type PixelQualityMetrics } from "./quality";
import type { Box, ReaderOutput, ReaderRegion, VisionAnalysis } from "./schema";

interface NamedRead { id: string; output: ReaderOutput }

const clamp = (value: number): number => Math.max(0, Math.min(1, value));

function normalizedBox(box: Box): Box {
  const x = Math.min(0.9999, clamp(box.x));
  const y = Math.min(0.9999, clamp(box.y));
  return { x, y, width: Math.max(0.0001, Math.min(box.width, 1 - x)), height: Math.max(0.0001, Math.min(box.height, 1 - y)) };
}

function intersectionOverUnion(left: Box, right: Box): number {
  const x1 = Math.max(left.x, right.x);
  const y1 = Math.max(left.y, right.y);
  const x2 = Math.min(left.x + left.width, right.x + right.width);
  const y2 = Math.min(left.y + left.height, right.y + right.height);
  const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const union = left.width * left.height + right.width * right.height - intersection;
  return union <= 0 ? 0 : intersection / union;
}

function centerDistance(left: Box, right: Box): number {
  return Math.hypot(left.x + left.width / 2 - right.x - right.width / 2, left.y + left.height / 2 - right.y - right.height / 2);
}

function averageBox(left: Box, right: Box): Box {
  return normalizedBox({
    x: (left.x + right.x) / 2,
    y: (left.y + right.y) / 2,
    width: (left.width + right.width) / 2,
    height: (left.height + right.height) / 2
  });
}

function normalizedText(value: string | null): string | null {
  if (value === null) return null;
  const normalized = value.normalize("NFKC").replace(/\s+/g, " ").trim();
  return normalized.length > 0 ? normalized : null;
}

function asRead(region: ReaderRegion, readerId: string) {
  const value = normalizedText(region.text);
  return {
    value,
    alternatives: [] as string[],
    status: value === null ? (region.confidence < 0.3 ? "unreadable" as const : "ambiguous" as const) : "read" as const,
    region: normalizedBox(region.box),
    readerIds: [readerId]
  };
}

export function reconcileReaders(readers: readonly NamedRead[], qualityMetrics: PixelQualityMetrics, image: RgbaImage): VisionAnalysis {
  if (readers.length !== 2) throw new Error("TWO_INDEPENDENT_READERS_REQUIRED");
  const [primary, secondary] = readers;
  const used = new Set<number>();
  const candidates: Array<{ primary?: ReaderRegion; secondary?: ReaderRegion; spatial: number }> = [];

  for (const region of primary.output.regions) {
    let bestIndex = -1;
    let bestScore = -1;
    for (const [index, other] of secondary.output.regions.entries()) {
      if (used.has(index) || other.class !== region.class) continue;
      const overlap = intersectionOverUnion(region.box, other.box);
      const distance = centerDistance(region.box, other.box);
      const score = overlap - distance * 0.25;
      if ((overlap >= 0.2 || distance <= 0.06) && score > bestScore) { bestIndex = index; bestScore = score; }
    }
    if (bestIndex >= 0) {
      used.add(bestIndex);
      const other = secondary.output.regions[bestIndex];
      candidates.push({ primary: region, secondary: other, spatial: clamp(Math.max(0, intersectionOverUnion(region.box, other.box)) + 0.35) });
    } else candidates.push({ primary: region, spatial: 0 });
  }
  for (const [index, region] of secondary.output.regions.entries()) if (!used.has(index)) candidates.push({ secondary: region, spatial: 0 });

  const geometryAgreement = primary.output.orientationDegrees === secondary.output.orientationDegrees &&
    Math.abs(primary.output.perspectiveDegrees - secondary.output.perspectiveDegrees) <= 3 &&
    Math.abs(primary.output.cropCompleteness - secondary.output.cropCompleteness) <= 0.12;

  candidates.sort((a, b) => {
    const left = a.primary ?? a.secondary!;
    const right = b.primary ?? b.secondary!;
    return left.box.y - right.box.y || left.box.x - right.box.x;
  });

  const regions: VisionAnalysis["regions"] = [];
  const reads: VisionAnalysis["reads"] = [];
  for (const [index, candidate] of candidates.entries()) {
    const first = candidate.primary ?? candidate.secondary!;
    const second = candidate.primary && candidate.secondary ? candidate.secondary : undefined;
    const box = second ? averageBox(first.box, second.box) : normalizedBox(first.box);
    const id = `r-${String(index + 1).padStart(4, "0")}`;
    const agreementConfidence = second
      ? Math.min(first.confidence, second.confidence, candidate.spatial, geometryAgreement ? 1 : 0.65)
      : Math.min(first.confidence, 0.49);
    const firstText = normalizedText(first.text);
    const secondText = second ? normalizedText(second.text) : null;
    const agreedText = second && firstText !== null && firstText === secondText ? firstText : undefined;
    const layersAgree = second !== undefined && first.layer === second.layer && first.layer !== "UNKNOWN";
    regions.push({
      id,
      class: first.class,
      box,
      confidence: agreementConfidence,
      ...(agreedText !== undefined ? { text: agreedText } : {}),
      ...(layersAgree ? { inkSignals: inkSignals(first.layer, box, measureRegionFeatures(image, box)) } : {})
    });
    const regionReads = candidate.primary ? [asRead(candidate.primary, primary.id)] : [];
    if (candidate.secondary) regionReads.push(asRead(candidate.secondary, secondary.id));
    reads.push({ regionId: id, reads: regionReads });
  }

  return {
    qualityMetrics,
    orientationDegrees: primary.output.orientationDegrees === secondary.output.orientationDegrees ? primary.output.orientationDegrees : 0,
    regions,
    reads
  };
}
