import type { RgbaImage } from "@mastery/shared/crop.js";
import { inkSignals, measureRegionFeatures } from "./quality";
import type { ReaderRegion, TargetedRegionRead, VisionAnalysis } from "./schema";

const clamp = (value: number): number => Math.max(0, Math.min(1, value));

function normalizedText(value: string | null): string | null {
  if (value === null) return null;
  const normalized = value.normalize("NFKC").replace(/\s+/g, " ").trim();
  return normalized.length > 0 ? normalized : null;
}

function layoutRead(region: ReaderRegion, readerId: string): VisionAnalysis["reads"][number]["reads"][number] {
  const value = normalizedText(region.text);
  return {
    value,
    alternatives: [],
    status: value !== null ? "read" : region.confidence < 0.3 ? "unreadable" : "ambiguous",
    region: region.box,
    readerIds: [readerId]
  };
}

function targetedRead(region: ReaderRegion, read: TargetedRegionRead, readerId: string): VisionAnalysis["reads"][number]["reads"][number] {
  return {
    value: read.status === "read" ? normalizedText(read.value) : null,
    alternatives: [...new Set(read.alternatives.map((item) => item.normalize("NFKC").replace(/\s+/g, " ").trim()).filter(Boolean))],
    status: read.status,
    region: region.box,
    readerIds: [readerId]
  };
}

export function reconcileTargetedRegion(input: {
  region: ReaderRegion;
  targeted: TargetedRegionRead;
  layoutReaderId: string;
  targetedReaderId: string;
  image: RgbaImage;
  id: string;
}): { region: VisionAnalysis["regions"][number]; reads: VisionAnalysis["reads"][number] } {
  const classAgrees = input.region.class === input.targeted.class;
  const layerAgrees = input.region.layer === input.targeted.layer && input.region.layer !== "UNKNOWN";
  const firstValue = normalizedText(input.region.text);
  const secondValue = normalizedText(input.targeted.value);
  const textAgrees = input.targeted.status === "read" && firstValue !== null && firstValue === secondValue;
  const statusConfidence = input.targeted.status === "read" ? 1 : input.targeted.status === "ambiguous" ? 0.69 : 0.49;
  const confidence = Math.min(
    input.region.confidence,
    input.targeted.confidence,
    classAgrees ? 1 : 0.49,
    layerAgrees ? 1 : 0.69,
    textAgrees ? 1 : 0.79,
    statusConfidence
  );
  const region: VisionAnalysis["regions"][number] = {
    id: input.id,
    class: input.region.class,
    box: input.region.box,
    confidence: clamp(confidence),
    ...(textAgrees && firstValue !== null ? { text: firstValue } : {}),
    ...(layerAgrees ? { inkSignals: inkSignals(input.region.layer, input.region.box, measureRegionFeatures(input.image, input.region.box)) } : {})
  };
  return {
    region,
    reads: {
      regionId: input.id,
      reads: [
        layoutRead(input.region, input.layoutReaderId),
        targetedRead(input.region, input.targeted, input.targetedReaderId)
      ]
    }
  };
}

export function retainLayoutRegion(region: ReaderRegion, image: RgbaImage, id: string): VisionAnalysis["regions"][number] {
  const text = normalizedText(region.text);
  return {
    id,
    class: region.class,
    box: region.box,
    confidence: region.confidence,
    ...(text !== null ? { text } : {}),
    ...(region.layer !== "UNKNOWN" ? { inkSignals: inkSignals(region.layer, region.box, measureRegionFeatures(image, region.box)) } : {})
  };
}
