import { cutRegion, imageDimensions, imageFormat, type RgbaImage } from "@mastery/shared/crop.js";
import { canConditionSafely, conditionImage } from "./conditioning";
import { measurePixelQuality, type PixelQualityMetrics } from "./quality";
import type { PageLayoutReader, TargetedRegionContext, TargetedRegionReader } from "./readers";
import { reconcileTargetedRegion, retainLayoutRegion } from "./reconcile";
import {
  VisionAnalysisSchema, parseSchema,
  type Box, type ReaderRegion, type VisionAnalysis, type VisionRequest
} from "./schema";

const MAX_IMAGE_BYTES = 12_000_000;
const MAX_PIXELS = 10_000_000;
const MAX_TARGETED_REGIONS = 120;
const TARGETED_READER_CONCURRENCY = 4;
const READABLE_CLASSES = new Set<ReaderRegion["class"]>([
  "question_number", "subquestion_number", "printed_question", "student_answer", "teacher_annotation",
  "teacher_comment", "marginal_mark", "marks_available", "reported_total", "page_number"
]);

export function decodeBase64(value: string): Uint8Array {
  if (value.length % 4 !== 0 || !/^[A-Za-z0-9+/]*={0,2}$/.test(value)) throw new Error("INVALID_BASE64");
  let binary: string;
  try { binary = atob(value); }
  catch { throw new Error("INVALID_BASE64"); }
  if (binary.length === 0 || binary.length > MAX_IMAGE_BYTES) throw new Error("IMAGE_SIZE_INVALID");
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index++) bytes[index] = binary.charCodeAt(index);
  return bytes;
}

function base64(bytes: Uint8Array): string {
  let result = "";
  for (let offset = 0; offset < bytes.length; offset += 0x8000) result += String.fromCharCode(...bytes.subarray(offset, offset + 0x8000));
  return btoa(result);
}

function expectedFormat(mimeType: VisionRequest["mimeType"]): "jpeg" | "png" | "webp" {
  return mimeType === "image/jpeg" ? "jpeg" : mimeType === "image/png" ? "png" : "webp";
}

function cropBox(box: Box, image: RgbaImage): { x: number; y: number; w: number; h: number } {
  const padX = Math.max(0.012, box.width * 0.2);
  const padY = Math.max(0.012, box.height * 0.25);
  const left = Math.max(0, Math.floor((box.x - padX) * image.width));
  const top = Math.max(0, Math.floor((box.y - padY) * image.height));
  const right = Math.min(image.width, Math.ceil((box.x + box.width + padX) * image.width));
  const bottom = Math.min(image.height, Math.ceil((box.y + box.height + padY) * image.height));
  return { x: left, y: top, w: Math.max(1, right - left), h: Math.max(1, bottom - top) };
}

function relatedText(regions: readonly ReaderRegion[], target: ReaderRegion): string[] {
  const targetCentre = target.box.y + target.box.height / 2;
  return regions
    .filter((region) => region !== target && region.text !== null)
    .map((region) => ({ region, distance: Math.abs(region.box.y + region.box.height / 2 - targetCentre) }))
    .sort((left, right) => left.distance - right.distance)
    .slice(0, 6)
    .map(({ region }) => `${region.class}: ${region.text!.slice(0, 160)}`);
}

async function mapBounded<T, U>(items: readonly T[], concurrency: number, task: (item: T, index: number) => Promise<U>): Promise<U[]> {
  const output = new Array<U>(items.length);
  let next = 0;
  const worker = async (): Promise<void> => {
    for (;;) {
      const index = next++;
      if (index >= items.length) return;
      output[index] = await task(items[index], index);
    }
  };
  await Promise.all(Array.from({ length: Math.min(concurrency, items.length) }, worker));
  return output;
}

interface TargetedResult {
  index: number;
  value: ReturnType<typeof reconcileTargetedRegion>;
}

async function readTargetedRegions(input: {
  regions: readonly ReaderRegion[];
  layoutReaderId: string;
  targetedReader: TargetedRegionReader;
  image: RgbaImage;
  quality: PixelQualityMetrics;
  encode: (image: RgbaImage) => Promise<Uint8Array>;
}): Promise<TargetedResult[]> {
  let encodeLock = Promise.resolve();
  const encodeSerially = async (crop: RgbaImage): Promise<Uint8Array> => {
    const previous = encodeLock;
    let release = (): void => undefined;
    encodeLock = new Promise<void>((resolve) => { release = resolve; });
    await previous;
    try { return await input.encode(crop); }
    finally { release(); }
  };
  const targets = input.regions
    .map((region, index) => ({ region, index }))
    .filter(({ region }) => READABLE_CLASSES.has(region.class))
    .slice(0, MAX_TARGETED_REGIONS);
  return mapBounded(targets, TARGETED_READER_CONCURRENCY, async ({ region, index }) => {
    const crop = cutRegion(input.image, cropBox(region.box, input.image));
    const encoded = await encodeSerially(crop);
    const context: TargetedRegionContext = {
      regionId: `r-${String(index + 1).padStart(4, "0")}`,
      expectedClass: region.class,
      expectedLayer: region.layer,
      pageBox: region.box,
      relatedText: relatedText(input.regions, region),
      quality: input.quality
    };
    const targeted = await input.targetedReader.readRegion(`data:image/webp;base64,${base64(encoded)}`, context, 17_000);
    return {
      index,
      value: reconcileTargetedRegion({
        region,
        targeted,
        layoutReaderId: input.layoutReaderId,
        targetedReaderId: input.targetedReader.id,
        image: input.image,
        id: context.regionId
      })
    };
  });
}

export async function analyzeDocument(
  request: VisionRequest,
  layoutReader: PageLayoutReader,
  targetedReader: TargetedRegionReader,
  decode: (bytes: Uint8Array) => Promise<RgbaImage>,
  encode: (image: RgbaImage) => Promise<Uint8Array>
): Promise<VisionAnalysis> {
  const bytes = decodeBase64(request.dataBase64);
  if (imageFormat(bytes) !== expectedFormat(request.mimeType)) throw new Error("MIME_TYPE_MISMATCH");
  const dimensions = imageDimensions(bytes);
  if (!dimensions) throw new Error("INVALID_IMAGE_HEADER");
  if (dimensions.width * dimensions.height > MAX_PIXELS) throw new Error("IMAGE_PIXEL_LIMIT_EXCEEDED");
  const image = await decode(bytes);
  if (image.width !== dimensions.width || image.height !== dimensions.height) throw new Error("IMAGE_DIMENSION_MISMATCH");
  const quality = measurePixelQuality(image, request.mimeType);
  const page = await layoutReader.discoverPage(`data:${request.mimeType};base64,${request.dataBase64}`, 17_000);
  const targeted = await readTargetedRegions({ regions: page.regions, layoutReaderId: layoutReader.id, targetedReader, image, quality, encode });
  const targetedByIndex = new Map(targeted.map((result) => [result.index, result.value]));
  const regions: VisionAnalysis["regions"] = [];
  const reads: VisionAnalysis["reads"] = [];
  for (const [index, region] of page.regions.entries()) {
    const id = `r-${String(index + 1).padStart(4, "0")}`;
    const result = targetedByIndex.get(index);
    if (result) {
      regions.push(result.region);
      reads.push(result.reads);
    } else {
      regions.push(retainLayoutRegion(region, image, id));
    }
  }
  const analysis: VisionAnalysis = {
    qualityMetrics: quality,
    orientationDegrees: page.orientationDegrees,
    regions,
    reads
  };
  const orientation = analysis.orientationDegrees as 0 | 90 | 180 | 270;
  if (canConditionSafely(quality, orientation)) {
    const conditioned = await encode(conditionImage(image, orientation));
    analysis.conditionedImageBase64 = base64(conditioned);
    analysis.conditionedImageMimeType = "image/webp";
  }
  return parseSchema(VisionAnalysisSchema, analysis);
}
