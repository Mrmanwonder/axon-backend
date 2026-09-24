import { imageDimensions, imageFormat, type RgbaImage } from "@mastery/shared/crop.js";
import { canConditionSafely, conditionImage } from "./conditioning";
import { measurePixelQuality } from "./quality";
import type { VisionReader } from "./readers";
import { reconcileReaders } from "./reconcile";
import { VisionAnalysisSchema, parseSchema, type VisionAnalysis, type VisionRequest } from "./schema";

const MAX_IMAGE_BYTES = 12_000_000;
const MAX_PIXELS = 10_000_000;

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

export async function analyzeDocument(
  request: VisionRequest,
  readers: readonly VisionReader[],
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
  const dataUrl = `data:${request.mimeType};base64,${request.dataBase64}`;
  const outputs = await Promise.all(readers.map(async (reader) => ({ id: reader.id, output: await reader.read(dataUrl, 17_000) })));
  const analysis = reconcileReaders(outputs, quality, image);
  const orientation = analysis.orientationDegrees as 0 | 90 | 180 | 270;
  if (canConditionSafely(quality, orientation)) {
    const conditioned = await encode(conditionImage(image, orientation));
    analysis.conditionedImageBase64 = base64(conditioned);
    analysis.conditionedImageMimeType = "image/webp";
  }
  return parseSchema(VisionAnalysisSchema, analysis);
}
