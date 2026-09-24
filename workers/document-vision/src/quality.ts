import type { RgbaImage } from "@mastery/shared/crop.js";
import type { Box, InkLayer } from "./schema";

export interface PixelQualityMetrics {
  blur: number;
  glareFraction: number;
  perspectiveDegrees: number;
  resolution: number;
  compression: number;
  cropCompleteness: number;
  shadowFraction: number;
}

export interface RegionPixelFeatures {
  chroma: number;
  edgeDensity: number;
  darkness: number;
}

const clamp = (value: number): number => Math.max(0, Math.min(1, value));
const luminance = (data: Uint8Array | Uint8ClampedArray, offset: number): number =>
  data[offset] * 0.2126 + data[offset + 1] * 0.7152 + data[offset + 2] * 0.0722;

function median(values: readonly number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0 ? ((sorted[middle - 1] ?? 0) + (sorted[middle] ?? 0)) / 2 : (sorted[middle] ?? 0);
}

function regressionSlope(points: ReadonlyArray<{ x: number; y: number }>): number | null {
  if (points.length < 4) return null;
  const meanX = points.reduce((sum, point) => sum + point.x, 0) / points.length;
  const meanY = points.reduce((sum, point) => sum + point.y, 0) / points.length;
  let numerator = 0;
  let denominator = 0;
  for (const point of points) {
    numerator += (point.y - meanY) * (point.x - meanX);
    denominator += (point.y - meanY) ** 2;
  }
  return denominator === 0 ? null : numerator / denominator;
}

function estimateSkew(image: RgbaImage, darkThreshold: number): number {
  const bins = 32;
  const left: Array<{ x: number; y: number }> = [];
  const right: Array<{ x: number; y: number }> = [];
  for (let bin = 0; bin < bins; bin++) {
    const yStart = Math.floor((bin / bins) * image.height);
    const yEnd = Math.max(yStart + 1, Math.floor(((bin + 1) / bins) * image.height));
    let minimum = image.width;
    let maximum = -1;
    const yStep = Math.max(1, Math.ceil((yEnd - yStart) / 8));
    const xStep = Math.max(1, Math.ceil(image.width / 600));
    for (let y = yStart; y < yEnd; y += yStep) {
      for (let x = 0; x < image.width; x += xStep) {
        if (luminance(image.data, (y * image.width + x) * 4) < darkThreshold) {
          minimum = Math.min(minimum, x);
          maximum = Math.max(maximum, x);
        }
      }
    }
    if (maximum >= 0 && minimum < image.width && maximum - minimum > image.width * 0.1) {
      const y = (yStart + yEnd) / 2 / image.height;
      left.push({ x: minimum / image.width, y });
      right.push({ x: maximum / image.width, y });
    }
  }
  const slopes = [regressionSlope(left), regressionSlope(right)].filter((value): value is number => value !== null);
  if (slopes.length === 0) return 0;
  const slope = slopes.reduce((sum, value) => sum + value, 0) / slopes.length;
  return Math.max(-45, Math.min(45, Math.atan(slope) * 180 / Math.PI));
}

function compressionPenalty(image: RgbaImage, mimeType: string, sampleStep: number): number {
  if (mimeType !== "image/jpeg") return 0;
  let boundary = 0;
  let boundaryCount = 0;
  let interior = 0;
  let interiorCount = 0;
  const yStep = Math.max(1, sampleStep);
  for (let y = yStep; y < image.height; y += yStep) {
    for (let x = 8; x < image.width; x += 8) {
      const atBoundary = luminance(image.data, (y * image.width + x) * 4);
      const beforeBoundary = luminance(image.data, (y * image.width + x - 1) * 4);
      boundary += Math.abs(atBoundary - beforeBoundary);
      boundaryCount++;
      if (x >= 5) {
        const interiorRight = luminance(image.data, (y * image.width + x - 4) * 4);
        const interiorLeft = luminance(image.data, (y * image.width + x - 5) * 4);
        interior += Math.abs(interiorRight - interiorLeft);
        interiorCount++;
      }
    }
  }
  const boundaryMean = boundary / Math.max(1, boundaryCount);
  const interiorMean = interior / Math.max(1, interiorCount);
  return clamp((boundaryMean / Math.max(1, interiorMean) - 1) / 1.5);
}

export function measurePixelQuality(image: RgbaImage, mimeType: string): PixelQualityMetrics {
  const sampleStep = Math.max(1, Math.ceil(Math.sqrt((image.width * image.height) / 250_000)));
  let count = 0;
  let sum = 0;
  let sumSquares = 0;
  let clipped = 0;
  let gradientSquares = 0;
  let gradientCount = 0;
  let edgeDark = 0;
  let edgeCount = 0;
  const cellSums = new Array<number>(64).fill(0);
  const cellCounts = new Array<number>(64).fill(0);

  for (let y = 0; y < image.height; y += sampleStep) {
    for (let x = 0; x < image.width; x += sampleStep) {
      const offset = (y * image.width + x) * 4;
      const value = luminance(image.data, offset);
      count++;
      sum += value;
      sumSquares += value * value;
      if (value >= 252 && Math.max(image.data[offset], image.data[offset + 1], image.data[offset + 2]) - Math.min(image.data[offset], image.data[offset + 1], image.data[offset + 2]) <= 6) clipped++;
      if (x + sampleStep < image.width && y + sampleStep < image.height) {
        const right = luminance(image.data, (y * image.width + x + sampleStep) * 4);
        const below = luminance(image.data, ((y + sampleStep) * image.width + x) * 4);
        gradientSquares += (right - value) ** 2 + (below - value) ** 2;
        gradientCount += 2;
      }
      const nearEdge = x < image.width * 0.035 || x >= image.width * 0.965 || y < image.height * 0.035 || y >= image.height * 0.965;
      if (nearEdge) {
        edgeCount++;
        if (value < 165) edgeDark++;
      }
      const cellX = Math.min(7, Math.floor((x / image.width) * 8));
      const cellY = Math.min(7, Math.floor((y / image.height) * 8));
      const cell = cellY * 8 + cellX;
      cellSums[cell] += value;
      cellCounts[cell]++;
    }
  }

  const mean = sum / Math.max(1, count);
  const variance = Math.max(0, sumSquares / Math.max(1, count) - mean * mean);
  const rmsGradient = Math.sqrt(gradientSquares / Math.max(1, gradientCount));
  const cellMeans = cellSums.map((value, index) => value / Math.max(1, cellCounts[index]));
  const cellMedian = median(cellMeans);
  const shadowCells = cellMeans.filter((value) => value < cellMedian - Math.max(18, Math.sqrt(variance) * 0.65)).length;
  const darkThreshold = Math.min(170, mean - Math.max(20, Math.sqrt(variance) * 0.4));

  return {
    blur: clamp(1 - rmsGradient / 42),
    glareFraction: clamp(((clipped / Math.max(1, count)) - 0.55) / 0.35),
    perspectiveDegrees: estimateSkew(image, darkThreshold),
    resolution: clamp(Math.min(image.width / 1200, image.height / 1600)),
    compression: compressionPenalty(image, mimeType, sampleStep),
    cropCompleteness: clamp(1 - (edgeDark / Math.max(1, edgeCount)) * 5),
    shadowFraction: clamp(shadowCells / cellMeans.length)
  };
}

export function measureRegionFeatures(image: RgbaImage, box: Box): RegionPixelFeatures {
  const left = Math.max(0, Math.floor(box.x * image.width));
  const top = Math.max(0, Math.floor(box.y * image.height));
  const right = Math.min(image.width, Math.ceil((box.x + box.width) * image.width));
  const bottom = Math.min(image.height, Math.ceil((box.y + box.height) * image.height));
  const step = Math.max(1, Math.ceil(Math.sqrt(Math.max(1, (right - left) * (bottom - top)) / 20_000)));
  let count = 0;
  let chroma = 0;
  let darkness = 0;
  let edges = 0;
  for (let y = top; y < bottom; y += step) {
    for (let x = left; x < right; x += step) {
      const offset = (y * image.width + x) * 4;
      const red = image.data[offset];
      const green = image.data[offset + 1];
      const blue = image.data[offset + 2];
      const value = luminance(image.data, offset);
      count++;
      chroma += (Math.max(red, green, blue) - Math.min(red, green, blue)) / 255;
      darkness += 1 - value / 255;
      if (x + step < right) {
        const adjacent = luminance(image.data, (y * image.width + x + step) * 4);
        if (Math.abs(adjacent - value) > 35) edges++;
      }
    }
  }
  return { chroma: clamp(chroma / Math.max(1, count)), edgeDensity: clamp(edges / Math.max(1, count)), darkness: clamp(darkness / Math.max(1, count)) };
}

export function inkSignals(layer: InkLayer, box: Box, features: RegionPixelFeatures) {
  const marginTendency = clamp(Math.max(0, 0.16 - box.x) / 0.16 + Math.max(0, box.x + box.width - 0.84) / 0.16);
  const printedProbability = layer === "PRINTED" ? 0.94 : layer === "UNKNOWN" ? 0.5 : 0.12;
  const handwriting = layer === "PRINTED" ? 0.08 : layer === "UNKNOWN" ? 0.5 : 0.9;
  return {
    printedProbability: clamp(printedProbability + (0.08 - features.chroma) * 0.2),
    colourDistanceFromPrint: clamp(features.chroma * 2.5),
    strokeDifference: clamp(handwriting * 0.7 + features.edgeDensity * 0.3),
    marginTendency,
    annotationOverlap: layer === "TEACHER" ? 0.9 : layer === "UNKNOWN" ? 0.5 : 0.1,
    handwritingDifference: layer === "TEACHER" ? 0.9 : layer === "STUDENT" ? 0.15 : layer === "PRINTED" ? 0.55 : 0.5
  };
}
