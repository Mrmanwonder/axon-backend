import type { RgbaImage } from "@mastery/shared/crop.js";
import type { PixelQualityMetrics } from "./quality";

const clampByte = (value: number): number => Math.max(0, Math.min(255, Math.round(value)));

function rotate(image: RgbaImage, degrees: 0 | 90 | 180 | 270): RgbaImage {
  if (degrees === 0) return { data: new Uint8ClampedArray(image.data), width: image.width, height: image.height };
  const width = degrees === 180 ? image.width : image.height;
  const height = degrees === 180 ? image.height : image.width;
  const output = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < image.height; y++) {
    for (let x = 0; x < image.width; x++) {
      let targetX: number;
      let targetY: number;
      if (degrees === 90) { targetX = image.height - 1 - y; targetY = x; }
      else if (degrees === 180) { targetX = image.width - 1 - x; targetY = image.height - 1 - y; }
      else { targetX = y; targetY = image.width - 1 - x; }
      const source = (y * image.width + x) * 4;
      const target = (targetY * width + targetX) * 4;
      output[target] = image.data[source];
      output[target + 1] = image.data[source + 1];
      output[target + 2] = image.data[source + 2];
      output[target + 3] = image.data[source + 3];
    }
  }
  return { data: output, width, height };
}

function normalizeContrast(image: RgbaImage): RgbaImage {
  const histogram = new Uint32Array(256);
  for (let offset = 0; offset < image.data.length; offset += 4) {
    const value = Math.round(image.data[offset] * 0.2126 + image.data[offset + 1] * 0.7152 + image.data[offset + 2] * 0.0722);
    histogram[value]++;
  }
  const pixelCount = image.width * image.height;
  const lowTarget = pixelCount * 0.01;
  const highTarget = pixelCount * 0.99;
  let cumulative = 0;
  let low = 0;
  let high = 255;
  for (let value = 0; value < 256; value++) {
    cumulative += histogram[value];
    if (cumulative >= lowTarget) { low = value; break; }
  }
  cumulative = 0;
  for (let value = 0; value < 256; value++) {
    cumulative += histogram[value];
    if (cumulative >= highTarget) { high = value; break; }
  }
  if (high - low < 40) return image;
  const output = new Uint8ClampedArray(image.data.length);
  const scale = 235 / (high - low);
  for (let offset = 0; offset < image.data.length; offset += 4) {
    const oldLuminance = image.data[offset] * 0.2126 + image.data[offset + 1] * 0.7152 + image.data[offset + 2] * 0.0722;
    const newLuminance = 10 + (oldLuminance - low) * scale;
    const delta = newLuminance - oldLuminance;
    output[offset] = clampByte(image.data[offset] + delta);
    output[offset + 1] = clampByte(image.data[offset + 1] + delta);
    output[offset + 2] = clampByte(image.data[offset + 2] + delta);
    output[offset + 3] = image.data[offset + 3];
  }
  return { data: output, width: image.width, height: image.height };
}

export function canConditionSafely(metrics: PixelQualityMetrics, orientation: number): boolean {
  const actionable = orientation !== 0 || metrics.compression >= 0.12 || metrics.shadowFraction >= 0.03;
  return actionable && metrics.blur < 0.35 && metrics.glareFraction < 0.05 && Math.abs(metrics.perspectiveDegrees) < 0.75 && metrics.resolution >= 0.6 && metrics.cropCompleteness >= 0.8;
}

export function conditionImage(image: RgbaImage, orientation: 0 | 90 | 180 | 270): RgbaImage {
  return normalizeContrast(rotate(image, orientation));
}
