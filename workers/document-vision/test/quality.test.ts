import { describe, expect, it } from "vitest";
import type { RgbaImage } from "@mastery/shared/crop.js";
import { conditionImage } from "../src/conditioning";
import { measurePixelQuality } from "../src/quality";

function image(width: number, height: number, pixel: (x: number, y: number) => [number, number, number]): RgbaImage {
  const data = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    const [red, green, blue] = pixel(x, y);
    const offset = (y * width + x) * 4;
    data[offset] = red;
    data[offset + 1] = green;
    data[offset + 2] = blue;
    data[offset + 3] = 255;
  }
  return { data, width, height };
}

describe("deterministic page quality", () => {
  it("penalizes a featureless page more than a sharp edge field", () => {
    const flat = measurePixelQuality(image(128, 128, () => [220, 220, 220]), "image/png");
    const sharp = measurePixelQuality(image(128, 128, (x, y) => (x + y) % 2 === 0 ? [0, 0, 0] : [255, 255, 255]), "image/png");
    expect(flat.blur).toBeGreaterThan(0.95);
    expect(sharp.blur).toBeLessThan(0.1);
    expect(flat.compression).toBe(0);
  });

  it("rotates pixels without inventing or resampling content", () => {
    const source = image(2, 1, (x) => x === 0 ? [10, 20, 30] : [200, 210, 220]);
    const rotated = conditionImage(source, 90);
    expect([rotated.width, rotated.height]).toEqual([1, 2]);
    expect(Array.from(rotated.data.filter((_, index) => index % 4 === 3))).toEqual([255, 255]);
  });
});
