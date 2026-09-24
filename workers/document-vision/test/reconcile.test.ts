import { describe, expect, it } from "vitest";
import type { RgbaImage } from "@mastery/shared/crop.js";
import { reconcileReaders } from "../src/reconcile";
import type { ReaderOutput } from "../src/schema";

const pixels: RgbaImage = { data: new Uint8ClampedArray(100 * 100 * 4).fill(220), width: 100, height: 100 };
const quality = { blur: 0.1, glareFraction: 0, perspectiveDegrees: 0, resolution: 1, compression: 0, cropCompleteness: 1, shadowFraction: 0 };

function output(text: string | null, x = 0.1, layer: ReaderOutput["regions"][number]["layer"] = "PRINTED"): ReaderOutput {
  return {
    orientationDegrees: 0,
    perspectiveDegrees: 0,
    cropCompleteness: 1,
    regions: [{ class: "printed_question", box: { x, y: 0.1, width: 0.5, height: 0.1 }, confidence: 0.96, text, layer }]
  };
}

describe("independent-reader reconciliation", () => {
  it("exposes two independent reads and consensus text only on agreement", () => {
    const result = reconcileReaders([{ id: "gemini", output: output("Solve x + 1 = 2") }, { id: "moondream", output: output("Solve x + 1 = 2", 0.11) }], quality, pixels);
    expect(result.regions).toHaveLength(1);
    expect(result.regions[0]?.text).toBe("Solve x + 1 = 2");
    expect(result.reads[0]?.reads.map((read) => read.readerIds[0])).toEqual(["gemini", "moondream"]);
  });

  it("does not collapse conflicting readings into a trusted value", () => {
    const result = reconcileReaders([{ id: "gemini", output: output("12") }, { id: "moondream", output: output("17") }], quality, pixels);
    expect(result.regions[0]?.text).toBeUndefined();
    expect(result.reads[0]?.reads.map((read) => read.value)).toEqual(["12", "17"]);
  });

  it("caps unmatched regions below automatic-trust confidence", () => {
    const secondary = output("Q2", 0.8);
    const result = reconcileReaders([{ id: "gemini", output: output("Q1") }, { id: "moondream", output: { ...secondary, regions: [{ ...secondary.regions[0], class: "header" }] } }], quality, pixels);
    expect(result.regions.every((region) => region.confidence <= 0.49)).toBe(true);
  });
});
