import { describe, expect, it } from "vitest";
import type { RgbaImage } from "@mastery/shared/crop.js";
import { reconcileTargetedRegion } from "../src/reconcile";
import type { ReaderRegion, TargetedRegionRead } from "../src/schema";

const pixels: RgbaImage = { data: new Uint8ClampedArray(100 * 100 * 4).fill(220), width: 100, height: 100 };
const region: ReaderRegion = {
  class: "printed_question",
  box: { x: 0.1, y: 0.1, width: 0.5, height: 0.1 },
  confidence: 0.96,
  text: "Solve x + 1 = 2",
  layer: "PRINTED"
};

function targeted(overrides: Partial<TargetedRegionRead> = {}): TargetedRegionRead {
  return {
    class: "printed_question",
    layer: "PRINTED",
    confidence: 0.95,
    value: "Solve x + 1 = 2",
    alternatives: [],
    status: "read",
    ...overrides
  };
}

function reconcile(read: TargetedRegionRead, layout: ReaderRegion = region) {
  return reconcileTargetedRegion({
    region: layout,
    targeted: read,
    layoutReaderId: "layout",
    targetedReaderId: "targeted",
    image: pixels,
    id: "r-0001"
  });
}

describe("targeted-region reconciliation", () => {
  it("retains agreed text and exposes two independent reads", () => {
    const result = reconcile(targeted());
    expect(result.region.text).toBe("Solve x + 1 = 2");
    expect(result.region.confidence).toBe(0.95);
    expect(result.reads.reads.map((read) => read.readerIds[0])).toEqual(["layout", "targeted"]);
  });

  it("does not collapse conflicting readings into a trusted value", () => {
    const result = reconcile(targeted({ value: "Solve x + 7 = 2", alternatives: ["Solve x + 1 = 2"] }));
    expect(result.region.text).toBeUndefined();
    expect(result.region.confidence).toBeLessThanOrEqual(0.79);
    expect(result.reads.reads.map((read) => read.value)).toEqual(["Solve x + 1 = 2", "Solve x + 7 = 2"]);
  });

  it("never trusts a value attached to an unreadable targeted result", () => {
    const result = reconcile(targeted({ status: "unreadable" }));
    expect(result.region.text).toBeUndefined();
    expect(result.region.confidence).toBeLessThanOrEqual(0.49);
    expect(result.reads.reads[1]?.value).toBeNull();
  });

  it("downgrades class and layer disagreement and withholds ink signals", () => {
    const result = reconcile(targeted({ class: "student_answer", layer: "STUDENT" }));
    expect(result.region.confidence).toBeLessThanOrEqual(0.49);
    expect(result.region.inkSignals).toBeUndefined();
  });
});
