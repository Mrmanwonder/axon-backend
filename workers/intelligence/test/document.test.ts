import { describe, expect, it } from "vitest";
import { assessPageQuality, conditioningPlan } from "../src/document/quality";
import { classifyInk } from "../src/document/ink";
import { buildQuestionGraph } from "../src/document/question-graph";
import { matchMarks } from "../src/document/mark-matcher";
import { detectDocumentType } from "../src/document/ingest";
import { readBoundedBytes } from "../src/shared/bounded-bytes";
import type { LayoutRegion } from "../src/document/layout/types";

const box = (x: number, y: number, width = 0.1, height = 0.05) => ({ x, y, width, height });

describe("AxonQuality", () => {
  it("classifies unreadable pages without repeated model attempts", () => {
    const result = assessPageQuality({ blur: 0.95, glareFraction: 0.4, perspectiveDegrees: 18, resolution: 0.1, compression: 0.8, cropCompleteness: 0.5, shadowFraction: 0.5 });
    expect(result.classification).toBe("UNREADABLE");
    expect(result.requiredAction).toBe("rescan");
  });
  it("penalizes perspective distortion in either direction", () => {
    const base = { blur: 0, glareFraction: 0, resolution: 1, compression: 0, cropCompleteness: 1, shadowFraction: 0 };
    expect(assessPageQuality({ ...base, perspectiveDegrees: -15 }).score).toBe(assessPageQuality({ ...base, perspectiveDegrees: 15 }).score);
  });
  it("never enables generative enhancement", () => {
    expect(conditioningPlan({ blur: 0.2, glareFraction: 0.1, perspectiveDegrees: 2, resolution: 0.7, compression: 0.2, cropCompleteness: 1, shadowFraction: 0.1 }, 91).generativeEnhancement).toBe(false);
  });
});

describe("paper ingestion", () => {
  it("sniffs supported immutable document bytes instead of trusting the header", () => {
    expect(detectDocumentType(new Uint8Array([0x25, 0x50, 0x44, 0x46, 0x2d, 0x31]).buffer)).toBe("application/pdf");
    expect(detectDocumentType(new TextEncoder().encode("not an image").buffer)).toBeNull();
  });
  it("bounds streamed upload bytes even when headers understate the body", async () => {
    const stream = new ReadableStream<Uint8Array>({ start(controller) { controller.enqueue(new Uint8Array(6)); controller.close(); } });
    await expect(readBoundedBytes(stream, 5)).rejects.toThrow("exceeds 5 bytes");
  });
});

describe("AxonInk", () => {
  it("uses multiple signals rather than colour alone", () => {
    const result = classifyInk({ printedProbability: 0.05, colourDistanceFromPrint: 0.1, strokeDifference: 0.8, marginTendency: 0.95, annotationOverlap: 0.9, handwritingDifference: 0.8 });
    expect(result.class).toBe("TEACHER");
  });
});

describe("QuestionGraph", () => {
  it("preserves hierarchy and cross-page continuation", () => {
    const regions: LayoutRegion[] = [
      { id: "q1", pageId: "p1", class: "question_number", box: box(0.1, 0.1), confidence: 1, text: "1" },
      { id: "a1", pageId: "p1", class: "student_answer", box: box(0.2, 0.2), confidence: 1 },
      { id: "cont", pageId: "p2", class: "continuation_region", box: box(0.2, 0.1), confidence: 1 }
    ];
    const graph = buildQuestionGraph(regions);
    expect(graph.questions[0]?.pageIds).toEqual(["p1", "p2"]);
    expect(graph.questions[0]?.continuationIds).toEqual(["cont"]);
  });
  it("uses explicit page order rather than opaque page ids", () => {
    const regions: LayoutRegion[] = [
      { id: "q", pageId: "z-page", class: "question_number", box: box(0.1, 0.1), confidence: 1, text: "7" },
      { id: "cont", pageId: "a-page", class: "continuation_region", box: box(0.2, 0.1), confidence: 1 }
    ];
    const graph = buildQuestionGraph(regions, new Map([["z-page", 0], ["a-page", 1]]));
    expect(graph.questions[0]?.pageIds).toEqual(["z-page", "a-page"]);
    expect(graph.unassignedRegionIds).toEqual([]);
  });
});

describe("AxonMarkMatcher", () => {
  it("assigns two close margin marks globally", () => {
    const result = matchMarks(
      [{ id: "m1", pageId: "p1", box: box(0.9, 0.22) }, { id: "m2", pageId: "p1", box: box(0.9, 0.7) }],
      [{ id: "q1", pageIds: ["p1"], box: box(0.2, 0.2, 0.5, 0.2), order: 1 }, { id: "q2", pageIds: ["p1"], box: box(0.2, 0.68, 0.5, 0.2), order: 2 }]
    );
    expect(result.map((item) => item.questionId)).toEqual(["q1", "q2"]);
  });
  it("leaves a distant mark unattributed", () => {
    const result = matchMarks([{ id: "m", pageId: "p2", box: box(0.95, 0.95) }], [{ id: "q", pageIds: ["p1"], box: box(0.1, 0.1), order: 1 }]);
    expect(result[0]?.questionId).toBeNull();
  });
});
