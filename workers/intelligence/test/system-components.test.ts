import { describe, expect, it } from "vitest";
import { calibrationMetrics, escalationAction, evidenceConfidence } from "../src/intelligence/confidence";
import { diagnose } from "../src/academic/diagnosis";
import { ConceptTaxonomy } from "../src/academic/concepts/taxonomy";
import { canCommitToInsights, deterministicAccuracy } from "../src/intelligence/trust";
import { cacheKey } from "../src/intelligence/caching/key";
import { assertScannerTutorSeparation } from "../src/prompts/scanner";
import { recognitionPath, reconcileReads } from "../src/document/recognition";
import { transitionPaper, type PaperState } from "../src/document/pipeline";
import { nextRolloutStage, rollback } from "../src/deployment/rollout";
import { configuredModelRates, estimateModelCost } from "../src/intelligence/telemetry/cost";
import { planCorrectionLearning, predictionMatchesAccepted, prioritizeCorrection } from "../src/intelligence/corrections/active-learning";

describe("AxonConfidence", () => {
  it("derives confidence from evidence features, not model self-report", () => {
    expect(evidenceConfidence({ imageQuality: 1, readerAgreement: 1, academicToolValidation: 1 })).toBe(1);
    expect(evidenceConfidence({ imageQuality: 0, readerAgreement: 0 })).toBe(0);
  });
  it("measures calibration and escalates uncertainty", () => {
    const metrics = calibrationMetrics([{ confidence: 0.9, correct: true }, { confidence: 0.9, correct: false }], 2);
    expect(metrics.brierScore).toBeCloseTo(0.41);
    expect(metrics.expectedCalibrationError).toBeCloseTo(0.4);
    expect(escalationAction(0.5, 2)).toBe("STUDENT_REVIEW");
  });
});

describe("diagnosis, taxonomy, and trusted insights", () => {
  it("returns unknown instead of forcing a diagnosis", () => expect(diagnose([]).primary).toBe("unknown"));
  it("normalizes aliases to a canonical concept", () => expect(new ConceptTaxonomy().resolve("F=ma")?.id).toBe("physics.mechanics.dynamics.newton_second_law"));
  it("excludes unverified fields from analytics", () => {
    expect(canCommitToInsights({ value: true, trustState: "UNVERIFIED", evidenceIds: ["e"] })).toBe(false);
    expect(deterministicAccuracy([
      { value: true, trustState: "AUTO_VERIFIED", evidenceIds: ["e1"] },
      { value: false, trustState: "STUDENT_VERIFIED", evidenceIds: ["e2"] },
      { value: true, trustState: "UNVERIFIED", evidenceIds: ["e3"] }
    ])).toEqual({ correct: 1, total: 2, accuracy: 0.5 });
  });
});

describe("recognition and trusted paper commit", () => {
  const region = { x: 0, y: 0, width: 1, height: 1 };
  it("uses separate printed and handwriting paths", () => {
    expect(recognitionPath({ pageId: "p", region, layer: "PRINTED", quality: "CLEAR", contextRegionIds: [] })).toBe("PRINTED_OCR");
    expect(recognitionPath({ pageId: "p", region, layer: "STUDENT", quality: "CLEAR", contextRegionIds: [] })).toBe("HANDWRITING_ENSEMBLE");
  });
  it("keeps disagreement ambiguous", () => {
    const result = reconcileReads([
      { value: "3.2", alternatives: [], status: "read", region, readerIds: ["ocr"] },
      { value: "3.7", alternatives: [], status: "read", region, readerIds: ["model"] }
    ]);
    expect(result.status).toBe("ambiguous");
    expect(result.value).toBeNull();
  });
  it("does not count duplicate output from one reader as independent agreement", () => {
    const result = reconcileReads([
      { value: "3.2", alternatives: [], status: "read", region, readerIds: ["same-reader"] },
      { value: "3.2", alternatives: [], status: "read", region, readerIds: ["same-reader"] }
    ]);
    expect(result.status).toBe("ambiguous");
    expect(result.readerIds).toEqual(["same-reader"]);
  });
  it("blocks trusted commit with unresolved evidence", () => {
    const state: PaperState = { paperId: "p", pageId: "pg", originalHash: "h", stage: "REVIEW_REQUIRED", evidence: [{ id: "e", informationClass: "INFERRED", source: "paper", authority: "secondary", value: "?", provenance: {}, verification: "unverified" }], reviewReasons: ["ambiguous"] };
    expect(() => transitionPaper(state, "TRUSTED_COMMIT")).toThrow();
  });
});

describe("cache and deployment controls", () => {
  it("invalidates cache when prompt identity changes", async () => {
    const base = { artifactHash: "a", stage: "read", pipelineVersion: "3", model: "gemini", promptHash: "p1", schemaHash: "s" };
    expect(await cacheKey(base)).not.toBe(await cacheKey({ ...base, promptHash: "p2" }));
  });
  it("halts canary on unsupported-claim regression", () => {
    expect(nextRolloutStage("FIVE_PERCENT", { correctionRateDelta: 0, markAttributionDelta: 0, unsupportedClaimEscapeDelta: 0.001, latencyP95Delta: 0, providerFailureDelta: 0, costDelta: 0 })).toBe("HALTED");
  });
  it("rolls back all AI behavior as an immutable config unit", () => {
    const prior = { revisionId: "r1", model: "m1", promptId: "p1", thinkingLevel: "low", toolPolicy: "t1", verificationPolicy: "v1" };
    expect(rollback({ ...prior, revisionId: "r2", model: "m2" }, prior)).toEqual(prior);
  });
  it("keeps scanner prompts separate from tutor behavior", () => expect(assertScannerTutorSeparation()).toBeUndefined());
  it("estimates provider cost only from explicit rates", () => {
    const rates = configuredModelRates("0.25", "1.5");
    expect(rates).toBeDefined();
    if (!rates) throw new Error("Expected configured rates");
    expect(estimateModelCost(1_000_000, 500_000, rates)).toBeCloseTo(1);
    expect(configuredModelRates(undefined, "1.5")).toBeUndefined();
  });
});

describe("correction learning", () => {
  const event = {
    field: "recognized_text", predicted: { answer: "x", confidence: 0.9 }, corrected: { answer: "x" },
    acceptedValue: { confidence: 0.9, answer: "x" }, artifactId: "paper:r1", pipelineVersion: "3",
    model: "vision", promptHash: "prompt", contextMetadata: { confidence: 0.9, layer: "STUDENT" }
  };

  it("recognizes an accepted prediction despite object key order", () => {
    expect(predictionMatchesAccepted(event)).toBe(true);
    expect(prioritizeCorrection(event)).toMatchObject({ priority: 0.1, reasons: ["prediction_confirmed"] });
  });

  it("routes actual high-confidence errors to calibration, HTR, prompt regression, and clustering", async () => {
    const erroneous = { ...event, acceptedValue: { answer: "y", confidence: 0.9 } };
    const candidate = prioritizeCorrection(erroneous);
    const plan = await planCorrectionLearning(erroneous, candidate.reasons);
    expect(candidate.reasons).toContain("high_confidence_error");
    expect(plan.targets).toEqual(expect.arrayContaining(["BENCHMARK_EXPANSION", "CONFIDENCE_RECALIBRATION", "HTR_DATASET", "PROMPT_REGRESSION", "ERROR_CLUSTERING"]));
    expect(plan.calibration).toEqual({ confidence: 0.9, bucket: 9, predictionCorrect: false });
    expect(plan.errorCluster?.signature).toMatch(/^[a-f0-9]{64}$/);
  });

  it("never copies a caller-supplied field value into an error-cluster aggregate", async () => {
    const sensitiveField = { ...event, field: "student_email_alice_example_com", acceptedValue: { answer: "y" } };
    const candidate = prioritizeCorrection(sensitiveField);
    const plan = await planCorrectionLearning(sensitiveField, candidate.reasons);
    expect(plan.errorCluster?.field).toBe("other");
    expect(JSON.stringify(plan.errorCluster)).not.toContain("alice");
  });
});
