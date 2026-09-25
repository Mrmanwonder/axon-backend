import { describe, expect, it } from "vitest";
import { GOLDEN_CASES, goldenCoverage } from "../src/evaluation/golden";
import { compareRuns, summarizeRun, ZERO_ESCAPE_METRICS } from "../src/evaluation/gates";

describe("evaluation platform release gates", () => {
  it("contains operational suites for every required category", () => {
    expect(goldenCoverage()).toMatchObject({ scanner: 3, tutor: 3, retrieval: 2, academic: 3, adversarial: 6 });
    expect(new Set(GOLDEN_CASES.map((item) => item.id)).size).toBe(GOLDEN_CASES.length);
  });
  it("blocks any high-risk hallucination escape", () => {
    const metrics = Object.fromEntries(ZERO_ESCAPE_METRICS.map((name) => [name, name === "invented_teacher_intent" ? 1 : 0]));
    const summary = summarizeRun([{ caseId: "x", passed: true, latencyMs: 1, metrics, failures: [] }]);
    expect(summary.releaseAllowed).toBe(false);
    expect(summary.gateFailures).toContain("invented_teacher_intent must equal 0");
  });
  it("blocks release until the minimum certified human-reviewed datasets exist", () => {
    const cleanMetrics = Object.fromEntries(ZERO_ESCAPE_METRICS.map((name) => [name, 0]));
    const results = Array.from({ length: 20 }, (_, index) => ({ caseId: `x${index}`, passed: true, latencyMs: 1, metrics: cleanMetrics, failures: [] }));
    const missing = summarizeRun(results);
    expect(missing.releaseAllowed).toBe(false);
    expect(missing.gateFailures).toContain("certified benchmark coverage is absent");
    const certified = summarizeRun(results, { scannerPapers: 100, scannerQuestions: 1_500, tutorCases: 500, handReviewedTutorCases: 500, privacyCertified: true, rollbackValidated: true });
    expect(certified.releaseAllowed).toBe(true);
  });
  it("requires candidate performance not to regress", () => {
    const cleanMetrics = Object.fromEntries(ZERO_ESCAPE_METRICS.map((name) => [name, 0]));
    const baseline = summarizeRun(Array.from({ length: 20 }, (_, index) => ({ caseId: `b${index}`, passed: true, latencyMs: 1, metrics: cleanMetrics, failures: [] })));
    const candidate = summarizeRun(Array.from({ length: 20 }, (_, index) => ({ caseId: `c${index}`, passed: index < 19, latencyMs: 1, metrics: cleanMetrics, failures: [] })));
    expect(compareRuns(baseline, candidate).allowed).toBe(false);
  });
});
