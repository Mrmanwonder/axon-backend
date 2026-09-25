import { describe, expect, it } from "vitest";
import { GOLDEN_CASES } from "../src/evaluation/golden";
import { runDeterministicGoldenCases } from "../src/evaluation/runner";

describe("executable deterministic golden suite", () => {
  it("executes every checked-in synthetic case instead of merely counting it", () => {
    const results = runDeterministicGoldenCases(GOLDEN_CASES);
    expect(results.map((result) => result.caseId)).toEqual(GOLDEN_CASES.map((item) => item.id));
    expect(results.filter((result) => !result.passed)).toEqual([]);
  });
});
