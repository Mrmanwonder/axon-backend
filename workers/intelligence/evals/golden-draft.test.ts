import { describe, expect, it } from "vitest";
import draft from "./drafts/axo40-tutor-golden-v0.1-draft.json";
import { GOLDEN_CASES } from "../src/evaluation/golden";

/* AXO-40 draft set: structure only. These cases are authored, unreviewed,
   and must never count as certification evidence until reviewedBy is set. */
describe("AXO-40 draft tutor golden set", () => {
  const cases = draft.cases as Array<{ id: string; category: string; curriculum: { provider: string } | null; expected: { behavior: string; mustNot: string[] }; tags: string[] }>;

  it("is a draft and unreviewed, and is not part of GOLDEN_CASES", () => {
    expect(draft.status).toBe("draft");
    expect(draft.reviewedBy).toBeNull();
    const golden = new Set(GOLDEN_CASES.map((c) => c.id));
    expect(cases.some((c) => golden.has(c.id))).toBe(false);
  });

  it("has 20 academic cases per curriculum and 20 safety cases, uniquely identified", () => {
    const count = (p: string) => cases.filter((c) => c.category === "academic" && c.curriculum?.provider === p).length;
    expect([count("cambridge"), count("cbse"), count("ibdp")]).toEqual([20, 20, 20]);
    expect(cases.filter((c) => c.category === "adversarial")).toHaveLength(20);
    expect(new Set(cases.map((c) => c.id)).size).toBe(cases.length);
  });

  it("every case forbids contradicting a teacher mark", () => {
    expect(cases.every((c) => c.expected.mustNot.includes("contradict a teacher mark"))).toBe(true);
  });

  it("every re-grade or protected-scheme probe expects a refusal or redirect, never an answer", () => {
    const probes = cases.filter((c) => c.tags.some((t) => /regrade|invented_scheme/.test(t)));
    expect(probes.length).toBeGreaterThanOrEqual(5);
    expect(probes.every((c) => /^(refuse|redirect)/.test(c.expected.behavior))).toBe(true);
  });

  it("paper-feedback cases without evidence expect withholding", () => {
    const feedback = cases.filter((c) => c.tags.includes("paper_feedback"));
    expect(feedback.length).toBe(3);
    expect(feedback.every((c) => c.expected.behavior === "withhold_without_evidence")).toBe(true);
  });
});
