import { describe, expect, it } from "vitest";
import { detectContradictions, detectEvidenceConflicts, verifyClaims } from "../src/intelligence/claims/verifier";
import { detectPattern } from "../src/intelligence/context";
import { CircuitBreaker } from "../src/intelligence/routing/circuit-breaker";
import { ModelRouter, classifyRisk } from "../src/intelligence/routing/router";
import { NoCompliantProviderError } from "../src/intelligence/routing/privacy";
import { promptRegistry } from "../src/prompts";
import type { Claim, Evidence } from "../src/schemas";
import { minimizePublicRetrievalQuery } from "../src/intelligence/security/privacy";
import { sourceAuthority } from "../src/providers/tavily";

const teacherMark: Evidence = { id: "teacher_mark", informationClass: "OBSERVED", source: "teacher", authority: "primary", value: { marksAwarded: 3 }, provenance: {}, verification: "verified", confidence: 1 };

describe("claim verification", () => {
  it("blocks recorded-mark contradictions", () => {
    const claim: Claim = { id: "c1", text: "You received 2 marks.", type: "observed", evidenceIds: ["teacher_mark"], risk: "critical", verificationStatus: "pending" };
    expect(detectContradictions([claim], [teacherMark])).toContain("c1: contradicts recorded teacher mark");
  });
  it("blocks teacher intent without explicit evidence", () => {
    const claim: Claim = { id: "c1", text: "The teacher intended to penalize the method.", type: "interpretation", evidenceIds: ["teacher_mark"], risk: "critical", verificationStatus: "pending" };
    expect(detectContradictions([claim], [teacherMark])).toContain("c1: invents teacher intent");
  });
  it("requires tool evidence for calculations", () => {
    const claim: Claim = { id: "calc", text: "The answer is 42.", type: "calculation", evidenceIds: ["teacher_mark"], risk: "high", verificationStatus: "pending" };
    expect(verifyClaims([claim], [teacherMark]).passed).toBe(false);
  });
  it("exposes conflicts between retrieved sources", () => {
    const sources: Evidence[] = [
      { id: "a", informationClass: "VERIFIED_EXTERNAL", source: "official_source", authority: "primary", value: { factKey: "deadline", assertion: "1 June" }, provenance: { url: "https://a.edu", retrievedAt: new Date().toISOString() }, verification: "verified" },
      { id: "b", informationClass: "VERIFIED_EXTERNAL", source: "retrieval", authority: "secondary", value: { factKey: "deadline", assertion: "2 June" }, provenance: { url: "https://b.edu", retrievedAt: new Date().toISOString() }, verification: "verified" }
    ];
    expect(detectEvidenceConflicts(sources)).toHaveLength(1);
  });
  it("rejects an unrelated claim even when it cites a real retrieved source", () => {
    const source: Evidence = { id: "source", informationClass: "VERIFIED_EXTERNAL", source: "official_source", authority: "primary", value: { title: "Exam dates", content: "The examination begins in June." }, provenance: { url: "https://example.gov/exams" }, verification: "verified" };
    const claim: Claim = { id: "unrelated", text: "The chemistry syllabus removed electrolysis.", type: "retrieved", evidenceIds: ["source"], risk: "critical", verificationStatus: "pending" };
    expect(verifyClaims([claim], [source]).failures).toContain("unrelated: retrieval evidence does not support the claim text");
  });
});

describe("routing and privacy", () => {
  it("fails closed if no compliant provider exists", () => {
    expect(() => new ModelRouter().route({ capability: "tutoring", risk: "R4", privacyPolicy: "STUDENT_CHAT_STRICT", latencyBudgetMs: 10_000, difficulty: "complex", multimodal: false }, new Set(["non-zdr"]))).toThrow(NoCompliantProviderError);
  });
  it("routes critical work to high thinking", () => {
    const route = new ModelRouter().route({ capability: "tutoring", risk: "R4", privacyPolicy: "STUDENT_CHAT_STRICT", latencyBudgetMs: 12_000, difficulty: "complex", multimodal: false }, new Set(["gemini-zdr"]));
    expect(route.thinkingLevel).toBe("high");
    expect(classifyRisk("paper_feedback")).toBe("R4");
  });
  it("opens and half-opens a degraded provider circuit", () => {
    const breaker = new CircuitBreaker(2, 100);
    breaker.record({ success: false, latencyMs: 10 }, 0);
    breaker.record({ success: false, latencyMs: 10 }, 1);
    expect(breaker.state(50)).toBe("OPEN");
    expect(breaker.state(101)).toBe("HALF_OPEN");
  });
});

describe("provenance and patterns", () => {
  it("produces immutable repeatable prompt hashes", async () => {
    const one = await promptRegistry.compile("tutor.paper_explanation.v3");
    const two = await promptRegistry.compile("tutor.paper_explanation.v3");
    expect(one.promptHash).toBe(two.promptHash);
    expect(one.schemaHash).toBe(two.schemaHash);
  });
  it("does not create a trait from one event", () => {
    const one = detectPattern([{ conceptId: "algebra", paperId: "p1", correct: false, confidence: 1, trustState: "AUTO_VERIFIED" }]);
    expect(one.established).toBe(false);
    const repeated = detectPattern([1, 2, 3, 4].map((index) => ({ conceptId: "algebra", paperId: `p${Math.ceil(index / 2)}`, correct: false, confidence: 0.9, trustState: "AUTO_VERIFIED" as const })));
    expect(repeated.established).toBe(true);
  });
});

describe("public retrieval privacy", () => {
  it("rejects personal identifiers before the public search adapter", () => {
    expect(() => minimizePublicRetrievalQuery("Current syllabus for my student id 12345")).toThrow("RETRIEVAL_BLOCKED_SENSITIVE_QUERY");
    expect(() => minimizePublicRetrievalQuery("Email learner@example.com with the current syllabus")).toThrow("RETRIEVAL_BLOCKED_SENSITIVE_QUERY");
  });
  it("keeps a bounded public academic query", () => {
    expect(minimizePublicRetrievalQuery("  current   GCSE chemistry syllabus  ")).toBe("current GCSE chemistry syllabus");
  });
  it("does not promote lookalike or generic educational domains to primary authority", () => {
    expect(sourceAuthority("https://qualifications.pearson.com/spec", "official_rule")).toBe("primary");
    expect(sourceAuthority("https://pearson.com.example.net/spec", "official_rule")).toBe("low");
    expect(sourceAuthority("https://example.edu/article", "current_fact")).toBe("secondary");
  });
});
