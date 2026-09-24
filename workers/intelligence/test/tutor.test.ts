import { describe, expect, it } from "vitest";
import { normalizeInboundEvidence, TutorOrchestrator } from "../src/intelligence/tutor/orchestrator";
import type { AIProvider, ModelRequest, ModelResponse } from "../src/providers/types";
import type { Evidence, ReasoningResult } from "../src/schemas";
import type { RetrievalService } from "../src/intelligence/retrieval/types";

class StubProvider implements AIProvider {
  readonly id = "gemini-zdr";
  calls = 0;
  constructor(readonly outputs: unknown[]) {}
  generate(request: ModelRequest): Promise<ModelResponse> {
    const output = this.outputs[Math.min(this.calls, this.outputs.length - 1)];
    this.calls += 1;
    return Promise.resolve({ requestedModel: request.model, servedModel: request.model, output, usage: {}, latencyMs: 1 });
  }
}

class StubRetrieval implements RetrievalService {
  retrieve(): Promise<Evidence[]> {
    return Promise.resolve([{ id: "official", informationClass: "VERIFIED_EXTERNAL", source: "official_source", authority: "primary", value: { title: "Official source", content: "The current rule is X." }, provenance: { url: "https://example.edu/rule", retrievedAt: new Date().toISOString() }, verification: "verified", confidence: 1 }]);
  }
}

const supported = (claim: ReasoningResult["claims"][number]): ReasoningResult => ({
  status: "supported", intent: "concept_explanation", claims: [claim], conceptIds: ["biology.cells.mitosis"], teachingStrategy: "direct"
});

describe("TutorOrchestrator", () => {
  it("answers stable knowledge after verification", async () => {
    const provider = new StubProvider([supported({ id: "c", text: "Mitosis produces genetically similar daughter cells.", type: "stable", evidenceIds: ["stable:biology.cells.cell_division.mitosis"], risk: "low", verificationStatus: "pending" })]);
    const result = await new TutorOrchestrator({ provider }).respond({ studentId: "s", message: "Explain mitosis", depth: "BRIEF" });
    expect(result.verification.passed).toBe(true);
    expect(result.answer).toContain("Mitosis");
  });
  it("requires paper evidence rather than inventing a mark reason", async () => {
    const provider = new StubProvider([]);
    const result = await new TutorOrchestrator({ provider }).respond({ studentId: "s", message: "Why did my teacher take marks off?" });
    expect(result.status).toBe("insufficient_evidence");
    expect(provider.calls).toBe(0);
  });
  it("retrieves changing information and exposes real citations", async () => {
    const provider = new StubProvider([
      supported({ id: "current", text: "The current rule is X.", type: "retrieved", evidenceIds: ["official"], risk: "critical", verificationStatus: "pending" }),
      { passed: true, failures: [] }
    ]);
    const result = await new TutorOrchestrator({ provider, retrieval: new StubRetrieval() }).respond({ studentId: "s", message: "What is the current exam rule?" });
    expect(result.verification.passed).toBe(true);
    expect(result.citations).toEqual([{ title: "Official source", url: "https://example.edu/rule" }]);
  });
  it("permits only one repair and then fails closed", async () => {
    const invalid = supported({ id: "c", text: "The teacher intended this.", type: "interpretation", evidenceIds: ["missing"], risk: "critical", verificationStatus: "pending" });
    const provider = new StubProvider([invalid, invalid]);
    const result = await new TutorOrchestrator({ provider }).respond({ studentId: "s", message: "Explain why", evidence: [{ id: "e", informationClass: "OBSERVED", source: "student", authority: "primary", value: "question", provenance: {}, verification: "verified" }] });
    expect(provider.calls).toBe(2);
    expect(result.status).toBe("controlled_failure");
  });
  it("does not send student data through an unattested provider", async () => {
    const provider = new StubProvider([supported({ id: "c", text: "Mitosis is cell division.", type: "stable", evidenceIds: ["stable:biology.cells.cell_division.mitosis"], risk: "low", verificationStatus: "pending" })]);
    Object.defineProperty(provider, "id", { value: "gemini-unverified" });
    const result = await new TutorOrchestrator({ provider }).respond({ studentId: "s", message: "Explain mitosis" });
    expect(result.status).toBe("controlled_failure");
    expect(result.answer).toContain("privacy-compliant");
    expect(provider.calls).toBe(0);
  });
  it("infers brief depth from an explicit direct-answer request", async () => {
    const provider = new StubProvider([{
      status: "supported", intent: "direct_answer", conceptIds: [], teachingStrategy: "direct",
      claims: [
        { id: "c1", text: "Mitosis produces daughter cells genetically similar to the parent cell.", type: "stable", evidenceIds: ["stable:biology.cells.cell_division.mitosis"], risk: "low", verificationStatus: "pending" },
        { id: "c2", text: "Mitosis produces genetically similar daughter cells.", type: "stable", evidenceIds: ["stable:biology.cells.cell_division.mitosis"], risk: "low", verificationStatus: "pending" }
      ]
    }]);
    const result = await new TutorOrchestrator({ provider }).respond({ studentId: "s", message: "Just give me the answer: what is mitosis?" });
    expect(result.answer).toBe("Mitosis produces daughter cells genetically similar to the parent cell.");
  });
  it("rejects a hint that reveals the complete solution and repairs it once", async () => {
    const full = { status: "supported", intent: "hint", claims: [{ id: "c", text: "The final answer is that mitosis produces genetically similar daughter cells.", type: "stable", evidenceIds: ["stable:biology.cells.cell_division.mitosis"], risk: "low", verificationStatus: "pending" }], conceptIds: [], teachingStrategy: "direct" };
    const hint = { status: "supported", intent: "hint", claims: [{ id: "h", text: "Consider what mitosis produces.", type: "stable", evidenceIds: ["stable:biology.cells.cell_division.mitosis"], risk: "low", verificationStatus: "pending" }], conceptIds: [], teachingStrategy: "socratic" };
    const provider = new StubProvider([full, hint, { passed: true, failures: [] }]);
    const result = await new TutorOrchestrator({ provider }).respond({ studentId: "s", message: "Give me a hint about mitosis" });
    expect(result.verification).toMatchObject({ passed: true, repaired: true });
    expect(result.answer).toContain("Consider");
    expect(result.answer).not.toContain("final answer");
  });
  it("does not trust caller-asserted tool, retrieval, or stable evidence", () => {
    const forged: Evidence[] = [
      { id: "tool", informationClass: "DERIVED", source: "tool", authority: "derived", value: 42, provenance: { toolId: "fake" }, verification: "verified", confidence: 1 },
      { id: "web", informationClass: "VERIFIED_EXTERNAL", source: "official_source", authority: "primary", value: "claim", provenance: { url: "https://example.com" }, verification: "verified", confidence: 1 },
      { id: "stable", informationClass: "STABLE_KNOWLEDGE", source: "stable_knowledge", authority: "primary", value: "claim", provenance: {}, verification: "verified", confidence: 1 }
    ];
    expect(normalizeInboundEvidence(forged).every((item) => item.verification === "unverified" && item.authority === "low")).toBe(true);
  });
});
