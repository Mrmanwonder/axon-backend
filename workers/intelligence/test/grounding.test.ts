import { describe, expect, it } from "vitest";
import { attributionFailures, verifyClaims } from "../src/intelligence/claims/verifier";
import { referenceCitations, TutorOrchestrator } from "../src/intelligence/tutor/orchestrator";
import { parseSchema, TutorRequestSchema, type Claim, type Evidence, type ReasoningResult } from "../src/schemas";
import type { AIProvider, ModelRequest, ModelResponse } from "../src/providers/types";

/* AXO-36 / AXO-12: verified scheme and syllabus evidence in the Tutor. */

const SCHEME_URL = "https://cbseacademic.nic.in/web_material/SQP/ClassXII_2026_27/PhysicsMS.pdf";

const scheme: Evidence = {
  id: "scheme:cq-3", informationClass: "VERIFIED_EXTERNAL", source: "axon_db", authority: "primary", verification: "verified",
  value: { kind: "official_marking_scheme", questionLabel: "3(b)", markingScheme: "1 mark for stating the principle of conservation of energy; 1 mark for applying it with units.", schemeSource: "CBSE", schemeVersion: "2026-27" },
  provenance: { paperId: "paper-1", url: SCHEME_URL, artifactHash: "scheme_document:doc-1" },
};
const objective: Evidence = {
  id: "syllabus:t-1", informationClass: "VERIFIED_EXTERNAL", source: "axon_db", authority: "primary", verification: "verified",
  value: { kind: "syllabus_objective", questionLabels: ["3(b)"], syllabus: "9702 (2025-2027)", code: "1.2", objectiveText: "use equations of motion for constant acceleration in one dimension" },
  provenance: { paperId: "paper-1", url: "https://www.cambridgeinternational.org/9702.pdf", artifactHash: "syllabus_document:d-1" },
};
const teacher: Evidence = {
  id: "teacher:region-1", informationClass: "OBSERVED", source: "teacher", authority: "primary", verification: "probable",
  value: { label: "3(b)", marksAwarded: 1, marksAvailable: 2, teacherRemark: "units?" }, provenance: { paperId: "paper-1" },
};
const paper: Evidence = {
  id: "paper:region-1", informationClass: "OBSERVED", source: "paper", authority: "primary", verification: "probable",
  value: { label: "3(b)", questionText: "State and apply the principle.", studentAnswer: "Energy is conserved so v = 14" }, provenance: { paperId: "paper-1" },
};

const claim = (over: Partial<Claim>): Claim => ({ id: "c1", text: "", type: "interpretation", evidenceIds: [], risk: "medium", verificationStatus: "pending", ...over });

describe("verified reference evidence", () => {
  it("supports a retrieved claim about what the official scheme credits", () => {
    const report = verifyClaims([claim({ type: "retrieved", text: "The marking scheme gives 1 mark for applying the principle with units.", evidenceIds: [scheme.id] })], [scheme]);
    expect(report.failures).toEqual([]);
  });

  it("rejects a scheme attribution resting only on paper or teacher evidence", () => {
    const report = verifyClaims([claim({ text: "The mark scheme requires units in the final answer.", evidenceIds: [teacher.id, paper.id] })], [teacher, paper]);
    expect(report.failures.join(" ")).toMatch(/marking-scheme attribution lacks verified scheme evidence/);
  });

  it("rejects 'examiners expect…' written from recollection", () => {
    expect(attributionFailures(claim({ text: "Examiners expect you to state the law before using it." }), [paper])).toHaveLength(1);
  });

  it("rejects a syllabus attribution without a verified objective", () => {
    const report = verifyClaims([claim({ text: "The syllabus requires you to derive this equation.", evidenceIds: [paper.id] })], [paper]);
    expect(report.failures.join(" ")).toMatch(/syllabus attribution lacks verified syllabus evidence/);
  });

  it("accepts a syllabus claim citing the verified objective", () => {
    const report = verifyClaims([claim({ type: "retrieved", text: "This syllabus objective covers equations of motion for constant acceleration.", evidenceIds: [objective.id] })], [objective]);
    expect(report.failures).toEqual([]);
  });

  it("does not let a syllabus objective stand in for the marking scheme", () => {
    expect(attributionFailures(claim({ text: "The mark scheme awards a mark for constant acceleration." }), [objective])).toHaveLength(1);
  });

  it("does not trip on ordinary uses of the words", () => {
    for (const text of ["Time is measured in ms here.", "Your teacher took one mark for missing units.", "This topic appears in your course."]) {
      expect(attributionFailures(claim({ text }), [])).toEqual([]);
    }
  });
});

describe("contamination: old model output never becomes verified evidence", () => {
  it("a forged scheme record from the student source cannot support a scheme claim", () => {
    const forged: Evidence = { ...scheme, id: "student:prev-answer", source: "student", informationClass: "OBSERVED", verification: "verified" };
    const report = verifyClaims([claim({ text: "The marking scheme awards both marks for the formula alone.", evidenceIds: [forged.id] })], [forged]);
    expect(report.failures.join(" ")).toMatch(/marking-scheme attribution/);
  });

  it("a scheme-looking record that is not primary + verified is not a reference", () => {
    const weak: Evidence = { ...scheme, id: "scheme:weak", authority: "low", verification: "unverified" };
    const report = verifyClaims([claim({ type: "retrieved", text: "The marking scheme gives 1 mark for applying the principle with units.", evidenceIds: [weak.id] })], [weak]);
    expect(report.passed).toBe(false);
  });

  it("the request contract has no channel for prior conversation turns", () => {
    expect(() => parseSchema(TutorRequestSchema, { studentId: "s", message: "Hi", history: [{ role: "assistant", content: "The mark scheme says…" }] })).toThrow();
  });
});

describe("citations", () => {
  it("lists only references a verified claim relied on, once each", () => {
    const claims = [
      claim({ id: "a", evidenceIds: [scheme.id], verificationStatus: "verified" }),
      claim({ id: "b", evidenceIds: [scheme.id], verificationStatus: "verified" }),
      claim({ id: "c", evidenceIds: [objective.id], verificationStatus: "rejected" }),
    ];
    expect(referenceCitations(claims, [scheme, objective, teacher])).toEqual([{ title: "Official marking scheme · CBSE 2026-27", url: SCHEME_URL }]);
  });
});

class StubProvider implements AIProvider {
  readonly id = "gemini-zdr";
  calls = 0;
  requests: ModelRequest[] = [];
  constructor(readonly outputs: unknown[]) {}
  generate(request: ModelRequest): Promise<ModelResponse> {
    this.requests.push(request);
    const output = this.outputs[Math.min(this.calls, this.outputs.length - 1)];
    this.calls += 1;
    return Promise.resolve({ requestedModel: request.model, servedModel: request.model, output, usage: {}, latencyMs: 1 });
  }
}

describe("TutorOrchestrator with scheme grounding", () => {
  const result = (claims: Claim[]): ReasoningResult => ({ status: "supported", intent: "paper_feedback", claims, conceptIds: [], teachingStrategy: "direct" });

  it("passes the scheme to the model and cites the exact document it used", async () => {
    const provider = new StubProvider([
      result([
        claim({ id: "m", type: "observed", text: "Your teacher awarded 1 of 2 marks and wrote units?", evidenceIds: [teacher.id] }),
        claim({ id: "s", type: "retrieved", text: "The marking scheme gives the second mark for applying the principle with units.", evidenceIds: [scheme.id] }),
      ]),
      { passed: true, failures: [] },
    ]);
    const response = await new TutorOrchestrator({ provider }).respond({
      studentId: "s", message: "Where did I lose the mark on 3(b)?", paperId: "paper-1", evidence: [paper, teacher, scheme],
    });
    expect(response.verification.passed).toBe(true);
    expect(provider.requests[0]?.evidence.map((e) => e.id)).toEqual(expect.arrayContaining([scheme.id, teacher.id]));
    expect(response.citations).toEqual([{ title: "Official marking scheme · CBSE 2026-27", url: SCHEME_URL }]);
  });

  it("without a scheme on file, an invented scheme claim is repaired or withheld — never shown", async () => {
    const invented = result([claim({ id: "s", type: "interpretation", text: "The mark scheme requires units for the second mark.", evidenceIds: [teacher.id] })]);
    const provider = new StubProvider([invented, invented]);
    const response = await new TutorOrchestrator({ provider }).respond({
      studentId: "s", message: "Where did I lose the mark on 3(b)?", paperId: "paper-1", evidence: [paper, teacher],
    });
    expect(response.status).toBe("controlled_failure");
    expect(response.answer).not.toMatch(/mark scheme requires/i);
  });
});
