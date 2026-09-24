import { describe, expect, it } from "vitest";
import {
  RELEASE_ARTIFACT_HASH_FIELDS,
  REQUIRED_SCANNER_CONDITIONS,
  REQUIRED_TUTOR_CATEGORIES,
  ROLLOUT_ORDER,
  sha256Hex,
  validateReleaseEvidence
} from "../scripts/release-evidence.mjs";

const reviewed = { humanReviewed: true, reviewer: "reviewer@example.test", reviewedAt: "2026-09-24T00:00:00.000Z" };

function scannerRecords() {
  return Array.from({ length: 1_500 }, (_, index) => ({
    ...reviewed,
    paperId: `paper-${Math.floor(index / 15)}`,
    questionId: `q-${index}`,
    classLevel: 9 + (index % 4),
    subject: `subject-${index % 4}`,
    teacherId: `teacher-${index % 2}`,
    writingStyleId: `style-${index % 3}`,
    deviceQuality: `quality-${index % 3}`,
    conditions: [REQUIRED_SCANNER_CONDITIONS[index % REQUIRED_SCANNER_CONDITIONS.length]],
    questionDiscovered: true,
    questionNumberCorrect: true,
    markAttributionApplicable: true,
    markAttributedCorrectly: true,
    numericMarkApplicable: true,
    numericMarkCorrect: true,
    reportedTotalApplicable: index % 15 === 0,
    reportedTotalCorrect: index % 15 === 0 ? true : undefined,
    catastrophicWrongBinding: false
  }));
}

function tutorRecords() {
  const rubric = {
    answersActualQuestion: true,
    factuallyCorrect: true,
    appropriateLevel: true,
    relevantConceptIdentified: true,
    reasoningUnderstandable: true,
    unnecessaryInformationAvoided: true,
    misconceptionIdentifiedWhenSupported: true,
    actionableAdviceWhenAppropriate: true
  };
  return Array.from({ length: 500 }, (_, index) => ({
    ...reviewed,
    caseId: `case-${index}`,
    category: REQUIRED_TUTOR_CATEGORIES[index % REQUIRED_TUTOR_CATEGORIES.length],
    rubric,
    factualClaims: 2,
    unsupportedFactualClaims: 0,
    violations: [],
    abstentionExpected: index < 50,
    sufficientEvidence: index >= 50,
    abstained: index < 50
  }));
}

function retrievalRecords() {
  return [
    { ...reviewed, caseId: "current", retrievalRequired: true, retrievalUsed: true, authoritativeSource: true, freshSource: true, citationCorrect: true, contradictionApplicable: false, latencyMs: 25, queryQualityScore: 1, fabricatedCitation: false },
    { ...reviewed, caseId: "conflict", retrievalRequired: true, retrievalUsed: true, authoritativeSource: true, freshSource: true, citationCorrect: true, contradictionApplicable: true, contradictionHandled: true, latencyMs: 30, queryQualityScore: 1, fabricatedCitation: false },
    { ...reviewed, caseId: "stable", retrievalRequired: false, retrievalUsed: false, contradictionApplicable: false, latencyMs: 0, queryQualityScore: 1, fabricatedCitation: false }
  ];
}

function jsonl(records) {
  return Buffer.from(records.map((record) => JSON.stringify(record)).join("\n"));
}

function fixture() {
  const scannerCases = jsonl(scannerRecords());
  const tutorCases = jsonl(tutorRecords());
  const retrievalCases = jsonl(retrievalRecords());
  const geminiZdrEvidence = Buffer.from("signed Gemini ZDR evidence");
  const visionZdrEvidence = Buffer.from("signed vision ZDR evidence");
  const capabilityProbe = Buffer.from(JSON.stringify({
    formatVersion: "axon-capability-probe.v1",
    model: "gemini-3.5-flash-lite",
    geminiPassed: true,
    tavilyPassed: true,
    visionPassed: true,
    deploymentSha: "0123456789abcdef",
    configRevision: "v3.test",
    observedAt: "2026-09-24T00:00:00.000Z"
  }));
  const comparisonResult = {
    cases: 500,
    datasetSha256: "set-after-tutor-hash",
    correctness: 0.96,
    unsupportedClaimRate: 0,
    falseUncertaintyRate: 0.01,
    pedagogyScore: 0.96,
    toolCallRate: 0.25,
    meanInputTokens: 500,
    meanOutputTokens: 200,
    meanCostUsd: 0.001,
    latencyMs: { simpleP50: 1_000, simpleP95: 2_000, standardP50: 2_000, standardP95: 5_000, toolP50: 4_000, toolP95: 9_000 }
  };
  const rollbackEvidence = Buffer.from(JSON.stringify({
    formatVersion: "axon-rollback-evidence.v1",
    validated: true,
    fromRevision: "v3.candidate",
    toRevision: "v3.prior",
    healthPassed: true,
    tutorProbePassed: true,
    documentProbePassed: true,
    traceProvenancePassed: true,
    reviewer: reviewed.reviewer,
    validatedAt: "2026-09-24T00:00:00.000Z"
  }));
  const canaryEvidence = Buffer.from(JSON.stringify({
    formatVersion: "axon-canary-evidence.v1",
    reviewer: reviewed.reviewer,
    reviewedAt: "2026-09-24T00:00:00.000Z",
    stages: ROLLOUT_ORDER.slice(0, -1).map((stage) => ({
      stage,
      passed: true,
      correctionRateDelta: 0,
      markAttributionDelta: 0,
      unsupportedClaimEscapeDelta: 0,
      latencyP95Delta: 0,
      providerFailureDelta: 0,
      costDelta: 0,
      completedAt: "2026-09-24T00:00:00.000Z"
    }))
  }));
  const artifacts = { scannerCases, tutorCases, retrievalCases, geminiZdrEvidence, visionZdrEvidence, capabilityProbe, rollbackEvidence, canaryEvidence };
  const certification = {
    formatVersion: "axon-release.v2",
    scannerPapers: 100,
    scannerQuestions: 1_500,
    tutorCases: 500,
    handReviewedTutorCases: 500,
    privacyCertified: true,
    rollbackValidated: true,
    evidenceUri: "r2://axon-release-evidence/test",
    reviewer: reviewed.reviewer,
    certifiedAt: "2026-09-24T00:00:00.000Z",
    deploymentSha: "0123456789abcdef",
    configRevision: "v3.test",
    artifacts: Object.fromEntries(Object.keys(RELEASE_ARTIFACT_HASH_FIELDS).map((name) => [name, `${name}.evidence`]))
  };
  for (const [artifactName, hashField] of Object.entries(RELEASE_ARTIFACT_HASH_FIELDS)) {
    if (["reviewManifest", "modelComparison"].includes(artifactName)) continue;
    certification[hashField] = sha256Hex(artifacts[artifactName]);
  }
  const modelComparison = Buffer.from(JSON.stringify({
    formatVersion: "axon-model-comparison.v1",
    reviewer: reviewed.reviewer,
    reviewedAt: "2026-09-24T00:00:00.000Z",
    baseline: { ...comparisonResult, model: "gemini-3.1-flash-lite", promptId: "paper_feedback.v1", datasetSha256: certification.tutorDatasetSha256 },
    candidate: { ...comparisonResult, model: "gemini-3.5-flash-lite", promptId: "paper_feedback.v2", datasetSha256: certification.tutorDatasetSha256, deploymentSha: certification.deploymentSha, configRevision: certification.configRevision }
  }));
  artifacts.modelComparison = modelComparison;
  certification.modelComparisonEvidenceSha256 = sha256Hex(modelComparison);
  const reviewManifest = Buffer.from(JSON.stringify({
    formatVersion: "axon-review-manifest.v1",
    approved: true,
    approvedAt: "2026-09-24T00:00:00.000Z",
    reviewers: [reviewed.reviewer],
    ...Object.fromEntries(Object.values(RELEASE_ARTIFACT_HASH_FIELDS).filter((name) => name !== "reviewManifestSha256").map((name) => [name, certification[name]]))
  }));
  artifacts.reviewManifest = reviewManifest;
  certification.reviewManifestSha256 = sha256Hex(reviewManifest);
  return { certification, artifacts };
}

function rehashArtifact(subject, artifactName) {
  const hashField = RELEASE_ARTIFACT_HASH_FIELDS[artifactName];
  subject.certification[hashField] = sha256Hex(subject.artifacts[artifactName]);
  const manifest = JSON.parse(subject.artifacts.reviewManifest.toString("utf8"));
  manifest[hashField] = subject.certification[hashField];
  subject.artifacts.reviewManifest = Buffer.from(JSON.stringify(manifest));
  subject.certification.reviewManifestSha256 = sha256Hex(subject.artifacts.reviewManifest);
}

describe("release evidence certification", () => {
  it("derives all minimum coverage and quality gates from reviewed artifacts", () => {
    const { certification, artifacts } = fixture();
    const result = validateReleaseEvidence(certification, artifacts);
    expect(result.errors).toEqual([]);
    expect(result.valid).toBe(true);
    expect(result.metrics.scanner).toMatchObject({ scannerPapers: 100, scannerQuestions: 1_500, questionDiscoveryRecall: 1 });
    expect(result.metrics.tutor).toMatchObject({ tutorCases: 500, handReviewedTutorCases: 500, unsupportedClaimEscapeRate: 0 });
    expect(result.metrics.retrieval).toMatchObject({ retrievalRequiredRecall: 1, citationCorrectness: 1 });
  });

  it("rejects typed-in counts when the evidence has fewer records", () => {
    const { certification, artifacts } = fixture();
    artifacts.scannerCases = jsonl(scannerRecords().slice(0, 1_499));
    certification.scannerDatasetSha256 = sha256Hex(artifacts.scannerCases);
    const manifest = JSON.parse(artifacts.reviewManifest.toString("utf8"));
    manifest.scannerDatasetSha256 = certification.scannerDatasetSha256;
    artifacts.reviewManifest = Buffer.from(JSON.stringify(manifest));
    certification.reviewManifestSha256 = sha256Hex(artifacts.reviewManifest);
    const result = validateReleaseEvidence(certification, artifacts);
    expect(result.valid).toBe(false);
    expect(result.errors).toContain("scanner evidence has 1499 questions; 1500 required");
    expect(result.errors).toContain("certified scannerQuestions does not equal derived evidence count");
  });

  it("rejects any zero-escape violation even when aggregate tutoring scores pass", () => {
    const { certification, artifacts } = fixture();
    const cases = tutorRecords();
    cases[0].violations = ["invented_teacher_intent"];
    artifacts.tutorCases = jsonl(cases);
    certification.tutorDatasetSha256 = sha256Hex(artifacts.tutorCases);
    const manifest = JSON.parse(artifacts.reviewManifest.toString("utf8"));
    manifest.tutorDatasetSha256 = certification.tutorDatasetSha256;
    artifacts.reviewManifest = Buffer.from(JSON.stringify(manifest));
    certification.reviewManifestSha256 = sha256Hex(artifacts.reviewManifest);
    const result = validateReleaseEvidence(certification, artifacts);
    expect(result.valid).toBe(false);
    expect(result.errors).toContain("invented_teacher_intent must equal 0; observed 1");
  });

  it("rejects a changed artifact whose digest no longer matches certification", () => {
    const { certification, artifacts } = fixture();
    artifacts.geminiZdrEvidence = Buffer.from("different evidence");
    const result = validateReleaseEvidence(certification, artifacts);
    expect(result.valid).toBe(false);
    expect(result.errors).toContain("geminiZdrEvidence digest does not match geminiZdrEvidenceSha256");
  });

  it("rejects a canary stage that crossed an automatic halt threshold", () => {
    const subject = fixture();
    const canary = JSON.parse(subject.artifacts.canaryEvidence.toString("utf8"));
    canary.stages.find((stage) => stage.stage === "FIVE_PERCENT").providerFailureDelta = 0.03;
    subject.artifacts.canaryEvidence = Buffer.from(JSON.stringify(canary));
    rehashArtifact(subject, "canaryEvidence");
    const result = validateReleaseEvidence(subject.certification, subject.artifacts);
    expect(result.valid).toBe(false);
    expect(result.errors).toContain("canary stage FIVE_PERCENT provider failures exceeded threshold");
  });

  it("rejects a candidate that misses the tool-grounded latency budget", () => {
    const subject = fixture();
    const comparison = JSON.parse(subject.artifacts.modelComparison.toString("utf8"));
    comparison.candidate.latencyMs.toolP95 = 10_000;
    subject.artifacts.modelComparison = Buffer.from(JSON.stringify(comparison));
    rehashArtifact(subject, "modelComparison");
    const result = validateReleaseEvidence(subject.certification, subject.artifacts);
    expect(result.valid).toBe(false);
    expect(result.errors).toContain("candidate tool-grounded p95 latency is not below 10000 ms");
  });
});
