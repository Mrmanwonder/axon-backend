import type { RiskLevel } from "../intelligence/routing/router";

export type EvalCategory = "scanner" | "tutor" | "retrieval" | "academic" | "adversarial" | "latency";
export interface EvalCase { id: string; category: EvalCategory; risk: RiskLevel; input: unknown; expected: Record<string, unknown>; tags: string[] }
export interface CaseResult { caseId: string; passed: boolean; latencyMs: number; metrics: Record<string, number>; failures: string[] }
export interface EvalRunSummary { total: number; passed: number; passRate: number; metrics: Record<string, number>; releaseAllowed: boolean; gateFailures: string[] }
export interface BenchmarkCoverage {
  scannerPapers: number;
  scannerQuestions: number;
  tutorCases: number;
  handReviewedTutorCases: number;
  privacyCertified: boolean;
  rollbackValidated: boolean;
}
