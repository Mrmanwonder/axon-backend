import type { Evidence } from "../../schemas";

export type DiagnosisCode = "conceptual_error" | "wrong_method" | "calculation_error" | "algebra_error" | "unit_error" | "sign_error" | "incomplete_reasoning" | "missing_keyword" | "misread_question" | "presentation" | "diagram_issue" | "unfinished" | "unknown";
export interface Diagnosis { primary: DiagnosisCode; evidenceIds: string[]; alternatives: DiagnosisCode[]; status: "verified" | "inferred" | "unknown" }

const toolToDiagnosis: Readonly<Record<string, DiagnosisCode>> = {
  "axon.units.v1": "unit_error", "axon.calculator.v1": "calculation_error", "axon.math.v1": "algebra_error", "axon.chem.v1": "conceptual_error"
};

export function diagnose(evidence: readonly Evidence[]): Diagnosis {
  const explicit = evidence.find((item) => item.source === "teacher" && typeof item.value === "object" && item.value !== null && "diagnosis" in item.value);
  if (explicit) {
    const value = String((explicit.value as Record<string, unknown>)["diagnosis"]);
    const known: DiagnosisCode[] = ["conceptual_error", "wrong_method", "calculation_error", "algebra_error", "unit_error", "sign_error", "incomplete_reasoning", "missing_keyword", "misread_question", "presentation", "diagram_issue", "unfinished", "unknown"];
    if (known.includes(value as DiagnosisCode)) return { primary: value as DiagnosisCode, evidenceIds: [explicit.id], alternatives: [], status: "verified" };
  }
  const failedTool = evidence.find((item) => item.source === "tool" && item.verification === "verified" && typeof item.value === "object" && item.value !== null && (item.value as Record<string, unknown>)["valid"] === false);
  const toolId = failedTool?.provenance.toolId;
  if (failedTool && toolId && toolToDiagnosis[toolId]) return { primary: toolToDiagnosis[toolId], evidenceIds: [failedTool.id], alternatives: [], status: "verified" };
  return { primary: "unknown", evidenceIds: [], alternatives: [], status: "unknown" };
}
