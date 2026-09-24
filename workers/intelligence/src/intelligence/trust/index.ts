export type TrustState = "AUTO_VERIFIED" | "STUDENT_VERIFIED" | "UNVERIFIED" | "UNKNOWN";
export interface TrustedField<T> { value: T | null; trustState: TrustState; evidenceIds: string[] }

export function canCommitToInsights<T>(field: TrustedField<T>): boolean {
  return field.value !== null && (field.trustState === "AUTO_VERIFIED" || field.trustState === "STUDENT_VERIFIED") && field.evidenceIds.length > 0;
}

export function deterministicAccuracy(fields: readonly TrustedField<boolean>[]): { correct: number; total: number; accuracy: number | null } {
  const trusted = fields.filter(canCommitToInsights);
  const correct = trusted.filter((field) => field.value === true).length;
  return { correct, total: trusted.length, accuracy: trusted.length ? correct / trusted.length : null };
}
