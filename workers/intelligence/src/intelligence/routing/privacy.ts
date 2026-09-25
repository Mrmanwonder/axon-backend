export type PrivacyPolicyId = "STUDENT_DOCUMENT_STRICT" | "STUDENT_CHAT_STRICT" | "PUBLIC_RESEARCH" | "SYNTHETIC_EVAL";

export interface PrivacyPolicy {
  id: PrivacyPolicyId;
  allowedProviders: readonly string[];
  retention: "zero" | "bounded";
  trainingAllowed: boolean;
  rawLoggingAllowed: boolean;
  imageHandling: "prohibited" | "ephemeral" | "allowed";
  piiHandling: "prohibited" | "minimized";
  fallbackMayRelax: false;
}

export const PRIVACY_POLICIES: Record<PrivacyPolicyId, PrivacyPolicy> = {
  STUDENT_DOCUMENT_STRICT: { id: "STUDENT_DOCUMENT_STRICT", allowedProviders: ["gemini-zdr"], retention: "zero", trainingAllowed: false, rawLoggingAllowed: false, imageHandling: "ephemeral", piiHandling: "minimized", fallbackMayRelax: false },
  STUDENT_CHAT_STRICT: { id: "STUDENT_CHAT_STRICT", allowedProviders: ["gemini-zdr"], retention: "zero", trainingAllowed: false, rawLoggingAllowed: false, imageHandling: "prohibited", piiHandling: "minimized", fallbackMayRelax: false },
  PUBLIC_RESEARCH: { id: "PUBLIC_RESEARCH", allowedProviders: ["gemini-zdr", "tavily"], retention: "bounded", trainingAllowed: false, rawLoggingAllowed: false, imageHandling: "prohibited", piiHandling: "prohibited", fallbackMayRelax: false },
  SYNTHETIC_EVAL: { id: "SYNTHETIC_EVAL", allowedProviders: ["gemini-zdr"], retention: "bounded", trainingAllowed: false, rawLoggingAllowed: true, imageHandling: "allowed", piiHandling: "prohibited", fallbackMayRelax: false }
};

export class NoCompliantProviderError extends Error {
  constructor(policy: PrivacyPolicyId) {
    super(`NO_COMPLIANT_PROVIDER for ${policy}`);
    this.name = "NoCompliantProviderError";
  }
}
