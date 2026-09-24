import type { Claim, Evidence } from "../../schemas";
import { EvidenceGraph } from "../evidence/graph";

export interface VerificationReport {
  passed: boolean;
  claims: Claim[];
  failures: string[];
}

const restrictedStablePatterns = /teacher|mark scheme|syllabus|exam date|admission|current|latest|according to|student (?:always|usually|struggles)/i;
const STOPWORDS = new Set(["the", "a", "an", "is", "are", "was", "were", "to", "of", "and", "that", "this", "it", "in", "on", "for", "with"]);
function meaningfulTokens(value: string): Set<string> {
  return new Set(value.toLowerCase().match(/[a-z0-9₂₃]+/g)?.filter((token) => token.length > 2 && !STOPWORDS.has(token)) ?? []);
}

export function verifyClaims(claims: readonly Claim[], evidence: readonly Evidence[]): VerificationReport {
  const graph = new EvidenceGraph(evidence);
  const failures: string[] = [];
  const verified = claims.map((original) => {
    const claim = structuredClone(original);
    const supporting = graph.supportFor(claim);
    const missing = claim.evidenceIds.filter((id) => !graph.evidence(id));
    if (missing.length) failures.push(`${claim.id}: missing evidence ${missing.join(", ")}`);
    if (claim.type === "stable" && restrictedStablePatterns.test(claim.text)) {
      failures.push(`${claim.id}: restricted claim cannot use stable knowledge`);
    }
    if (supporting.length === 0) failures.push(`${claim.id}: unsupported claim`);
    if (claim.type === "stable" && !supporting.some((item) => item.source === "stable_knowledge" && item.verification === "verified")) {
      failures.push(`${claim.id}: stable claim is absent from canonical knowledge`);
    }
    if (claim.type === "stable" && supporting.length > 0) {
      const claimTokens = meaningfulTokens(claim.text);
      const evidenceTokens = meaningfulTokens(supporting.map((item) => typeof item.value === "string" ? item.value : JSON.stringify(item.value)).join(" "));
      const overlap = [...claimTokens].filter((token) => evidenceTokens.has(token)).length;
      if (claimTokens.size > 0 && overlap < Math.min(2, claimTokens.size)) failures.push(`${claim.id}: canonical evidence does not support the stable claim text`);
    }
    if (claim.type === "calculation" && !supporting.some((item) => item.source === "tool" && item.verification === "verified")) {
      failures.push(`${claim.id}: calculation lacks verified tool evidence`);
    }
    if (claim.type === "retrieved" && !supporting.some((item) =>
      (item.source === "retrieval" || item.source === "official_source") && item.verification === "verified")) {
      failures.push(`${claim.id}: retrieved claim lacks verified retrieval evidence`);
    }
    if (claim.type === "retrieved" && supporting.length > 0) {
      const claimTokens = meaningfulTokens(claim.text);
      const sourceTokens = meaningfulTokens(supporting.filter((item) => item.source === "retrieval" || item.source === "official_source").map((item) => JSON.stringify(item.value)).join(" "));
      const overlap = [...claimTokens].filter((token) => sourceTokens.has(token)).length;
      if (claimTokens.size > 0 && overlap < Math.min(2, claimTokens.size)) failures.push(`${claim.id}: retrieval evidence does not support the claim text`);
    }
    const claimFailures = failures.some((failure) => failure.startsWith(`${claim.id}:`));
    claim.verificationStatus = claimFailures ? "rejected" : "verified";
    return claim;
  });
  return { passed: failures.length === 0, claims: verified, failures };
}

export function detectContradictions(claims: readonly Claim[], evidence: readonly Evidence[]): string[] {
  const failures: string[] = [];
  const normalizedEvidence = evidence.map((item) => JSON.stringify(item.value).toLowerCase());
  for (const claim of claims) {
    const text = claim.text.toLowerCase();
    const awarded = text.match(/(?:awarded|received|got)\s+(\d+(?:\.\d+)?)\s*(?:marks?)?/);
    if (awarded) {
      const teacherMarks = evidence.filter((item) => item.source === "teacher" && typeof item.value === "object" && item.value !== null)
        .map((item) => (item.value as Record<string, unknown>)["marksAwarded"])
        .filter((value): value is number => typeof value === "number");
      const claimed = Number(awarded[1]);
      if (teacherMarks.length > 0 && !teacherMarks.includes(claimed)) failures.push(`${claim.id}: contradicts recorded teacher mark`);
    }
    if (claim.evidenceIds.length > 0 && !claim.evidenceIds.some((id) => evidence.some((item) => item.id === id))) {
      failures.push(`${claim.id}: references absent evidence`);
    }
    if (text.includes("teacher intended") && !normalizedEvidence.some((value) => value.includes("intended"))) {
      failures.push(`${claim.id}: invents teacher intent`);
    }
  }
  return failures;
}

export function detectEvidenceConflicts(evidence: readonly Evidence[]): string[] {
  const groups = new Map<string, Array<{ id: string; assertion: string }>>();
  for (const item of evidence) {
    if ((item.source !== "retrieval" && item.source !== "official_source") || typeof item.value !== "object" || item.value === null) continue;
    const record = item.value as Record<string, unknown>;
    if (typeof record["factKey"] !== "string" || typeof record["assertion"] !== "string") continue;
    const group = groups.get(record["factKey"]) ?? [];
    group.push({ id: item.id, assertion: record["assertion"].trim().toLowerCase() });
    groups.set(record["factKey"], group);
  }
  return [...groups.entries()].flatMap(([factKey, items]) => new Set(items.map((item) => item.assertion)).size > 1 ? [`conflicting evidence for ${factKey}: ${items.map((item) => item.id).join(", ")}`] : []);
}
