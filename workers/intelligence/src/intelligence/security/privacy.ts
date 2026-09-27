function bytesToHex(bytes: ArrayBuffer): string {
  return [...new Uint8Array(bytes)].map((value) => value.toString(16).padStart(2, "0")).join("");
}

export async function pseudonymizeIdentifier(identifier: string, secret: string): Promise<string> {
  if (!secret) throw new Error("Missing pseudonymization key");
  const key = await crypto.subtle.importKey("raw", new TextEncoder().encode(secret), { name: "HMAC", hash: "SHA-256" }, false, ["sign"]);
  const signature = await crypto.subtle.sign("HMAC", key, new TextEncoder().encode(identifier));
  return `psn_${bytesToHex(signature)}`;
}

export async function oneWayHash(value: string): Promise<string> {
  return bytesToHex(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(value)));
}

const PRIVATE_RETRIEVAL_PATTERNS = [
  /\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b/i,
  /(?:\+?\d[\s().-]*){7,}/,
  /\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b/i,
  /\b(?:student\s*id|student\s*answer|my\s+answer|teacher\s*(?:remark|comment|feedback)|date\s*of\s*birth|dob|home\s*address|my\s+name\s+is)\b/i,
  /\b(?:authorization|bearer|api[_ -]?key|access[_ -]?token|refresh[_ -]?token|session[_ -]?token|jwt)\b/i,
  /https?:\/\//i,
  /(?:^|[?&])(?:sig|signature|token|key|x-amz-signature|x-amz-credential|x-goog-signature|expires|exp)=/i,
] as const;

/**
 * Final DLP gate for Tavily query material.
 *
 * This function is deliberately NOT a general chat sanitizer. It accepts only
 * server-authored public academic context. Raw student prose must never be
 * passed here as the source of a public-web query.
 */
export function minimizePublicRetrievalQuery(query: string): string {
  const normalized = query.replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim();
  if (PRIVATE_RETRIEVAL_PATTERNS.some((pattern) => pattern.test(normalized))) {
    throw new Error("RETRIEVAL_BLOCKED_SENSITIVE_QUERY");
  }
  if (!normalized || normalized.length > 400) throw new Error("RETRIEVAL_FAILURE invalid public context");
  return normalized;
}

/**
 * Reduce arbitrary chat wording to one closed, non-user-authored research facet.
 * The returned text is selected from constants only; no substring from the
 * student's message is copied into the outbound Tavily query.
 */
export function publicRetrievalFacet(message: string, intent: string): string {
  if (/\b(?:exam\s+date|timetable|schedule)\b/i.test(message)) return "official exam timetable dates";
  if (/\b(?:syllabus|specification|curriculum)\b/i.test(message)) return "official syllabus specification";
  if (/\b(?:admission|entry requirement|eligibility)\b/i.test(message)) return "official admission requirements";
  if (/\b(?:law|regulation|rule)\b/i.test(message)) return "official rules and regulations";
  if (/\b(?:statistics|statistic|data)\b/i.test(message)) return "official statistics";
  if (/\b(?:software\s+version|version)\b/i.test(message)) return "official version information";
  if (intent === "source_question") return "official source guidance";
  return "current official academic information";
}

/**
 * Construct a Tavily-safe query from trusted curriculum context plus a closed
 * intent facet. The raw user message is used only to choose a constant facet.
 */
export function buildPublicRetrievalQuery(publicContext: string | undefined, message: string, intent: string): string {
  if (!publicContext) throw new Error("RETRIEVAL_BLOCKED_NO_PUBLIC_CONTEXT");
  const context = minimizePublicRetrievalQuery(publicContext);
  return minimizePublicRetrievalQuery(`${context} ${publicRetrievalFacet(message, intent)}`);
}
