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

export function minimizePublicRetrievalQuery(query: string): string {
  const normalized = query.replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim();
  const sensitive = [
    /\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b/i,
    /(?:\+?\d[\s().-]*){7,}/,
    /\b(?:student\s*id|date\s*of\s*birth|dob|home\s*address|my\s+name\s+is)\b/i
  ];
  if (sensitive.some((pattern) => pattern.test(normalized))) throw new Error("RETRIEVAL_BLOCKED_SENSITIVE_QUERY");
  if (!normalized || normalized.length > 400) throw new Error("RETRIEVAL_FAILURE invalid minimized query");
  return normalized;
}
