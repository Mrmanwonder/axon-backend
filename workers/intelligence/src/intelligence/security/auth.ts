import { timingSafeEqual } from "node:crypto";

async function digest(value: string): Promise<ArrayBuffer> {
  return crypto.subtle.digest("SHA-256", new TextEncoder().encode(value));
}
export async function constantTimeEqual(provided: string, expected: string): Promise<boolean> {
  const [left, right] = await Promise.all([digest(provided), digest(expected)]);
  return timingSafeEqual(new Uint8Array(left), new Uint8Array(right));
}

export async function authenticateInternalRequest(request: Request, expectedToken?: string): Promise<boolean> {
  if (!expectedToken) return false;
  const authorization = request.headers.get("authorization");
  if (!authorization?.startsWith("Bearer ")) return false;
  return constantTimeEqual(authorization.slice("Bearer ".length), expectedToken);
}
