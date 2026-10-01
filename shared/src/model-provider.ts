// Where a model call goes. `model_route.provider` selects the endpoint, so moving
// from Google AI Studio to Vertex AI (zero data retention) is a route change, not a
// code change. Both speak the same OpenAI-compatible chat-completions contract.
import type { Env } from "./env.js";

export type ModelProvider = "ai_studio" | "vertex";

export interface ProviderTarget {
  url: string;
  headers: Record<string, string>;
  /** The model name in the form this provider expects. */
  model: string;
}

const AI_STUDIO_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions";

export class ProviderConfigError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ProviderConfigError";
  }
}

/** Providers return the served model with a publisher prefix on some endpoints. */
export function normalizeServedModel(model: string | null | undefined): string | null {
  if (!model) return null;
  return model.replace(/^(models|publishers\/google\/models)\//, "").replace(/^google\//, "");
}

export async function resolveProviderTarget(
  env: Env,
  provider: ModelProvider | null | undefined,
  model: string,
): Promise<ProviderTarget> {
  if (provider === "vertex") {
    const project = env.VERTEX_PROJECT;
    if (!project) throw new ProviderConfigError("VERTEX_PROJECT is not set for this worker");
    const location = env.VERTEX_LOCATION || "global";
    const host = location === "global" ? "aiplatform.googleapis.com" : `${location}-aiplatform.googleapis.com`;
    const token = await vertexAccessToken(env);
    return {
      url: `https://${host}/v1beta1/projects/${project}/locations/${location}/endpoints/openapi/chat/completions`,
      headers: { Authorization: `Bearer ${token}`, "Content-Type": "application/json" },
      model: model.startsWith("google/") ? model : `google/${model}`,
    };
  }

  const key = env.GOOGLE_API_KEY;
  if (!key) throw new ProviderConfigError("GOOGLE_API_KEY is not set for this worker");
  return {
    url: AI_STUDIO_URL,
    headers: { Authorization: `Bearer ${key}`, "Content-Type": "application/json" },
    model,
  };
}

// ── Vertex AI auth: service-account JWT bearer grant, no SDK ─────────────────

interface ServiceAccount {
  client_email: string;
  private_key: string;
  token_uri?: string;
}

const tokenCache = new Map<string, { token: string; expiresAt: number }>();

function base64url(input: ArrayBuffer | string): string {
  const bytes = typeof input === "string" ? new TextEncoder().encode(input) : new Uint8Array(input);
  let binary = "";
  for (const b of bytes) binary += String.fromCharCode(b);
  return btoa(binary).replaceAll("+", "-").replaceAll("/", "_").replaceAll("=", "");
}

function pemToPkcs8(pem: string): ArrayBuffer {
  const body = pem.replace(/-----BEGIN [A-Z ]+-----/, "").replace(/-----END [A-Z ]+-----/, "").replace(/\s+/g, "");
  const raw = atob(body);
  const out = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i++) out[i] = raw.charCodeAt(i);
  return out.buffer;
}

export async function signServiceAccountJwt(sa: ServiceAccount, nowSeconds: number): Promise<string> {
  const header = base64url(JSON.stringify({ alg: "RS256", typ: "JWT" }));
  const claims = base64url(JSON.stringify({
    iss: sa.client_email,
    scope: "https://www.googleapis.com/auth/cloud-platform",
    aud: sa.token_uri ?? "https://oauth2.googleapis.com/token",
    iat: nowSeconds,
    exp: nowSeconds + 3600,
  }));
  const key = await crypto.subtle.importKey(
    "pkcs8",
    pemToPkcs8(sa.private_key),
    { name: "RSASSA-PKCS1-v1_5", hash: "SHA-256" },
    false,
    ["sign"],
  );
  const signature = await crypto.subtle.sign("RSASSA-PKCS1-v1_5", key, new TextEncoder().encode(`${header}.${claims}`));
  return `${header}.${claims}.${base64url(signature)}`;
}

async function vertexAccessToken(env: Env): Promise<string> {
  const raw = env.VERTEX_SERVICE_ACCOUNT_JSON;
  if (!raw) throw new ProviderConfigError("VERTEX_SERVICE_ACCOUNT_JSON is not set for this worker");
  let sa: ServiceAccount;
  try {
    sa = JSON.parse(raw) as ServiceAccount;
  } catch {
    throw new ProviderConfigError("VERTEX_SERVICE_ACCOUNT_JSON is not valid JSON");
  }
  if (!sa.client_email || !sa.private_key) {
    throw new ProviderConfigError("VERTEX_SERVICE_ACCOUNT_JSON is missing client_email or private_key");
  }

  const now = Math.floor(Date.now() / 1000);
  const cached = tokenCache.get(sa.client_email);
  if (cached && cached.expiresAt - 60 > now) return cached.token;

  const assertion = await signServiceAccountJwt(sa, now);
  const res = await fetch(sa.token_uri ?? "https://oauth2.googleapis.com/token", {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: new URLSearchParams({
      grant_type: "urn:ietf:params:oauth:grant-type:jwt-bearer",
      assertion,
    }),
  });
  if (!res.ok) throw new ProviderConfigError(`Vertex token exchange failed (${res.status})`);
  const data = (await res.json()) as { access_token?: string; expires_in?: number };
  if (!data.access_token) throw new ProviderConfigError("Vertex token exchange returned no access_token");
  tokenCache.set(sa.client_email, { token: data.access_token, expiresAt: now + (data.expires_in ?? 3600) });
  return data.access_token;
}

/** Test seam: drop cached Vertex tokens. */
export function clearProviderTokenCache(): void {
  tokenCache.clear();
}
