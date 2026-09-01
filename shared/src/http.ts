import { createClient, type SupabaseClient } from "@supabase/supabase-js";
import type { Env } from "./env.js";

export const CORS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
  "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
} as const;

export function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { ...CORS, "Content-Type": "application/json" },
  });
}

export function failure(message: string, status = 400, detail?: unknown): Response {
  return json({ error: message, detail: detail ?? null }, status);
}

/** A client scoped to the caller's own JWT, so RLS applies. Null if unauthenticated. */
export function clientFor(req: Request, env: Env): SupabaseClient | null {
  const auth = req.headers.get("Authorization");
  if (!auth?.startsWith("Bearer ")) return null;
  return createClient(env.SUPABASE_URL!, env.SUPABASE_ANON_KEY!, {
    global: { headers: { Authorization: auth } },
    auth: { persistSession: false },
  });
}

/** A service-role client that bypasses RLS. Only ever used from within a worker, never in a response path that echoes caller input back as another caller's data. */
export function serviceClient(env: Env): SupabaseClient {
  if (!env.SUPABASE_SERVICE_ROLE_KEY) {
    throw new Error("SUPABASE_SERVICE_ROLE_KEY is not set for this worker");
  }
  return createClient(env.SUPABASE_URL!, env.SUPABASE_SERVICE_ROLE_KEY, {
    auth: { persistSession: false },
  });
}

export async function readJson<T = unknown>(req: Request): Promise<T | null> {
  try {
    return (await req.json()) as T;
  } catch {
    return null;
  }
}
