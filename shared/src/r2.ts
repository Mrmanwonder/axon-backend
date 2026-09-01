import { AwsClient } from "aws4fetch";
import type { Env } from "./env.js";

export type BucketKind = "originals" | "derived";
export type ObjectKind = "upload" | "raw" | "page" | "mask" | "crop" | "cropmask" | "thumb";

const PUT_TTL_SECONDS = 900;
const GET_TTL_SECONDS = 600;

function bucketName(env: Env, bucket: BucketKind): string {
  const name = bucket === "originals" ? env.R2_BUCKET_ORIGINALS : env.R2_BUCKET_DERIVED;
  if (!name) throw new Error(`R2 bucket for '${bucket}' is not configured`);
  return name;
}

function binding(env: Env, bucket: BucketKind): R2Bucket {
  const b = bucket === "originals" ? env.ORIGINALS : env.DERIVED;
  if (!b) throw new Error(`R2 binding for '${bucket}' is not attached to this worker`);
  return b;
}

function nonce(): string {
  const bytes = crypto.getRandomValues(new Uint8Array(16));
  return btoa(String.fromCharCode(...bytes)).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

export const BUCKET_FOR: Record<ObjectKind, BucketKind> = {
  upload: "originals",
  raw: "originals",
  page: "derived",
  mask: "derived",
  crop: "derived",
  cropmask: "derived",
  thumb: "derived",
};

export interface ObjectKeyOptions {
  studentId: string;
  paperId: string;
  kind: ObjectKind;
  name: string;
  extension: string;
  /** Set false for a deterministic key (no random suffix) — e.g. a crop derived from a fixed region. */
  unguessable?: boolean;
}

export function objectKey(opts: ObjectKeyOptions): string {
  const ext = opts.extension.replace(/^\./, "");
  const stem = opts.unguessable === false ? String(opts.name) : `${opts.name}-${nonce()}`;
  return `${opts.studentId}/${opts.paperId}/${opts.kind}/${stem}.${ext}`;
}

function objectUrl(env: Env, bucket: BucketKind, key: string): string {
  const path = key.split("/").map(encodeURIComponent).join("/");
  return `${env.R2_ENDPOINT!.replace(/\/+$/, "")}/${bucketName(env, bucket)}/${path}`;
}

function signer(env: Env): AwsClient {
  if (!env.R2_ACCESS_KEY_ID || !env.R2_SECRET_ACCESS_KEY) {
    throw new Error("R2 credentials are not set for this worker");
  }
  return new AwsClient({
    accessKeyId: env.R2_ACCESS_KEY_ID,
    secretAccessKey: env.R2_SECRET_ACCESS_KEY,
    service: "s3",
    region: "auto",
  });
}

/** A presigned PUT URL the client uploads directly to R2 with, bypassing the Worker for the bytes themselves. */
export async function presignPut(env: Env, bucket: BucketKind, key: string, contentType: string, ttlSeconds = PUT_TTL_SECONDS): Promise<string> {
  const url = new URL(objectUrl(env, bucket, key));
  url.searchParams.set("X-Amz-Expires", String(ttlSeconds));
  const signed = await signer(env).sign(
    new Request(url, { method: "PUT", headers: { "Content-Type": contentType } }),
    { aws: { signQuery: true, allHeaders: false } } as any
  );
  return signed.url;
}

export interface HeadResult {
  bytes: number;
  etag: string | null;
  contentType: string | null;
}

export async function headObject(env: Env, bucket: BucketKind, key: string): Promise<HeadResult | null> {
  const obj = await binding(env, bucket).head(key);
  if (!obj) return null;
  return {
    bytes: obj.size,
    etag: obj.httpEtag?.replace(/"/g, "") ?? null,
    contentType: obj.httpMetadata?.contentType ?? null,
  };
}

export async function deleteObject(env: Env, bucket: BucketKind, key: string): Promise<void> {
  await binding(env, bucket).delete(key);
}

export interface DeletePrefixOptions {
  maxKeys?: number;
  cursor?: string;
}
export interface DeletePrefixResult {
  deleted: number;
  done: boolean;
  cursor?: string;
}

/** Bounded prefix delete: walks up to `maxKeys` objects per call so a single sweep tick can't run away. */
export async function deletePrefix(env: Env, bucket: BucketKind, prefix: string, opts: DeletePrefixOptions = {}): Promise<DeletePrefixResult> {
  const budget = opts.maxKeys ?? 200;
  const b = binding(env, bucket);
  let cursor = opts.cursor;
  let deleted = 0;
  while (deleted < budget) {
    const listing = await b.list({ prefix, limit: Math.min(1000, budget - deleted), cursor });
    if (listing.objects.length) {
      await Promise.all(listing.objects.map((o) => b.delete(o.key)));
      deleted += listing.objects.length;
    }
    if (listing.truncated) {
      cursor = listing.cursor;
    } else {
      return { deleted, done: true };
    }
  }
  return { deleted, done: false, cursor };
}

async function hmacKey(secret: string): Promise<CryptoKey> {
  return crypto.subtle.importKey("raw", new TextEncoder().encode(secret), { name: "HMAC", hash: "SHA-256" }, false, ["sign", "verify"]);
}

function base64url(bytes: Uint8Array): string {
  let binary = "";
  for (const b of bytes) binary += String.fromCharCode(b);
  return btoa(binary).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

function timingSafeEqual(a: string, b: string): boolean {
  if (a.length !== b.length) return false;
  let diff = 0;
  for (let i = 0; i < a.length; i++) diff |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return diff === 0;
}

/** A short-lived, HMAC-signed URL to mastery-api's /asset proxy — the path the frontend actually reads pages and crops through. */
export async function signAssetUrl(env: Env, bucket: BucketKind, key: string, ttlSeconds = GET_TTL_SECONDS): Promise<string> {
  const exp = Math.floor(Date.now() / 1000) + ttlSeconds;
  const cryptoKey = await hmacKey(env.ASSET_SIGNING_SECRET!);
  const mac = await crypto.subtle.sign("HMAC", cryptoKey, new TextEncoder().encode(`${bucket}:${key}:${exp}`));
  const sig = base64url(new Uint8Array(mac));
  const base = env.MASTERY_ASSET_URL ?? "https://mastery-api.workers.dev";
  return `${base}/asset/${bucket}/${encodeURIComponent(key)}?exp=${exp}&sig=${sig}`;
}

export async function verifyAssetSignature(env: Env, bucket: BucketKind, key: string, exp: number, sig: string): Promise<boolean> {
  if (!exp || exp < Date.now() / 1000) return false;
  const cryptoKey = await hmacKey(env.ASSET_SIGNING_SECRET!);
  const mac = await crypto.subtle.sign("HMAC", cryptoKey, new TextEncoder().encode(`${bucket}:${key}:${exp}`));
  const expected = base64url(new Uint8Array(mac));
  return timingSafeEqual(expected, sig);
}

const imageRefCache: Map<string, string> =
  (globalThis as any).__imageRefCache ?? ((globalThis as any).__imageRefCache = new Map());

/**
 * Reads an R2 object and returns it as a data: URL suitable for a model's
 * image_url content part. `detail` is accepted for the caller's intent (and
 * some callers — triage — pass "low" deliberately) but is NOT currently sent
 * to the model; see AXON_FIX_BRIEF.md §C4 for why re-adding it needs its own
 * tested change, not a silent fix bundled into this port.
 */
export async function imageRef(env: Env, bucket: BucketKind, key: string, detail: "low" | "high" = "high"): Promise<{ url: string; key: string; detail: "low" | "high" }> {
  const cacheKey = bucket + "/" + key;
  let dataUrl = imageRefCache.get(cacheKey);
  if (!dataUrl) {
    const b = bucket === "originals" ? env.ORIGINALS : env.DERIVED;
    if (!b) throw new Error("imageRef: R2 binding not configured for bucket " + bucket);
    const obj = await b.get(key);
    if (!obj) throw new Error("imageRef: object not found " + bucket + "/" + key);
    const mime = obj.httpMetadata?.contentType || "image/jpeg";
    const bytes = new Uint8Array(await obj.arrayBuffer());
    let b64: string;
    if (typeof (bytes as any).toBase64 === "function") {
      b64 = (bytes as any).toBase64();
    } else if (typeof Buffer !== "undefined") {
      b64 = Buffer.from(bytes).toString("base64");
    } else {
      let binary = "";
      const chunkSize = 8192;
      for (let i = 0; i < bytes.length; i += chunkSize) {
        binary += String.fromCharCode.apply(null, Array.from(bytes.subarray(i, i + chunkSize)));
      }
      b64 = btoa(binary);
    }
    dataUrl = `data:${mime};base64,${b64}`;
    if (imageRefCache.size > 50) imageRefCache.clear();
    imageRefCache.set(cacheKey, dataUrl);
  }
  return { url: dataUrl, key, detail };
}
