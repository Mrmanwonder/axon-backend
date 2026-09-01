// Cloudflare Worker bindings shared across the mastery-* pipeline.
//
// Not every worker binds every one of these — see each worker's wrangler.toml
// for the actual subset. Marking them optional here means a worker that omits
// a binding still type-checks; the guards at each call site (`if (env.X)`,
// `if (!key) throw ...`) are what actually enforce presence at runtime.
export interface Env {
  // Supabase
  SUPABASE_URL?: string;
  SUPABASE_SERVICE_ROLE_KEY?: string;
  // api only: used to build a request-scoped client that carries the caller's
  // own JWT, so RLS applies to student-facing endpoints.
  SUPABASE_ANON_KEY?: string;

  // Model
  GOOGLE_API_KEY?: string;
  // Legacy, from before the Gemini migration (see AXON_FIX_BRIEF.md §F3).
  // Bindings still exist on the deployed workers; nothing reads them anymore.
  GROQ_API_KEY?: string;
  OPENROUTER_API_KEY?: string;
  OPENCODE_API_KEY?: string;

  // Asset signing (shared/r2.ts signAssetUrl / verifyAssetSignature)
  ASSET_SIGNING_SECRET?: string;
  MASTERY_ASSET_URL?: string;
  // Sent as the OpenAI-compatible request's HTTP-Referer; cosmetic.
  MASTERY_SITE_URL?: string;

  // R2 buckets (bound directly, for read/delete from within a Worker)
  ORIGINALS?: R2Bucket;
  DERIVED?: R2Bucket;

  // R2 S3-compatible credentials (api/sweep only, for presigned PUT URLs and
  // aws4fetch-signed requests via shared/r2.ts's AwsClient)
  R2_ACCESS_KEY_ID?: string;
  R2_SECRET_ACCESS_KEY?: string;
  R2_ENDPOINT?: string;
  R2_BUCKET_ORIGINALS?: string;
  R2_BUCKET_DERIVED?: string;

  // Queues. Each worker only binds the ones it produces to or consumes from
  // (see AXON_FIX_BRIEF.md §4.D2 — reconcile is missing SELF_QUEUE, which is
  // a deliberate gap tracked for §9.2, not an omission here).
  TRIAGE_QUEUE?: Queue;
  STRUCTURE_QUEUE?: Queue;
  CROP_QUEUE?: Queue;
  CONTENT_QUEUE?: Queue;
  RECONCILE_QUEUE?: Queue;
  ADJUDICATE_QUEUE?: Queue;
  EXPLAIN_QUEUE?: Queue;
  SELF_QUEUE?: Queue;
}
