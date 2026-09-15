import { CORS, json, failure, clientFor, readJson, serviceClient } from "@mastery/shared/http.js";
import { presignPut, headObject, signAssetUrl, verifyAssetSignature, objectKey, BUCKET_FOR, type BucketKind } from "@mastery/shared/r2.js";
import { PIPELINE_VERSION } from "@mastery/shared/contract.js";
import type { Env } from "@mastery/shared/env.js";

const MAX_PAGES = 25;
const MAX_BYTES = 25 * 1024 * 1024;
const MAX_OBJECTS = 60;
const IO_CONCURRENCY = 8;

const ALLOWED_CONTENT_TYPES: Record<string, string[]> = {
  "image/webp": ["webp"],
  "image/jpeg": ["jpg"],
  "image/png": ["png"],
  "image/heic": ["heic"],
  "application/pdf": ["pdf"],
};

/**
 * Run independent remote operations concurrently without turning a booklet into
 * an unbounded burst of Worker subrequests. Ordering is preserved so callers can
 * safely zip results back to the input array.
 */
async function mapLimit<T, R>(
  items: readonly T[],
  limit: number,
  fn: (item: T, index: number) => Promise<R>,
): Promise<R[]> {
  if (!items.length) return [];
  const out = new Array<R>(items.length);
  let cursor = 0;
  const workers = Array.from({ length: Math.min(limit, items.length) }, async () => {
    while (true) {
      const index = cursor++;
      if (index >= items.length) return;
      out[index] = await fn(items[index], index);
    }
  });
  await Promise.all(workers);
  return out;
}

export default {
  async fetch(req: Request, env: Env): Promise<Response> {
    if (req.method === "OPTIONS") return new Response("ok", { headers: CORS });
    const url = new URL(req.url);
    const path = url.pathname.replace(/\/+$/, "");
    try {
      if (path === "/asset" || path.startsWith("/asset/")) return await serveAsset(req, env, url);
      if (req.method !== "POST") return failure("not found", 404);
      switch (path) {
        case "/paper-submit":
          return await paperSubmit(req, env);
        case "/upload-intent":
          return await uploadIntent(req, env);
        case "/upload-complete":
          return await uploadComplete(req, env);
        case "/review-complete":
          return await reviewComplete(req, env);
        case "/page-asset-urls":
          return await pageAssetUrls(req, env);
        default:
          return failure("not found", 404);
      }
    } catch (cause) {
      console.error("mastery-api unhandled error", String(cause));
      return failure("Something went wrong on our end. Nothing was lost — try again.", 500);
    }
  },
} satisfies ExportedHandler<Env>;

async function serveAsset(req: Request, env: Env, url: URL): Promise<Response> {
  const parts = url.pathname.split("/").filter(Boolean);
  if (parts.length < 3 || parts[0] !== "asset") return failure("not found", 404);
  const bucket = parts[1] as BucketKind;
  const key = decodeURIComponent(parts.slice(2).join("/"));
  const exp = Number(url.searchParams.get("exp"));
  const sig = url.searchParams.get("sig") ?? "";
  if (bucket !== "originals" && bucket !== "derived") return failure("unknown bucket", 400);
  const ok = await verifyAssetSignature(env, bucket, key, exp, sig);
  if (!ok) return failure("expired or invalid", 403);
  const binding = bucket === "originals" ? env.ORIGINALS : env.DERIVED;
  if (!binding) return failure("asset storage is not configured", 500);
  const obj = await binding.get(key);
  if (!obj) return failure("not found", 404);
  return new Response(obj.body, {
    headers: {
      ...CORS,
      "Content-Type": obj.httpMetadata?.contentType ?? "application/octet-stream",
      // Signed asset URLs are private to the browser and immutable by key. Keep
      // this comfortably below the signature TTL while avoiding a Worker + R2
      // round trip every time the same crop is reopened.
      "Cache-Control": "private, max-age=300, immutable",
    },
  });
}

async function paperSubmit(req: Request, env: Env): Promise<Response> {
  const user = clientFor(req, env);
  if (!user) return failure("Sign in first.", 401);
  const body = await readJson<any>(req);
  if (!body?.student_id || !body?.type || !body?.idempotency_key) {
    return failure("That paper is missing something we need to file it.");
  }
  if (!Array.isArray(body.pages) || !body.pages.length) return failure("A paper needs at least one page.");
  if (body.pages.length > MAX_PAGES) return failure(`We can take up to ${MAX_PAGES} pages in one paper.`);
  if (body.pages.some((p: any) => !p.r2_key || !Number.isInteger(p.page_number) || p.page_number < 1)) {
    return failure("One of those pages has not finished uploading.");
  }

  const { data, error } = await user.rpc("submit_paper", {
    p_student_id: body.student_id,
    p_type: body.type,
    p_tier: body.tier ?? "tier_1",
    p_date_taken: body.date_taken ?? null,
    p_subject: body.subject ?? null,
    p_pages: body.pages,
    p_idempotency_key: body.idempotency_key,
    p_reported_total: body.reported_total ?? null,
    p_stated_maximum: body.stated_maximum ?? null,
    p_pipeline_version: PIPELINE_VERSION,
    p_paper_id: body.paper_id ?? null,
  });
  if (error) return failure("We could not save that paper. Nothing was lost — try again.", 500, error.message);

  const runId = data.run_id;
  const admin = serviceClient(env);
  const { data: run } = await admin.from("extraction_run").select("status").eq("id", runId).single();
  if (run?.status === "queued") {
    if (!env.TRIAGE_QUEUE) return json({ ...data, queued: false, reason: "Reading has not started yet." }, 202);
    try {
      await env.TRIAGE_QUEUE.send({ run_id: runId });
      await admin.rpc("run_advance", { p_run_id: runId, p_to: "queued" });
    } catch {
      return json({ ...data, queued: false, reason: "Reading has not started yet." }, 202);
    }
  }
  return json({ ...data, queued: true }, 202);
}

async function uploadIntent(req: Request, env: Env): Promise<Response> {
  const user = clientFor(req, env);
  if (!user) return failure("Sign in first.", 401);
  const body = await readJson<any>(req);
  if (!body?.student_id || !body?.paper_id || !Array.isArray(body.objects) || !body.objects.length) {
    return failure("Nothing to upload.");
  }
  if (body.objects.length > MAX_OBJECTS) return failure(`That is more than ${MAX_OBJECTS} files in one go.`);

  const { data: paper } = await user.from("paper").select("id, student_id").eq("id", body.paper_id).eq("student_id", body.student_id).maybeSingle();
  if (!paper) return failure("That paper is not yours.", 403);

  // Validate the whole request before creating any ledger rows or signing URLs.
  // A bad object in slot N can no longer leave N-1 partial upload intents behind.
  const prepared: Array<{
    object: any;
    bucket: BucketKind;
    key: string;
  }> = [];
  for (const object of body.objects) {
    const extensions = ALLOWED_CONTENT_TYPES[object.content_type];
    if (!extensions) return failure(`We cannot take a ${object.content_type} file.`);
    if (object.bytes && object.bytes > MAX_BYTES) return failure("One of those files is too large to upload.");
    const bucket = BUCKET_FOR[object.kind as keyof typeof BUCKET_FOR];
    if (!bucket) return failure("Unknown file kind.");
    const key = objectKey({
      studentId: body.student_id,
      paperId: body.paper_id,
      kind: object.kind,
      name: object.name,
      extension: extensions[0],
    });
    prepared.push({ object, bucket, key });
  }

  const admin = serviceClient(env);
  const ledgerRows = prepared
    .filter(({ object }) => object.kind === "upload" || object.kind === "raw")
    .map(({ object, bucket, key }) => ({
      paper_id: body.paper_id,
      student_id: body.student_id,
      kind: object.content_type === "application/pdf" ? "pdf" : "image",
      r2_bucket: bucket,
      r2_key: key,
      content_type: object.content_type,
    }));

  const ledgerByKey = new Map<string, string>();
  if (ledgerRows.length) {
    const { data: rows, error } = await admin
      .from("upload")
      .insert(ledgerRows)
      .select("id,r2_key");
    if (error) {
      return failure("We could not prepare those files for upload. Nothing was uploaded yet.", 500, error.message);
    }
    for (const row of rows ?? []) ledgerByKey.set(row.r2_key, row.id);
  }

  // Presigning is local crypto; all URLs are independent and can be minted in
  // parallel once ownership and validation have succeeded.
  const minted = await Promise.all(prepared.map(async ({ object, bucket, key }) => ({
    kind: object.kind,
    name: object.name,
    bucket,
    key,
    url: await presignPut(env, bucket, key, object.content_type),
    ...(ledgerByKey.has(key) ? { upload_id: ledgerByKey.get(key) } : {}),
  })));

  return json({ objects: minted, expires_in: 900 });
}

async function uploadComplete(req: Request, env: Env): Promise<Response> {
  const user = clientFor(req, env);
  if (!user) return failure("Sign in first.", 401);
  const body = await readJson<any>(req);
  if (!body?.paper_id || !Array.isArray(body.uploads) || !body.uploads.length) return failure("Nothing to confirm.");
  if (body.uploads.length > MAX_OBJECTS) return failure(`That is more than ${MAX_OBJECTS} files in one go.`);

  const { data: paper } = await user.from("paper").select("id, student_id").eq("id", body.paper_id).maybeSingle();
  if (!paper) return failure("That paper is not yours.", 403);

  type Checked = {
    key: string;
    claim: any;
    head: Awaited<ReturnType<typeof headObject>> | null;
    reason?: string;
  };

  // R2 HEADs do not depend on one another. Bound concurrency avoids the former
  // N*RTT waterfall without letting a malicious 60-object request fan out all
  // at once.
  const checked = await mapLimit<any, Checked>(body.uploads, IO_CONCURRENCY, async (claim) => {
    if (!claim.key?.startsWith(`${paper.student_id}/${paper.id}/`)) {
      return { key: claim.key ?? "", claim, head: null, reason: "that file does not belong to this paper" };
    }
    try {
      const head = await headObject(env, claim.bucket, claim.key);
      if (!head) return { key: claim.key, claim, head: null, reason: "that file did not arrive" };
      if (claim.bytes && claim.bytes !== head.bytes) {
        return { key: claim.key, claim, head, reason: "that file arrived incomplete" };
      }
      return { key: claim.key, claim, head };
    } catch (cause) {
      return { key: claim.key, claim, head: null, reason: `we could not check that file (${cause})` };
    }
  });

  const missing = checked
    .filter((item) => item.reason)
    .map((item) => ({ key: item.key, reason: item.reason! }));
  const valid = checked.filter((item) => !item.reason && item.head);

  const admin = serviceClient(env);
  await mapLimit(valid, IO_CONCURRENCY, async ({ claim, head }) => {
    const { error } = await admin
      .from("upload")
      .update({
        confirmed: true,
        bytes: head!.bytes,
        etag: head!.etag,
        sha256: claim.sha256 ?? null,
      })
      .eq("paper_id", body.paper_id)
      .eq("r2_key", claim.key);
    if (error) throw error;
  });

  return json({ confirmed: valid.map((item) => item.key), missing }, missing.length ? 409 : 200);
}

async function pageAssetUrls(req: Request, env: Env): Promise<Response> {
  const user = clientFor(req, env);
  if (!user) return failure("Sign in first.", 401);
  const body = await readJson<any>(req);
  if (!body?.paper_id || !Array.isArray(body.page_numbers) || !body.page_numbers.length) {
    return failure("Which pages?");
  }
  const { data: pages, error } = await user
    .from("paper_page")
    .select("page_number, r2_bucket, r2_key, mask_key")
    .eq("paper_id", body.paper_id)
    .in("page_number", body.page_numbers);
  if (error) return failure("We could not look up those pages.", 500, error.message);

  const signed = await Promise.all((pages ?? []).map(async (page) => {
    const bucket = (page.r2_bucket as BucketKind) ?? "derived";
    const [url, maskUrl] = await Promise.all([
      page.r2_key ? signAssetUrl(env, bucket, page.r2_key) : Promise.resolve(null),
      page.mask_key ? signAssetUrl(env, bucket, page.mask_key) : Promise.resolve(null),
    ]);
    return [page.page_number, { url, mask_url: maskUrl }] as const;
  }));

  return json({ urls: Object.fromEntries(signed) });
}

// This is the ONLY place that starts explanations: it gates on every
// review-required region being confirmed, then calls begin_explanations and
// fans out to EXPLAIN_QUEUE.
async function reviewComplete(req: Request, env: Env): Promise<Response> {
  const user = clientFor(req, env);
  if (!user) return failure("Sign in first.", 401);
  const body = await readJson<any>(req);
  if (!body?.run_id) return failure("Which paper?");

  const { data: run } = await user.from("extraction_run").select("id, status").eq("id", body.run_id).maybeSingle();
  if (!run) return failure("That paper is not yours.", 403);

  const { count } = await user
    .from("question_region")
    .select("id", { count: "exact", head: true })
    .eq("run_id", body.run_id)
    .eq("needs_review", true)
    .is("student_confirmed_at", null);
  if ((count ?? 0) > 0) {
    return failure(`${count} question${count === 1 ? "" : "s"} still need${count === 1 ? "s" : ""} your eyes.`, 409, { outstanding: count });
  }

  const admin = serviceClient(env);
  const { data: begin, error } = await admin.rpc("begin_explanations", { p_run_id: body.run_id });
  if (error) return failure("We could not start the explanations. Your corrections are saved.", 500, error.message);

  const regionIds: string[] = begin?.region_ids ?? [];
  if (env.EXPLAIN_QUEUE) {
    await mapLimit(regionIds, 16, (regionId) =>
      env.EXPLAIN_QUEUE!.send({ run_id: body.run_id, region_id: regionId }),
    );
  }
  return json({ run_id: body.run_id, explaining: begin?.queued ?? 0 });
}
