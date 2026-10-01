import { CORS, corsFor, withCors, json, failure, clientFor, readJson, serviceClient } from "@mastery/shared/http.js";
import { presignPut, headObject, signAssetUrl, verifyAssetSignature, objectKey, BUCKET_FOR, type BucketKind } from "@mastery/shared/r2.js";
import { CAPTURE, PIPELINE_VERSION, SAFE_OBJECT_NAME } from "@mastery/shared/contract.js";
import type { Env } from "@mastery/shared/env.js";
import { chunkedSendBatch } from "@mastery/shared/chunked_send.js";
import { loadPaperEvidence, type TutorEvidence } from "./tutor_evidence.js";
import { jwtSubject, tutorRollout } from "./tutor_rollout.js";

const MAX_BYTES = 25 * 1024 * 1024;
const MAX_OBJECTS = 60;
export const IO_CONCURRENCY = 8;
const PAGE_PAIR_CONCURRENCY = Math.max(1, Math.floor(IO_CONCURRENCY / 2));

/**
 * Execute independent work with an explicit maximum number of in-flight
 * operations. Results preserve input order, while failures reject the whole
 * operation instead of being silently converted into success.
 */
export async function mapLimit<T, R>(
  items: readonly T[],
  limit: number,
  fn: (item: T, index: number) => Promise<R>,
): Promise<R[]> {
  if (!items.length) return [];
  if (!Number.isInteger(limit) || limit < 1) throw new Error("Concurrency limit must be a positive integer.");
  const out = new Array<R>(items.length);
  let cursor = 0;
  const workers = Array.from({ length: Math.min(limit, items.length) }, async () => {
    while (true) {
      const index = cursor++;
      if (index >= items.length) return;
      out[index] = await fn(items[index]!, index);
    }
  });
  await Promise.all(workers);
  return out;
}

export default {
  async fetch(req: Request, env: Env): Promise<Response> {
    if (req.method === "OPTIONS") return new Response("ok", { headers: corsFor(req) });
    const url = new URL(req.url);
    const path = url.pathname.replace(/\/+$/, "");
    const respond = (response: Response) => withCors(req, response);
    try {
      if (path === "/asset" || path.startsWith("/asset/")) return respond(await serveAsset(req, env, url));
      if (req.method !== "POST") return respond(failure("not found", 404));
      switch (path) {
        case "/paper-submit":
          return respond(await paperSubmit(req, env));
        case "/paper-retry":
          return respond(await paperRetry(req, env));
        case "/upload-intent":
          return respond(await uploadIntent(req, env));
        case "/upload-complete":
          return respond(await uploadComplete(req, env));
        case "/review-complete":
          return respond(await reviewComplete(req, env));
        case "/explain-retry":
          return respond(await explainRetry(req, env));
        case "/tutor":
          return respond(await tutor(req, env));
        case "/page-asset-urls":
          return respond(await pageAssetUrls(req, env));
        default:
          return respond(failure("not found", 404));
      }
    } catch (cause) {
      console.error("mastery-api unhandled error", String(cause));
      return respond(failure("Something went wrong on our end. Nothing was lost — try again.", 500));
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
      // CORS belongs on the success path too. `failure()` spreads it, so every
      // way this route could say no was readable from the browser and the one
      // way it says yes was not: the image came back 200 with no
      // Access-Control-Allow-Origin, so `fetch` rejected it before a single
      // byte reached the page. Every crop in QuestionDetail and in the review
      // screen — the provenance payoff the whole extraction contract exists to
      // pay off — failed on that missing header, and failed as "we could not
      // show this part of the page", which reads like a scan problem.
      ...CORS,
      "Content-Type": obj.httpMetadata?.contentType ?? "application/octet-stream",
      "Cache-Control": "private, max-age=60",
    },
  });
}

function cleanPublicContextPart(value: unknown, max = 100): string | null {
  if (typeof value !== "string") return null;
  const cleaned = value.replace(/\s+/g, " ").trim();
  if (!cleaned) return null;
  // Curriculum labels/codes are database-authored. URLs and obvious secrets are
  // still excluded here so retrievalContext stays a plain public identifier set.
  if (/https?:\/\//i.test(cleaned) || /\b(?:bearer|token|secret|api[_ -]?key)\b/i.test(cleaned)) return null;
  return cleaned.slice(0, max);
}

function selectedSubject(rows: any[], requested: unknown, paper: any | null): { label: string; code: string | null } | null {
  const paperLabel = cleanPublicContextPart(paper?.subject_display_snapshot ?? paper?.subject);
  const paperCode = cleanPublicContextPart(paper?.subject_external_code_snapshot);
  if (paperLabel) return { label: paperLabel, code: paperCode };

  const canonical = rows.map((row) => ({
    row,
    label: cleanPublicContextPart(row?.display_name_snapshot ?? row?.subject),
    code: cleanPublicContextPart(row?.external_code_snapshot ?? row?.syllabus_code),
  })).filter((item) => item.label);

  if (typeof requested === "string" && requested.trim()) {
    const needle = requested.trim().toLocaleLowerCase();
    const match = canonical.find(({ row, label, code }) =>
      [row?.subject, row?.display_name_snapshot, row?.external_code_snapshot, row?.syllabus_code, label, code]
        .some((value) => typeof value === "string" && value.trim().toLocaleLowerCase() === needle));
    return match ? { label: match.label!, code: match.code } : null;
  }
  return canonical.length === 1 ? { label: canonical[0]!.label!, code: canonical[0]!.code } : null;
}

async function tutor(req: Request, env: Env): Promise<Response> {
  const user = clientFor(req, env);
  if (!user) return failure("Sign in first.", 401);
  const contentLength = Number(req.headers.get("content-length") ?? "0");
  if (Number.isFinite(contentLength) && contentLength > 100_000) {
    return failure("That tutor request is too large.", 413);
  }
  const body = await readJson<any>(req);
  if (typeof body?.studentId !== "string" || !body.studentId || typeof body.message !== "string" || !body.message.trim() || body.message.length > 20_000) {
    return failure("Choose a student and write a question.");
  }

  // The guardian-owned student table remains intentionally enumerable so a
  // parent can choose a profile. It therefore cannot be the authority boundary
  // for a daily Student Mode request. Bind Tutor to the one live student scope
  // carried by this exact signed auth session before any sibling lookup occurs.
  const { data: scope, error: scopeError } = await user.rpc("student_scope_state");
  if (scopeError) return failure("We could not verify the active student.", 503);
  if (!scope?.active || scope.student_id !== body.studentId) {
    return failure("Switch to that student first.", 403);
  }
  if (!tutorRollout(env, jwtSubject(req.headers.get("authorization"))).allowed) {
    return failure("The tutor is not available yet.", 503);
  }

  const { data: student, error } = await user
    .from("student")
    .select("id,board,class_level,programme_id,stage_id")
    .eq("id", body.studentId)
    .maybeSingle();
  if (error || !student) return failure("That student profile is not yours.", 403);

  let paper: any | null = null;
  if (body.paperId !== undefined) {
    if (typeof body.paperId !== "string" || !body.paperId) return failure("Choose a valid paper.");
    const paperResult = await user
      .from("paper")
      .select("id,subject,subject_offering_id,subject_display_snapshot,subject_external_code_snapshot")
      .eq("id", body.paperId)
      .eq("student_id", student.id)
      .maybeSingle();
    if (paperResult.error || !paperResult.data) return failure("That paper is not available for this student.", 403);
    paper = paperResult.data;
  }
  if (body.questionId !== undefined) {
    if (!paper) return failure("Choose the paper this question belongs to.");
    if (typeof body.questionId !== "string" || !body.questionId || body.questionId.length > 128) {
      return failure("Choose a valid question.");
    }
  }
  if (!env.INTELLIGENCE || !env.AXON_INTERNAL_TOKEN) {
    return failure("The tutor is not available yet.", 503);
  }

  // Paper evidence comes from the database under this session's own scope,
  // never from the request body (AXO-36).
  let evidence: TutorEvidence[] = [];
  if (paper) {
    const loaded = await loadPaperEvidence(user, {
      studentId: student.id,
      paperId: paper.id,
      ...(typeof body.questionId === "string" ? { questionId: body.questionId } : {}),
    });
    if (!loaded.ok) return failure(loaded.message, loaded.status);
    evidence = loaded.evidence;
  }

  // Build public research context exclusively from authenticated database
  // records. Caller-authored board/grade/topic/subject values are never sent as
  // retrieval authority. A requested subject may only select an exact stored
  // subject row; its raw spelling is not forwarded.
  const [subjectResult, programmeResult, stageResult] = await Promise.all([
    user.from("student_subject")
      .select("subject,syllabus_code,subject_offering_id,display_name_snapshot,external_code_snapshot")
      .eq("student_id", student.id),
    student.programme_id
      ? user.from("curriculum_programme").select("id,provider_id,key,label").eq("id", student.programme_id).maybeSingle()
      : Promise.resolve({ data: null, error: null }),
    student.stage_id
      ? user.from("curriculum_stage").select("id,key,label,school_year_label,legacy_class_level").eq("id", student.stage_id).maybeSingle()
      : Promise.resolve({ data: null, error: null }),
  ]);
  const programme = programmeResult.error ? null : programmeResult.data;
  const stage = stageResult.error ? null : stageResult.data;
  const providerResult = programme?.provider_id
    ? await user.from("curriculum_provider").select("id,key,name").eq("id", programme.provider_id).maybeSingle()
    : { data: null, error: null };
  const provider = providerResult.error ? null : providerResult.data;
  const subjects = subjectResult.error ? [] : (subjectResult.data ?? []);
  const subject = selectedSubject(subjects, body.subject, paper);

  const board = cleanPublicContextPart(provider?.name ?? provider?.key ?? student.board);
  const programmeLabel = cleanPublicContextPart(programme?.label ?? programme?.key);
  const stageLabel = cleanPublicContextPart(stage?.school_year_label ?? stage?.label ?? stage?.key);
  const stageClassLevel = stage?.legacy_class_level;
  const grade = Number.isInteger(stageClassLevel)
    ? Number(stageClassLevel)
    : Number.isInteger(student.class_level) ? Number(student.class_level) : undefined;
  const retrievalContext = [board, programmeLabel, stageLabel, subject?.label, subject?.code]
    .filter((value): value is string => Boolean(value))
    .join(" ")
    .replace(/\s+/g, " ")
    .trim()
    .slice(0, 400);

  // The public gateway never accepts caller-authored evidence or retrieval
  // context. Evidence used by Intelligence must come from authenticated
  // server-side sources.
  const tutorRequest = {
    studentId: student.id,
    message: body.message.trim(),
    ...(typeof body.requestId === "string" && body.requestId.length <= 128 ? { requestId: body.requestId } : {}),
    ...(grade !== undefined ? { grade } : {}),
    ...(board ? { board } : {}),
    ...(subject ? { subject: subject.label } : {}),
    ...(retrievalContext ? { retrievalContext } : {}),
    ...(typeof body.paperId === "string" ? { paperId: body.paperId } : {}),
    ...(["BRIEF", "NORMAL", "DEEP"].includes(body.depth) ? { depth: body.depth } : {}),
    ...(evidence.length ? { evidence } : {}),
  };

  const upstream = await env.INTELLIGENCE.fetch("https://axon-intelligence.internal/v1/tutor", {
    method: "POST",
    headers: {
      "content-type": "application/json",
      authorization: `Bearer ${env.AXON_INTERNAL_TOKEN}`,
      "x-axon-client-id": student.id,
    },
    body: JSON.stringify(tutorRequest),
  });
  return new Response(upstream.body, {
    status: upstream.status,
    headers: { ...CORS, "Content-Type": upstream.headers.get("content-type") ?? "application/json", "Cache-Control": "no-store" },
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
  if (body.pages.length > CAPTURE.MAX_PAGES) return failure(`We can take up to ${CAPTURE.MAX_PAGES} pages in one paper.`);
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


async function paperRetry(req: Request, env: Env): Promise<Response> {
  const user = clientFor(req, env);
  if (!user) return failure("Sign in first.", 401);
  const body = await readJson<any>(req);
  if (!body?.paper_id) return failure("Which paper?", 400);

  const { data: paper, error: paperError } = await user
    .from("paper")
    .select("id,student_id,type,tier,date_taken,subject,reported_total,stated_maximum")
    .eq("id", body.paper_id)
    .maybeSingle();
  if (paperError) return failure("We could not check that paper.", 500, paperError.message);
  if (!paper) return failure("That paper is not yours.", 403);

  const { data: latest, error: runError } = await user
    .from("extraction_run")
    .select("id,status,started_at")
    .eq("paper_id", paper.id)
    .order("started_at", { ascending: false })
    .limit(1)
    .maybeSingle();
  if (runError) return failure("We could not check that paper's reading status.", 500, runError.message);
  if (!latest) return json({ retry: "not_retryable", reason: "no_failed_run" }, 409);
  if (latest.status === "rejected") {
    return json({ retry: "not_retryable", reason: "rejected" }, 409);
  }
  if (latest.status !== "failed") {
    if (latest.status === "queued" && env.TRIAGE_QUEUE) {
      try { await env.TRIAGE_QUEUE.send({ run_id: latest.id }); } catch { /* durable queued row remains recoverable */ }
    }
    const active = !["committed", "rejected", "failed"].includes(latest.status);
    return json({
      retry: active ? "already_in_progress" : "not_retryable",
      reason: latest.status,
      run_id: latest.id,
      status: latest.status,
    }, active ? 202 : 409);
  }

  const { data: pages, error: pageError } = await user
    .from("paper_page")
    .select("page_number,source_kind,r2_bucket,r2_key,mask_key,original_key,thumb_key,bytes,sha256,etag,preprocess_version,quality_verdict,quality_signals,conditioning_meta,layer_fallback,teacher_marks")
    .eq("paper_id", paper.id)
    .eq("student_id", paper.student_id)
    .order("page_number", { ascending: true });
  if (pageError) return failure("We could not load the stored pages for that paper.", 500, pageError.message);

  const storedPages = (pages ?? []).filter((page: any) => page.r2_key);
  if (!storedPages.length) {
    return json({ retry: "not_retryable", reason: "stored_pages_unavailable" }, 409);
  }

  // A retry must use the exact already-stored paper, never client-supplied object
  // keys. Verify every page still exists before starting a fresh run.
  for (const page of storedPages as any[]) {
    const bucket: BucketKind = page.r2_bucket === "originals" ? "originals" : "derived";
    if (!page.r2_key.startsWith(`${paper.student_id}/${paper.id}/`)) {
      return json({ retry: "not_retryable", reason: "stored_pages_unavailable" }, 409);
    }
    let head: Awaited<ReturnType<typeof headObject>>;
    try {
      head = await headObject(env, bucket, page.r2_key);
    } catch {
      return json({ retry: "temporarily_unavailable", reason: "storage_check_failed" }, 503);
    }
    if (!head) return json({ retry: "not_retryable", reason: "stored_pages_unavailable" }, 409);
  }

  const { data, error } = await user.rpc("submit_paper", {
    p_student_id: paper.student_id,
    p_type: paper.type,
    p_tier: paper.tier ?? "tier_1",
    p_date_taken: paper.date_taken ?? null,
    p_subject: paper.subject ?? null,
    p_pages: storedPages,
    // p_paper_id is the identity. This key only satisfies the shared submit
    // contract and is never trusted to choose another paper.
    p_idempotency_key: crypto.randomUUID(),
    p_reported_total: paper.reported_total ?? null,
    p_stated_maximum: paper.stated_maximum ?? null,
    p_pipeline_version: PIPELINE_VERSION,
    p_paper_id: paper.id,
  });
  if (error) return failure("We could not restart that paper. Nothing was lost — try again.", 500, error.message);

  const runId = data.run_id;
  const admin = serviceClient(env);
  const { data: run } = await admin.from("extraction_run").select("status").eq("id", runId).single();
  let queued = false;
  if (run?.status === "queued" && env.TRIAGE_QUEUE) {
    try {
      await env.TRIAGE_QUEUE.send({ run_id: runId });
      await admin.rpc("run_advance", { p_run_id: runId, p_to: "queued" });
      queued = true;
    } catch {
      queued = false;
    }
  }

  return json({
    paper_id: paper.id,
    run_id: runId,
    retry: data.run_created ? "started" : "already_in_progress",
    queued,
    ...(queued ? {} : { reason: "Reading is queued and can be resumed safely." }),
  }, 202);
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

  type PreparedUpload = {
    object: any;
    objectName: string;
    extension: string;
    bucket: BucketKind;
    key: string;
  };

  // Validate every object before any database side effect or signed URL work.
  // A bad object late in the request therefore cannot leave partial ledger rows.
  const prepared: PreparedUpload[] = [];
  for (const object of body.objects) {
    const extension = CAPTURE.UPLOAD_EXTENSIONS[object.content_type as keyof typeof CAPTURE.UPLOAD_EXTENSIONS];
    if (!extension) return failure(`We cannot take a ${object.content_type} file.`);
    if (object.bytes && object.bytes > MAX_BYTES) return failure("One of those files is too large to upload.");
    const bucket = BUCKET_FOR[object.kind as keyof typeof BUCKET_FOR] as BucketKind | undefined;
    if (!bucket) return failure("Unknown file kind.");

    // This value becomes one path segment in the signed R2 key. Reject separators,
    // dot-segments and every other path-like spelling before any URL is minted.
    const objectName = typeof object.name === "number" ? String(object.name) : object.name;
    if (typeof objectName !== "string" || !SAFE_OBJECT_NAME.test(objectName)) {
      return failure("One of those files has an invalid upload name.");
    }

    const key = objectKey({
      studentId: body.student_id,
      paperId: body.paper_id,
      kind: object.kind,
      name: objectName,
      extension,
    });
    prepared.push({ object, objectName, extension, bucket, key });
  }

  let signedUrls: string[];
  try {
    signedUrls = await mapLimit(prepared, IO_CONCURRENCY, ({ object, bucket, key }) =>
      presignPut(env, bucket, key, object.content_type)
    );
  } catch (cause) {
    return failure("We could not prepare those files for upload. Nothing was uploaded yet.", 503, String(cause));
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
    if (error || (rows?.length ?? 0) !== ledgerRows.length) {
      return failure(
        "We could not prepare those files for upload. Nothing was uploaded yet.",
        500,
        error?.message ?? "Upload ledger did not return every requested row.",
      );
    }
    for (const row of rows ?? []) ledgerByKey.set(row.r2_key, row.id);
  }

  const minted = prepared.map(({ object, objectName, bucket, key }, index) => ({
    kind: object.kind,
    name: objectName,
    bucket,
    key,
    url: signedUrls[index],
    ...(ledgerByKey.has(key) ? { upload_id: ledgerByKey.get(key) } : {}),
  }));
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

  const missing: Array<{ key: string; reason: string }> = [];
  const validClaims: any[] = [];
  for (const claim of body.uploads) {
    // JSON is untyped at runtime. Do not let an arbitrary string choose a binding.
    if (claim.bucket !== "originals" && claim.bucket !== "derived") {
      missing.push({ key: claim.key ?? "", reason: "that storage bucket is not valid" });
      continue;
    }
    if (!claim.key?.startsWith(`${paper.student_id}/${paper.id}/`)) {
      missing.push({ key: claim.key ?? "", reason: "that file does not belong to this paper" });
      continue;
    }
    validClaims.push(claim);
  }

  type CheckedUpload = {
    claim: any;
    head: NonNullable<Awaited<ReturnType<typeof headObject>>> | null;
    reason?: string;
  };
  const checked = await mapLimit<any, CheckedUpload>(validClaims, IO_CONCURRENCY, async (claim) => {
    try {
      const head = await headObject(env, claim.bucket, claim.key);
      if (!head) return { claim, head: null, reason: "that file did not arrive" };
      if (claim.bytes && claim.bytes !== head.bytes) {
        return { claim, head, reason: "that file arrived incomplete" };
      }
      return { claim, head };
    } catch (cause) {
      return { claim, head: null, reason: `we could not check that file (${cause})` };
    }
  });

  for (const item of checked) {
    if (item.reason) missing.push({ key: item.claim.key, reason: item.reason });
  }
  const confirmedCandidates = checked.filter(
    (item): item is CheckedUpload & { head: NonNullable<CheckedUpload["head"]> } => !item.reason && item.head !== null,
  );

  const admin = serviceClient(env);
  try {
    await mapLimit(confirmedCandidates, IO_CONCURRENCY, async ({ claim, head }) => {
      const { error } = await admin
        .from("upload")
        .update({
          confirmed: true,
          bytes: head.bytes,
          etag: head.etag,
          // This is retained only as client telemetry. The column name makes it
          // explicit that no integrity decision may rely on it.
          client_reported_sha256: typeof claim.sha256 === "string" ? claim.sha256 : null,
        })
        .eq("paper_id", body.paper_id)
        .eq("r2_key", claim.key);
      if (error) throw new Error(error.message ?? "upload confirmation update failed");
    });
  } catch (cause) {
    return failure("We could not confirm those uploaded files. Try again.", 500, String(cause));
  }

  const confirmed = confirmedCandidates.map(({ claim }) => claim.key);
  return json({ confirmed, missing }, missing.length ? 409 : 200);
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

  // Sign back to the exact API origin the browser reached. This makes the asset
  // URL immune to a missing/stale MASTERY_ASSET_URL secret and guarantees that
  // /page-asset-urls cannot hand the frontend a URL for a different Worker.
  const assetOrigin = new URL(req.url).origin;
  const signed = await mapLimit(pages ?? [], PAGE_PAIR_CONCURRENCY, async (page) => {
    const bucket = (page.r2_bucket as BucketKind) ?? "derived";
    // Each page's independent page/mask pair can be signed together, while the
    // outer page concurrency is bounded so a large booklet cannot create a burst.
    const [url, maskUrl] = await Promise.all([
      page.r2_key ? signAssetUrl(env, bucket, page.r2_key, undefined, assetOrigin) : Promise.resolve(null),
      page.mask_key ? signAssetUrl(env, bucket, page.mask_key, undefined, assetOrigin) : Promise.resolve(null),
    ]);
    return [page.page_number, { url, mask_url: maskUrl }] as const;
  });
  return json({ urls: Object.fromEntries(signed) });
}

// This is the ONLY place that starts explanations: it gates on every
// review-required region being confirmed, then calls begin_explanations and
// fans out to EXPLAIN_QUEUE. See AXON_FIX_BRIEF.md §4.A1 — the frontend bug
// is calling this before review is complete, when it is guaranteed to 409.
// This endpoint itself is not the bug; it needs to be called at the right
// time, which is a frontend fix (§6.1), not a change here.
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
    // ⚡ Bolt: Optimize queue dispatch with batched messages to avoid N+1 latency bottleneck
    await chunkedSendBatch(
      env.EXPLAIN_QUEUE,
      regionIds,
      (regionId) => ({ body: { run_id: body.run_id, region_id: regionId } }),
    );
  }
  return json({ run_id: body.run_id, explaining: begin?.queued ?? 0 });
}

// AXO-124: the student's "try again" on a question whose explanation failed. Ownership is checked
// through the user's own RLS-scoped client; the re-queue itself is a service-role RPC that only
// touches failed, student-confirmed, mark-losing regions, so a repeat tap is harmless.
async function explainRetry(req: Request, env: Env): Promise<Response> {
  const user = clientFor(req, env);
  if (!user) return failure("Sign in first.", 401);
  const body = await readJson<any>(req);
  if (!body?.run_id) return failure("Which paper?");

  const { data: run } = await user.from("extraction_run").select("id").eq("id", body.run_id).maybeSingle();
  if (!run) return failure("That paper is not yours.", 403);

  const admin = serviceClient(env);
  const { data, error } = await admin.rpc("retry_failed_explanations", { p_run_id: body.run_id });
  if (error) return failure("We could not retry the explanations. Nothing was changed.", 500, error.message);

  const regionIds: string[] = data?.region_ids ?? [];
  if (env.EXPLAIN_QUEUE && regionIds.length) {
    await chunkedSendBatch(
      env.EXPLAIN_QUEUE,
      regionIds,
      (regionId) => ({ body: { run_id: body.run_id, region_id: regionId } }),
    );
  }
  return json({ run_id: body.run_id, retrying: data?.queued ?? 0 });
}
