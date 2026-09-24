import { evidenceConfidence } from "../intelligence/confidence";
import { recordProviderObservation, writeTrace } from "../intelligence/telemetry/repository";
import { classifyInk } from "./ink";
import type { PaperIngestMetadata } from "./ingest";
import type { LayoutRegion } from "./layout/types";
import { matchMarks, type QuestionCandidate } from "./mark-matcher";
import { assessPageQuality, conditioningPlan } from "./quality";
import { buildQuestionGraph } from "./question-graph";
import { reconcileReads, type RegionRead } from "./recognition";
import type { DocumentVisionProvider, VisionAnalysis } from "./vision/provider";
import { reconcilePaperTopology } from "./reconcile-paper";

const encoder = new TextEncoder();
function hex(buffer: ArrayBuffer): string { return [...new Uint8Array(buffer)].map((value) => value.toString(16).padStart(2, "0")).join(""); }
async function digest(value: ArrayBuffer | string): Promise<string> {
  return hex(await crypto.subtle.digest("SHA-256", typeof value === "string" ? encoder.encode(value) : value));
}

async function event(db: D1Database, pageId: string, stage: string, status: "STARTED" | "COMPLETED" | "FAILED" | "REVIEW_REQUIRED", inputHash: string, details: unknown, outputHash?: string): Promise<void> {
  await db.prepare("INSERT INTO paper_stage_event (id, page_id, stage, status, input_hash, output_hash, details_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)")
    .bind(crypto.randomUUID(), pageId, stage, status, inputHash, outputHash ?? null, JSON.stringify(details), new Date().toISOString()).run();
}

function unionBox(regions: readonly LayoutRegion[]): LayoutRegion["box"] {
  const x = Math.min(...regions.map((item) => item.box.x));
  const y = Math.min(...regions.map((item) => item.box.y));
  const right = Math.max(...regions.map((item) => item.box.x + item.box.width));
  const bottom = Math.max(...regions.map((item) => item.box.y + item.box.height));
  return { x, y, width: right - x, height: bottom - y };
}

function base64Bytes(value: string): ArrayBuffer {
  const binary = atob(value);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index);
  return bytes.buffer;
}

export async function markPageForReview(db: D1Database, metadata: PaperIngestMetadata, reason: string): Promise<void> {
  const now = new Date().toISOString();
  await db.prepare("UPDATE paper_page SET processing_state = 'REVIEW_REQUIRED', review_reasons_json = ?, pipeline_version = ?, updated_at = ? WHERE page_id = ?")
    .bind(JSON.stringify([reason]), "3.0.0", now, metadata.pageId).run();
  await event(db, metadata.pageId, "VISION_ANALYSIS", "REVIEW_REQUIRED", metadata.originalHash, { reason });
}

export async function processPaperPage(env: Env, metadata: PaperIngestMetadata, provider: DocumentVisionProvider): Promise<void> {
  if (provider.privacyMode !== "zdr") {
    await markPageForReview(env.DB, metadata, "NO_PRIVACY_COMPLIANT_DOCUMENT_PROVIDER");
    return;
  }
  const object = await env.PAPER_ARTIFACTS.get(metadata.objectKey);
  if (!object) throw new Error("Original paper artifact missing");
  const bytes = await object.arrayBuffer();
  await event(env.DB, metadata.pageId, "VISION_ANALYSIS", "STARTED", metadata.originalHash, { provider: provider.id });
  const started = Date.now();
  let result: Awaited<ReturnType<DocumentVisionProvider["analyze"]>>;
  try {
    result = await provider.analyze({ bytes, mimeType: metadata.sourceType, pageId: metadata.pageId, timeoutMs: 20_000 });
    await recordProviderObservation(env.DB, provider.id, "document-vision-v1", { success: true, latencyMs: result.latencyMs });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    await recordProviderObservation(env.DB, provider.id, "document-vision-v1", {
      success: false, latencyMs: Date.now() - started, timeout: /abort|timeout/i.test(message),
      rateLimited: /429/.test(message), serverError: /VISION_PROVIDER_5\d\d/.test(message), schemaFailure: /schema validation/i.test(message)
    });
    throw error;
  }
  const { analysis, latencyMs } = result;
  const outputHash = await digest(JSON.stringify(analysis));
  await event(env.DB, metadata.pageId, "VISION_ANALYSIS", "COMPLETED", metadata.originalHash, { provider: provider.id, latencyMs }, outputHash);
  await persistAnalysis(env, metadata, analysis, outputHash);
}

async function persistAnalysis(env: Env, metadata: PaperIngestMetadata, analysis: VisionAnalysis, analysisHash: string): Promise<void> {
  const quality = assessPageQuality(analysis.qualityMetrics);
  const plan = conditioningPlan(analysis.qualityMetrics, analysis.orientationDegrees);
  const reviewReasons: string[] = [];
  let conditionedHash: string | undefined;
  let conditionedObjectKey: string | undefined;
  if (quality.requiredAction === "rescan") reviewReasons.push("PAGE_UNREADABLE_RESCAN_REQUIRED");
  if (quality.requiredAction === "condition" && !analysis.conditionedImageBase64) reviewReasons.push("CONDITIONED_ARTIFACT_REQUIRED");
  if (analysis.conditionedImageBase64) {
    const conditioned = base64Bytes(analysis.conditionedImageBase64);
    conditionedHash = await digest(conditioned);
    conditionedObjectKey = `papers/${metadata.paperId}/conditioned/${metadata.pageId}/${conditionedHash}`;
    await env.PAPER_ARTIFACTS.put(conditionedObjectKey, conditioned, { httpMetadata: { contentType: analysis.conditionedImageMimeType ?? metadata.sourceType }, customMetadata: { originalHash: metadata.originalHash, conditionedHash } });
  }

  const regions: LayoutRegion[] = analysis.regions.map((region) => ({
    id: `${metadata.pageId}:${region.id}`, pageId: metadata.pageId, class: region.class as LayoutRegion["class"], box: region.box,
    confidence: region.confidence, ...(region.text !== undefined ? { text: region.text } : {})
  }));
  if (regions.length === 0) reviewReasons.push("NO_LAYOUT_REGIONS");
  if (regions.some((region) => region.confidence < 0.7)) reviewReasons.push("LOW_LAYOUT_CONFIDENCE");
  const graph = buildQuestionGraph(regions);

  const inkByRegion = new Map<string, ReturnType<typeof classifyInk>>();
  for (const source of analysis.regions) {
    if (source.inkSignals) inkByRegion.set(`${metadata.pageId}:${source.id}`, classifyInk(source.inkSignals));
  }
  const inkSensitiveClasses = new Set<LayoutRegion["class"]>(["student_answer", "teacher_annotation", "teacher_comment", "marginal_mark", "crossed_out_work"]);
  if (regions.some((region) => inkSensitiveClasses.has(region.class) && !inkByRegion.has(region.id))) reviewReasons.push("MISSING_INK_CLASSIFICATION");
  if ([...inkByRegion.values()].some((item) => item.class === "UNKNOWN")) reviewReasons.push("AMBIGUOUS_INK_LAYER");

  const regionById = new Map(regions.map((region) => [region.id, region]));
  const questionCandidates: QuestionCandidate[] = graph.questions.flatMap((question, order) => {
    const owned = question.regionIds.map((id) => regionById.get(id)).filter((item): item is LayoutRegion => Boolean(item));
    if (owned.length === 0) return [];
    return [{ id: question.id, pageIds: question.pageIds, box: unionBox(owned), order, continuationPageIds: question.pageIds, commentRegionIds: owned.filter((item) => item.class === "teacher_comment").map((item) => item.id) }];
  });
  const marks = regions.filter((region) => region.class === "marginal_mark").map((region) => ({ id: region.id, pageId: region.pageId, box: region.box }));
  const assignments = matchMarks(marks, questionCandidates);

  const reconciled = new Map<string, RegionRead>();
  for (const group of analysis.reads) {
    const regionId = `${metadata.pageId}:${group.regionId}`;
    if (!regionById.has(regionId)) { reviewReasons.push("READ_FOR_UNKNOWN_REGION"); continue; }
    const read = reconcileReads(group.reads);
    reconciled.set(regionId, read);
    if (read.status !== "read" || read.readerIds.length < 2) reviewReasons.push("AMBIGUOUS_CONTENT_READ");
  }
  const readableClasses = new Set<LayoutRegion["class"]>(["question_number", "subquestion_number", "printed_question", "student_answer", "teacher_annotation", "teacher_comment", "marginal_mark", "marks_available", "reported_total", "page_number"]);
  if (regions.some((region) => readableClasses.has(region.class) && !reconciled.has(region.id))) reviewReasons.push("MISSING_CONTENT_READ");

  const statements: D1PreparedStatement[] = [];
  for (const region of regions) {
    const ink = inkByRegion.get(region.id);
    statements.push(env.DB.prepare(`INSERT OR REPLACE INTO layout_region
      (id, page_id, class, box_json, confidence, ink_class, text_value, trust_state) VALUES (?, ?, ?, ?, ?, ?, ?, ?)`)
      .bind(region.id, region.pageId, region.class, JSON.stringify(region.box), region.confidence, ink?.class ?? null, region.text ?? null,
        region.confidence >= 0.9 && (!inkSensitiveClasses.has(region.class) || (ink !== undefined && ink.class !== "UNKNOWN")) ? "AUTO_VERIFIED" : "UNVERIFIED"));
  }
  for (const [order, question] of graph.questions.entries()) {
    statements.push(env.DB.prepare("INSERT OR REPLACE INTO question_node (id, paper_id, label, parent_id, page_ids_json, reading_order) VALUES (?, ?, ?, ?, ?, ?)")
      .bind(question.id, metadata.paperId, question.label, question.parentId ?? null, JSON.stringify(question.pageIds), order));
    for (const regionId of question.regionIds) statements.push(env.DB.prepare("INSERT OR REPLACE INTO question_region (question_id, region_id, relationship) VALUES (?, ?, ?)").bind(question.id, regionId, "contains"));
  }
  for (const assignment of assignments) statements.push(env.DB.prepare(`INSERT OR REPLACE INTO mark_assignment
    (mark_region_id, question_id, confidence, second_best_gap, features_json, trust_state) VALUES (?, ?, ?, ?, ?, ?)`)
    .bind(assignment.markId, assignment.questionId, assignment.confidence, assignment.secondBestGap, JSON.stringify(assignment.features), assignment.confidence >= 0.9 && assignment.secondBestGap >= 0.2 ? "AUTO_VERIFIED" : "UNVERIFIED"));
  for (const [regionId, read] of reconciled) {
    const readId = `${regionId}:${analysisHash.slice(0, 16)}`;
    statements.push(env.DB.prepare("INSERT OR REPLACE INTO recognition_read (id, region_id, value_text, alternatives_json, status, reader_ids_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)")
      .bind(readId, regionId, read.value, JSON.stringify(read.alternatives), read.status, JSON.stringify(read.readerIds), new Date().toISOString()));
    const region = regionById.get(regionId)!;
    const confidence = evidenceConfidence({ imageQuality: quality.score, layoutProbability: region.confidence, readerAgreement: read.status === "read" && read.readerIds.length >= 2 ? 1 : 0 });
    const trustState = read.status === "read" && confidence >= 0.9 ? "AUTO_VERIFIED" : "UNVERIFIED";
    statements.push(env.DB.prepare("INSERT OR REPLACE INTO trusted_field (id, artifact_id, field_name, value_json, trust_state, evidence_ids_json, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)")
      .bind(`${regionId}:recognized_text`, regionId, "recognized_text", JSON.stringify(read.value), trustState, JSON.stringify([readId]), new Date().toISOString()));
  }
  if (statements.length > 0) await env.DB.batch(statements);
  const topology = await reconcilePaperTopology(env.DB, metadata.paperId);
  if (topology.graph.questions.length === 0) reviewReasons.push("NO_QUESTION_GRAPH");
  if (topology.assignments.some((item) => item.questionId === null || item.confidence < 0.8 || item.secondBestGap < 0.15)) reviewReasons.push("AMBIGUOUS_MARK_ASSIGNMENT");
  const uniqueReasons = [...new Set(reviewReasons)];
  const finalState = uniqueReasons.length === 0 ? "TRUSTED_COMMIT" : "REVIEW_REQUIRED";
  await env.DB.prepare(`UPDATE paper_page SET processing_state = ?, quality_class = ?, quality_json = ?, review_reasons_json = ?,
    conditioned_hash = ?, conditioned_object_key = ?, pipeline_version = ?, updated_at = ? WHERE page_id = ?`)
    .bind(finalState, quality.classification, JSON.stringify({ metrics: analysis.qualityMetrics, assessment: quality, conditioningPlan: plan }), JSON.stringify(uniqueReasons),
      conditionedHash ?? null, conditionedObjectKey ?? null, "3.0.0", new Date().toISOString(), metadata.pageId).run();
  await event(env.DB, metadata.pageId, "TRUSTED_COMMIT", finalState === "TRUSTED_COMMIT" ? "COMPLETED" : "REVIEW_REQUIRED", analysisHash, { reasons: uniqueReasons, quality: quality.classification });
  await writeTrace(env.DB, {
    traceId: crypto.randomUUID(), paperId: metadata.paperId, stage: "paper", capability: "document_reading", deploymentSha: env.AXON_DEPLOYMENT_SHA,
    configRevision: env.AXON_CONFIG_REVISION, pipelineVersion: env.AXON_PIPELINE_VERSION, provider: "axon-vision",
    verificationStatus: finalState === "TRUSTED_COMMIT" ? "verified" : "review_required", repairAttempted: false,
    toolCalls: ["quality", "layout", "ink", "mark-matcher", "recognition-reconciler"], retrievalUsed: false,
    inputArtifactHashes: [metadata.originalHash]
  });
}
