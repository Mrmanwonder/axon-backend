import { pseudonymizeIdentifier } from "../../intelligence/security/privacy";
import { readBoundedBytes } from "../../shared/bounded-bytes";

export interface PaperIngestMetadata { paperId: string; pageId: string; studentId: string; originalHash: string; perceptualHash?: string; sourceType: string; timestamp: string; objectKey: string; pageIndex: number }
export interface PaperJob { type: "PROCESS_PAGE"; metadata: PaperIngestMetadata }

function hex(buffer: ArrayBuffer): string { return [...new Uint8Array(buffer)].map((value) => value.toString(16).padStart(2, "0")).join(""); }

export function detectDocumentType(buffer: ArrayBuffer): "image/jpeg" | "image/png" | "image/webp" | "image/heic" | "application/pdf" | null {
  const bytes = new Uint8Array(buffer);
  const ascii = (start: number, length: number): string => String.fromCharCode(...bytes.slice(start, start + length));
  if (bytes[0] === 0xff && bytes[1] === 0xd8 && bytes[2] === 0xff) return "image/jpeg";
  if (bytes.length >= 8 && bytes[0] === 0x89 && ascii(1, 3) === "PNG" && bytes[4] === 0x0d && bytes[5] === 0x0a && bytes[6] === 0x1a && bytes[7] === 0x0a) return "image/png";
  if (bytes.length >= 12 && ascii(0, 4) === "RIFF" && ascii(8, 4) === "WEBP") return "image/webp";
  if (bytes.length >= 12 && ascii(4, 4) === "ftyp" && /^(?:heic|heix|hevc|hevx|mif1|msf1)$/.test(ascii(8, 4))) return "image/heic";
  if (bytes.length >= 5 && ascii(0, 5) === "%PDF-") return "application/pdf";
  return null;
}
export async function ingestPaperPage(request: Request, env: Env): Promise<{ metadata: PaperIngestMetadata; duplicate: boolean }> {
  const sourceStudentId = request.headers.get("x-axon-student-id");
  if (!sourceStudentId) throw new Error("Missing x-axon-student-id");
  if (sourceStudentId.length > 256) throw new Error("x-axon-student-id is too large");
  const studentId = await pseudonymizeIdentifier(sourceStudentId, env.AXON_PSEUDONYM_KEY);
  const contentLength = Number(request.headers.get("content-length") ?? "0");
  if (!Number.isFinite(contentLength) || contentLength < 0 || contentLength > 20_000_000) throw new Error("Paper page must be between 1 byte and 20 MB");
  const bytes = await readBoundedBytes(request.body, 20_000_000);
  const detectedType = detectDocumentType(bytes);
  if (!detectedType) throw new Error("Unsupported or invalid paper document format");
  const declaredType = request.headers.get("content-type")?.split(";", 1)[0]?.trim().toLowerCase();
  if (declaredType && declaredType !== "application/octet-stream" && declaredType !== detectedType) throw new Error("Paper document content type does not match its bytes");
  const originalHash = hex(await crypto.subtle.digest("SHA-256", bytes));
  const rawPageIndex = request.headers.get("x-axon-page-index") ?? "0";
  const pageIndex = Number(rawPageIndex);
  if (!Number.isInteger(pageIndex) || pageIndex < 0 || pageIndex > 10_000) throw new Error("Invalid x-axon-page-index");
  const existing = await env.DB.prepare("SELECT paper_id, page_id, object_key, page_index FROM paper_page WHERE original_hash = ? AND student_id = ?").bind(originalHash, studentId).first<{ paper_id: string; page_id: string; object_key: string; page_index: number }>();
  if (existing) {
    return { duplicate: true, metadata: { paperId: existing.paper_id, pageId: existing.page_id, studentId, originalHash, sourceType: detectedType, timestamp: new Date().toISOString(), objectKey: existing.object_key, pageIndex: existing.page_index } };
  }
  const requestedPaperId = request.headers.get("x-axon-paper-id");
  if (requestedPaperId && !/^[A-Za-z0-9_-]{1,128}$/.test(requestedPaperId)) throw new Error("Invalid x-axon-paper-id");
  const paperId = requestedPaperId ?? crypto.randomUUID();
  const pageId = crypto.randomUUID();
  const timestamp = new Date().toISOString();
  const sourceType = detectedType;
  const objectKey = `papers/${paperId}/original/${pageId}`;
  await env.PAPER_ARTIFACTS.put(objectKey, bytes, { httpMetadata: { contentType: sourceType }, customMetadata: { originalHash, studentId } });
  const metadata: PaperIngestMetadata = { paperId, pageId, studentId, originalHash, sourceType, timestamp, objectKey, pageIndex };
  await env.DB.prepare("INSERT INTO paper_page (paper_id, page_id, student_id, original_hash, source_type, created_at, object_key, processing_state, page_index) VALUES (?, ?, ?, ?, ?, ?, ?, 'QUEUED', ?)")
    .bind(paperId, pageId, studentId, originalHash, sourceType, timestamp, objectKey, pageIndex).run();
  await env.PAPER_QUEUE.send({ type: "PROCESS_PAGE", metadata } satisfies PaperJob);
  return { metadata, duplicate: false };
}

export async function processPaperBatch(batch: MessageBatch<PaperJob>, env: Env): Promise<void> {
  const { HttpDocumentVisionProvider } = await import("../vision/provider");
  const { markPageForReview, processPaperPage } = await import("../orchestrator");
  for (const message of batch.messages) {
    try {
      const { metadata } = message.body;
      const object = await env.PAPER_ARTIFACTS.head(metadata.objectKey);
      if (!object) throw new Error("Original paper artifact missing");
      if (!env.AXON_VISION_API_BASE || !env.AXON_VISION_TOKEN || String(env.AXON_VISION_PRIVACY_MODE) !== "zdr") {
        await markPageForReview(env.DB, metadata, "NO_PRIVACY_COMPLIANT_DOCUMENT_PROVIDER");
      } else {
        const provider = new HttpDocumentVisionProvider(env.AXON_VISION_API_BASE, env.AXON_VISION_TOKEN, "zdr");
        await processPaperPage(env, metadata, provider);
      }
      message.ack();
    } catch (error) {
      console.error(JSON.stringify({ event: "paper_job_failed", messageId: message.id, error: error instanceof Error ? error.message : String(error) }));
      if (message.attempts >= 3) {
        await markPageForReview(env.DB, message.body.metadata, "DOCUMENT_PIPELINE_FAILURE");
        message.ack();
      } else message.retry();
    }
  }
}
