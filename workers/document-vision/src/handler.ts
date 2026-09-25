import { RequestSchema, parseSchema, type VisionAnalysis, type VisionRequest } from "./schema";

const JSON_HEADERS = { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" };
const MAX_JSON_BYTES = 17_000_000;
const EXPECTED_CONTRACT = "axon-document-vision.v1";
// A small, immutable, synthetic printed page. It contains no student data and
// forces the production probe through layout discovery and a targeted crop read.
const PROBE_IMAGE_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAUAAAAC0AQMAAADfKmdSAAAABlBMVEX///8AAABVwtN+AAAACXBIWXMAAAsTAAALEwEAmpwYAAABxUlEQVRYw+3ZMUvDQBQA4Duu5BRq4lgwJk7OneTA0uYH+CMCDl2FLgUFM8WlmJ/g31AcmlJpl6Kr4wX/QLqlEI13VEKXPCNUcXhve/Al797LHTeEEAwoaFErgh/AoFZdhAgR/ib0aZwkuS2s9+ImKYJqKKlcTj86GkYrAFLJ5GB+1xU0kc6wU12aS5aeRrwn6CS2hYAgv+iZfKyhSwBIFHRNNr6nkyKSMBSOyR41vMlAaAr1xodMQQpDK+/dcg3jnc438DLiQkPbBqFtD+aRGNKJdAQMzeXbh4bFC7xGYckkF2qOxTwVeBQQIkT4vyCVxFun6pkLALKMxGWWAdBoraHHVDaCYLMdK0M8S2VNAHKSv/bjc+LtBtcBBFkyW5z5JyzTA2hB41mFi8WRzXINBQQPw6eX1CYeVaWh8TT2wtmIfEEJdb0XzkdHJxoSGkNd3z9Pb/3+OsvBL3M15Wo8OmM57nCECBEi3IS0LmSVhDfEJuTV0ChhY58QtxIaTqu8FdpuelANuyXkviur12gcl5BJqBnDKKG+hGpCN60FmQTXuAH9mqWNdk1oipoQ3g4G2TbkZIinECFChAgR/gHc+j8kDCg+AfhL0r/YuEyWAAAAAElFTkSuQmCC";

const json = (value: unknown, status = 200): Response => new Response(JSON.stringify(value), { status, headers: JSON_HEADERS });

async function readBoundedJson(request: Request): Promise<unknown> {
  const declared = Number(request.headers.get("content-length") ?? "0");
  if (Number.isFinite(declared) && declared > MAX_JSON_BYTES) throw new Error("REQUEST_TOO_LARGE");
  const reader = request.body?.getReader();
  if (!reader) throw new Error("EMPTY_REQUEST_BODY");
  const chunks: Uint8Array[] = [];
  let total = 0;
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    if (!value) continue;
    total += value.byteLength;
    if (total > MAX_JSON_BYTES) { await reader.cancel(); throw new Error("REQUEST_TOO_LARGE"); }
    chunks.push(value);
  }
  const bytes = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.byteLength; }
  try { return JSON.parse(new TextDecoder().decode(bytes)); }
  catch { throw new Error("INVALID_JSON"); }
}

function safeError(error: unknown): { code: string; status: number } {
  const message = error instanceof Error ? error.message : "UNKNOWN_ERROR";
  if (/REQUEST_TOO_LARGE|IMAGE_PIXEL_LIMIT_EXCEEDED/.test(message)) return { code: message, status: 413 };
  if (/INVALID_|EMPTY_|MIME_TYPE_MISMATCH|IMAGE_SIZE_INVALID|SCHEMA_VALIDATION_FAILED/.test(message)) return { code: message.split(":", 1)[0], status: 400 };
  if (/UNSUPPORTED_IMAGE_CODEC/.test(message)) return { code: message, status: 415 };
  if (/GEMINI_|MOONDREAM_|READER_|TWO_INDEPENDENT/.test(message)) return { code: "VISION_READER_FAILURE", status: 502 };
  return { code: "INTERNAL_ERROR", status: 500 };
}

export function privacyIsReady(env: Env): boolean {
  return Boolean(env.GOOGLE_API_KEY && env.AI) && String(env.GEMINI_PRIVACY_MODE) === "zdr" && String(env.WORKERS_AI_PRIVACY_MODE) === "zdr";
}

export async function handleRequestWith(request: Request, env: Env, analyze: (input: VisionRequest) => Promise<VisionAnalysis>): Promise<Response> {
  const url = new URL(request.url);
  if (request.method === "GET" && url.pathname === "/health") {
    const privacyReady = privacyIsReady(env);
    return json({ status: privacyReady ? "ready" : "not_ready", version: env.AXON_VISION_VERSION, privacyReady }, privacyReady ? 200 : 503);
  }
  if (request.method === "POST" && url.pathname === "/v1/probe") {
    if (request.headers.get("x-axon-contract-version") !== EXPECTED_CONTRACT) return json({ error: "CONTRACT_VERSION_REQUIRED" }, 400);
    if (!privacyIsReady(env)) return json({ error: "PRIVACY_ATTESTATION_REQUIRED" }, 503);
    try {
      const analysis = await analyze({
        contractVersion: EXPECTED_CONTRACT,
        pageId: "synthetic-capability-probe",
        mimeType: "image/png",
        dataBase64: PROBE_IMAGE_BASE64
      });
      const readerIds = new Set(analysis.reads.flatMap((group) => group.reads.flatMap((read) => read.readerIds)));
      const independentlyReadRegion = analysis.reads.some((group) => new Set(group.reads.flatMap((read) => read.readerIds)).size >= 2);
      if (analysis.regions.length === 0 || analysis.reads.length === 0 || !independentlyReadRegion) throw new Error("TWO_INDEPENDENT_READERS_REQUIRED");
      return json({
        status: "passed",
        contractVersion: EXPECTED_CONTRACT,
        version: env.AXON_VISION_VERSION,
        regionCount: analysis.regions.length,
        readGroupCount: analysis.reads.length,
        readerCount: readerIds.size
      });
    } catch (error) {
      const failure = safeError(error);
      console.error(JSON.stringify({ event: "document_vision_probe_failed", code: failure.code }));
      return json({ status: "failed", error: failure.code }, 502);
    }
  }
  if (url.pathname !== "/v1/analyze") return json({ error: "NOT_FOUND" }, 404);
  if (request.method !== "POST") return json({ error: "METHOD_NOT_ALLOWED" }, 405);
  if (request.headers.get("x-axon-contract-version") !== EXPECTED_CONTRACT) return json({ error: "CONTRACT_VERSION_REQUIRED" }, 400);
  if (!privacyIsReady(env)) return json({ error: "PRIVACY_ATTESTATION_REQUIRED" }, 503);
  try {
    return json(await analyze(parseSchema(RequestSchema, await readBoundedJson(request))));
  } catch (error) {
    const failure = safeError(error);
    console.error(JSON.stringify({ event: "document_vision_failed", code: failure.code }));
    return json({ error: failure.code }, failure.status);
  }
}
