/** JSON metadata only: images use direct signed R2 uploads. */
export const MAX_JSON_BYTES = 256 * 1024;
export const JSON_BODY_TIMEOUT_MS = 10_000;
export class JsonBodyError extends Error {
  constructor(public readonly status: 408 | 413 | 415, message: string) { super(message); }
}

/** Count actual streamed bytes, even when Content-Length is absent or forged.
 * Keep malformed JSON's existing null contract; transport refusals are typed 4xx.
 */
export async function readBoundedJson<T = unknown>(req: Request, maxBytes = MAX_JSON_BYTES, timeoutMs = JSON_BODY_TIMEOUT_MS): Promise<T | null> {
  const media = req.headers.get("content-type")?.split(";", 1)[0].trim().toLowerCase();
  if (media !== "application/json" && !/^application\/[\w.-]+\+json$/.test(media ?? "")) {
    void req.body?.cancel().catch(() => {});
    throw new JsonBodyError(415, "Expected application/json.");
  }
  if (Number(req.headers.get("content-length")) > maxBytes) {
    void req.body?.cancel().catch(() => {});
    throw new JsonBodyError(413, "JSON body is too large.");
  }
  if (!req.body) return null;
  const reader = req.body.getReader();
  let timer: ReturnType<typeof setTimeout> | undefined;
  let expired = false;
  const deadline = new Promise<never>((_, reject) => {
    timer = setTimeout(() => {
      expired = true;
      reject(new JsonBodyError(408, "JSON body read timed out."));
      void reader.cancel().catch(() => {});
    }, timeoutMs);
  });
  try {
    return await Promise.race([(async () => {
    const decoder = new TextDecoder("utf-8", { fatal: true, ignoreBOM: false });
    let bytes = 0;
    let text = "";
    while (true) {
      const { done, value } = await reader.read();
      if (expired) throw new JsonBodyError(408, "JSON body read timed out.");
      if (done) break;
      bytes += value.byteLength;
      if (bytes > maxBytes) throw new JsonBodyError(413, "JSON body is too large.");
      text += decoder.decode(value, { stream: true });
    }
    text += decoder.decode();
    return JSON.parse(text) as T;
    })(), deadline]);
  } catch (error) {
    void reader.cancel().catch(() => {});
    if (error instanceof JsonBodyError) throw error;
    return null;
  } finally {
    if (timer !== undefined) clearTimeout(timer);
    reader.releaseLock();
  }
}
