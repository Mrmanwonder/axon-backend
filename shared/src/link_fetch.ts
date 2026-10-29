/**
 * Fetch a paper the student pasted as a link (AXO-295).
 *
 * A browser cannot read a cross-origin PDF, so `mastery-api` fetches it and hands the bytes back.
 * The device then imports them exactly as it would a file the student chose, so conditioning and
 * PDF rasterisation stay on the device (SCANNING_SYSTEM.md stages 0–2) and nothing downstream
 * changes. This module is the only thing between a student-supplied URL and the network, so it
 * refuses more than it needs to:
 *
 *   - https only, default port, no credentials in the URL;
 *   - no IP literals of any spelling and no local or internal hostnames. A real school link names
 *     a host. Cloudflare's own fetch also cannot reach private networks, so a public name that
 *     resolves to a private address does not open one either;
 *   - redirects followed by hand, at most three, every hop checked again;
 *   - a hard byte cap, checked against Content-Length and again while reading;
 *   - the file type decided from its first bytes, never from the server's header, and only the
 *     types the pipeline contract accepts.
 *
 * Google Drive and Docs share links are rewritten to their direct download, since those are what
 * schools share. A link that needs a sign-in comes back as an HTML page and is refused with that
 * reason rather than as a broken file.
 */

export const LINK_MAX_BYTES = 25 * 1024 * 1024;
export const LINK_MAX_REDIRECTS = 3;
export const LINK_TIMEOUT_MS = 20_000;

export type LinkedFileType = "application/pdf" | "image/png" | "image/jpeg" | "image/webp" | "image/heic";

export class LinkRefused extends Error {
  constructor(public readonly code: string, message: string, public readonly status = 422) {
    super(message);
    this.name = "LinkRefused";
  }
}

const LOCAL_SUFFIXES = [".localhost", ".local", ".internal", ".lan", ".home", ".corp", ".intranet", ".arpa", ".test", ".invalid", ".example"];

/** True for any hostname that spells an IP address, in any base or form the URL parser accepts. */
function isIpLiteral(hostname: string): boolean {
  if (hostname.startsWith("[") || hostname.includes(":")) return true; // IPv6
  // WHATWG URL normalises decimal, octal and hex IPv4 forms to dotted decimal before we see them.
  return /^\d{1,3}(\.\d{1,3}){3}$/.test(hostname);
}

/**
 * Check a URL against the rules above. Returns the URL to fetch (after any Drive rewrite).
 * @throws LinkRefused
 */
export function checkLinkUrl(raw: string): URL {
  let url: URL;
  try {
    url = new URL(raw.trim());
  } catch {
    throw new LinkRefused("link_invalid", "That does not look like a link. Paste the full address, including https://");
  }
  if (url.protocol !== "https:") throw new LinkRefused("link_not_https", "Only secure (https) links can be fetched.");
  if (url.username || url.password) throw new LinkRefused("link_credentials", "Links with a username or password in them cannot be fetched.");
  if (url.port && url.port !== "443") throw new LinkRefused("link_port", "That link uses an unusual port and cannot be fetched.");
  const host = url.hostname.toLowerCase().replace(/\.$/, "");
  if (!host || isIpLiteral(host)) throw new LinkRefused("link_ip_literal", "That link points at an address rather than a website, so it cannot be fetched.");
  if (!host.includes(".") || host === "localhost" || LOCAL_SUFFIXES.some((s) => host.endsWith(s))) {
    throw new LinkRefused("link_local_host", "That link is not on the public internet, so it cannot be fetched.");
  }
  return rewriteShareLink(url);
}

/** Google Drive and Docs share pages are HTML; their direct-download forms are the file. */
export function rewriteShareLink(url: URL): URL {
  const host = url.hostname.toLowerCase();
  if (host === "drive.google.com") {
    const fileMatch = url.pathname.match(/^\/file\/d\/([A-Za-z0-9_-]{10,})/);
    const id = fileMatch?.[1] ?? (url.pathname === "/open" ? url.searchParams.get("id") : null);
    if (id && /^[A-Za-z0-9_-]{10,}$/.test(id)) {
      return new URL(`https://drive.google.com/uc?export=download&id=${id}`);
    }
  }
  if (host === "docs.google.com") {
    const docMatch = url.pathname.match(/^\/(document|presentation|spreadsheets)\/d\/([A-Za-z0-9_-]{10,})/);
    if (docMatch) {
      const [, kind, id] = docMatch;
      return new URL(`https://docs.google.com/${kind}/d/${id}/export?format=pdf`);
    }
  }
  return url;
}

/** The file type, from its first bytes. Null when it is none Axon can read. */
export function sniffType(head: Uint8Array): LinkedFileType | null {
  const ascii = (from: number, to: number) => String.fromCharCode(...head.subarray(from, to));
  if (head.length >= 5 && ascii(0, 5) === "%PDF-") return "application/pdf";
  if (head.length >= 8 && head[0] === 0x89 && ascii(1, 4) === "PNG" && head[4] === 0x0d && head[5] === 0x0a) return "image/png";
  if (head.length >= 3 && head[0] === 0xff && head[1] === 0xd8 && head[2] === 0xff) return "image/jpeg";
  if (head.length >= 12 && ascii(0, 4) === "RIFF" && ascii(8, 12) === "WEBP") return "image/webp";
  if (head.length >= 12 && ascii(4, 8) === "ftyp" && ["heic", "heix", "heim", "heis", "hevc", "mif1", "msf1"].includes(ascii(8, 12))) return "image/heic";
  return null;
}

function looksLikeHtml(head: Uint8Array): boolean {
  const text = new TextDecoder().decode(head.subarray(0, 512)).trimStart().toLowerCase();
  return text.startsWith("<!doctype html") || text.startsWith("<html") || text.startsWith("<head") || text.startsWith("<");
}

const EXTENSION: Record<LinkedFileType, string> = {
  "application/pdf": "pdf", "image/png": "png", "image/jpeg": "jpg", "image/webp": "webp", "image/heic": "heic",
};

/** A safe display name for the file: from Content-Disposition, else the URL path, else a default. */
export function fileNameFor(url: URL, disposition: string | null, type: LinkedFileType): string {
  let name: string | null = null;
  const star = disposition?.match(/filename\*\s*=\s*(?:UTF-8'')?([^;]+)/i)?.[1];
  const plain = disposition?.match(/filename\s*=\s*"?([^";]+)"?/i)?.[1];
  try { name = star ? decodeURIComponent(star.trim().replace(/^"|"$/g, "")) : plain?.trim() ?? null; } catch { name = plain?.trim() ?? null; }
  if (!name) {
    const last = url.pathname.split("/").filter(Boolean).pop();
    try { name = last ? decodeURIComponent(last) : null; } catch { name = last ?? null; }
  }
  const cleaned = (name ?? "").replace(/[\u0000-\u001f\u007f/\\]+/g, " ").replace(/\s+/g, " ").trim().slice(0, 120);
  const ext = EXTENSION[type];
  if (!cleaned || cleaned === "uc" || cleaned === "export") return `Linked paper.${ext}`;
  return new RegExp(`\\.${ext}$`, "i").test(cleaned) ? cleaned : `${cleaned}.${ext}`;
}

export interface LinkedFile {
  bytes: Uint8Array;
  type: LinkedFileType;
  name: string;
  finalUrl: string;
}

async function readCapped(body: ReadableStream<Uint8Array> | null, maxBytes: number): Promise<Uint8Array> {
  if (!body) return new Uint8Array();
  const reader = body.getReader();
  const chunks: Uint8Array[] = [];
  let total = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    total += value.byteLength;
    if (total > maxBytes) {
      await reader.cancel().catch(() => {});
      throw new LinkRefused("link_too_large", `That file is larger than ${Math.round(maxBytes / 1024 / 1024)} MB. Download it and add it in parts.`, 413);
    }
    chunks.push(value);
  }
  const out = new Uint8Array(total);
  let at = 0;
  for (const chunk of chunks) { out.set(chunk, at); at += chunk.byteLength; }
  return out;
}

/**
 * Fetch the linked file under every rule above.
 * @throws LinkRefused for anything refused; network errors are rethrown as LinkRefused too.
 */
export async function fetchLinkedFile(
  raw: string,
  {
    fetchImpl = fetch,
    maxBytes = LINK_MAX_BYTES,
    timeoutMs = LINK_TIMEOUT_MS,
  }: { fetchImpl?: typeof fetch; maxBytes?: number; timeoutMs?: number } = {},
): Promise<LinkedFile> {
  let url = checkLinkUrl(raw);
  const signal = AbortSignal.timeout(timeoutMs);
  for (let hop = 0; ; hop++) {
    let response: Response;
    try {
      response = await fetchImpl(url.toString(), {
        method: "GET",
        redirect: "manual",
        signal,
        headers: { Accept: "application/pdf,image/*;q=0.9,*/*;q=0.1", "User-Agent": "AxonLinkFetcher/1 (+https://axonstudy.online)" },
      });
    } catch (cause) {
      const timedOut = (cause as Error)?.name === "TimeoutError" || (cause as Error)?.name === "AbortError";
      throw new LinkRefused(timedOut ? "link_timeout" : "link_unreachable",
        timedOut ? "That link took too long to answer. Try again, or download the file and add it." : "That link could not be reached.", 502);
    }
    if (response.status >= 300 && response.status < 400) {
      const location = response.headers.get("location");
      if (!location) throw new LinkRefused("link_bad_redirect", "That link redirects nowhere.", 502);
      if (hop >= LINK_MAX_REDIRECTS) throw new LinkRefused("link_too_many_redirects", "That link redirects too many times.", 502);
      url = checkLinkUrl(new URL(location, url).toString());
      continue;
    }
    if (response.status === 401 || response.status === 403) {
      throw new LinkRefused("link_needs_sign_in", "That link needs a sign-in. Open it, download the file, and add it from your files.", 422);
    }
    if (!response.ok) throw new LinkRefused("link_http_error", `That link answered with an error (${response.status}).`, 502);
    const declared = Number(response.headers.get("content-length"));
    if (Number.isFinite(declared) && declared > maxBytes) {
      await response.body?.cancel().catch(() => {});
      throw new LinkRefused("link_too_large", `That file is larger than ${Math.round(maxBytes / 1024 / 1024)} MB. Download it and add it in parts.`, 413);
    }
    const bytes = await readCapped(response.body, maxBytes);
    if (!bytes.byteLength) throw new LinkRefused("link_empty", "That link returned an empty file.", 422);
    const type = sniffType(bytes.subarray(0, 32));
    if (!type) {
      throw looksLikeHtml(bytes)
        ? new LinkRefused("link_is_a_page", "That link opens a web page, not a file. If it needs a sign-in or is not shared publicly, download the file and add it instead.")
        : new LinkRefused("link_unsupported_type", "That link is not a PDF or a photo, so it cannot be read.");
    }
    return { bytes, type, name: fileNameFor(url, response.headers.get("content-disposition"), type), finalUrl: url.toString() };
  }
}
