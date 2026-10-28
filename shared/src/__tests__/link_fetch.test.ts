import { test } from "node:test";
import assert from "node:assert/strict";
import { checkLinkUrl, fetchLinkedFile, fileNameFor, LinkRefused, rewriteShareLink, sniffType } from "../link_fetch.js";

const PDF = new TextEncoder().encode("%PDF-1.7\n1 0 obj\n");
const PNG = new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0, 0, 0, 13]);
const JPEG = new Uint8Array([0xff, 0xd8, 0xff, 0xe0, 0, 16]);
const WEBP = new TextEncoder().encode("RIFF\x00\x00\x00\x00WEBPVP8 ");
const HEIC = new Uint8Array([0, 0, 0, 24, ...new TextEncoder().encode("ftypheic")]);
const HTML = new TextEncoder().encode("<!DOCTYPE html><html><body>Sign in</body></html>");

const refused = (fn: () => unknown, code: string) =>
  assert.throws(fn, (e: unknown) => e instanceof LinkRefused && e.code === code, `expected ${code}`);
const refusedAsync = (p: Promise<unknown>, code: string) =>
  assert.rejects(p, (e: unknown) => e instanceof LinkRefused && e.code === code, `expected ${code}`);

/** A fetch that answers from a table of URL -> response, recording what was asked. */
function fakeFetch(routes: Record<string, () => Response>) {
  const asked: string[] = [];
  const impl = (async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    asked.push(url);
    assert.equal(init?.redirect, "manual", "redirects must never be followed automatically");
    const route = routes[url];
    if (!route) throw new TypeError(`network: ${url}`);
    return route();
  }) as typeof fetch;
  return { impl, asked };
}
const ok = (bytes: Uint8Array, headers: Record<string, string> = {}) => () => new Response(bytes, { status: 200, headers });
const redirect = (to: string) => () => new Response(null, { status: 302, headers: { location: to } });

test("link checks refuse anything that is not a public https host", () => {
  refused(() => checkLinkUrl("not a link"), "link_invalid");
  refused(() => checkLinkUrl("http://school.example.org/p.pdf"), "link_not_https");
  refused(() => checkLinkUrl("ftp://school.org/p.pdf"), "link_not_https");
  refused(() => checkLinkUrl("https://user:pass@school.org/p.pdf"), "link_credentials");
  refused(() => checkLinkUrl("https://school.org:8443/p.pdf"), "link_port");
  for (const host of ["127.0.0.1", "10.0.0.5", "169.254.169.254", "[::1]", "[fd00::1]", "2130706433", "0x7f000001", "0177.0.0.1"]) {
    refused(() => checkLinkUrl(`https://${host}/p.pdf`), "link_ip_literal");
  }
  for (const host of ["localhost", "intranet", "printer.local", "metadata.internal", "api.localhost", "x.test"]) {
    refused(() => checkLinkUrl(`https://${host}/p.pdf`), "link_local_host");
  }
  assert.equal(checkLinkUrl("https://school.edu.in/papers/maths.pdf").toString(), "https://school.edu.in/papers/maths.pdf");
  assert.equal(checkLinkUrl("https://school.org:443/a.pdf").toString(), "https://school.org/a.pdf");
});

test("Drive and Docs share links become their direct downloads", () => {
  const id = "1AbCdEfGhIjKlMnOp_qrs-tu";
  assert.equal(rewriteShareLink(new URL(`https://drive.google.com/file/d/${id}/view?usp=sharing`)).toString(),
    `https://drive.google.com/uc?export=download&id=${id}`);
  assert.equal(rewriteShareLink(new URL(`https://drive.google.com/open?id=${id}`)).toString(),
    `https://drive.google.com/uc?export=download&id=${id}`);
  assert.equal(rewriteShareLink(new URL(`https://docs.google.com/document/d/${id}/edit`)).toString(),
    `https://docs.google.com/document/d/${id}/export?format=pdf`);
  assert.equal(rewriteShareLink(new URL("https://school.org/a.pdf")).toString(), "https://school.org/a.pdf");
});

test("the file type comes from its bytes, never its header", () => {
  assert.equal(sniffType(PDF), "application/pdf");
  assert.equal(sniffType(PNG), "image/png");
  assert.equal(sniffType(JPEG), "image/jpeg");
  assert.equal(sniffType(WEBP), "image/webp");
  assert.equal(sniffType(HEIC), "image/heic");
  assert.equal(sniffType(HTML), null);
  assert.equal(sniffType(new TextEncoder().encode("MZ\x90\x00")), null);
});

test("file names come from the server, then the URL, and always carry the right extension", () => {
  const url = new URL("https://school.org/files/Paper%201.pdf");
  assert.equal(fileNameFor(url, `attachment; filename="Maths P1.pdf"`, "application/pdf"), "Maths P1.pdf");
  assert.equal(fileNameFor(url, `attachment; filename*=UTF-8''Gr%C3%BCn.pdf`, "application/pdf"), "Grün.pdf");
  assert.equal(fileNameFor(url, null, "application/pdf"), "Paper 1.pdf");
  assert.equal(fileNameFor(new URL("https://drive.google.com/uc?export=download&id=x"), null, "application/pdf"), "Linked paper.pdf");
  assert.equal(fileNameFor(url, `attachment; filename="../../etc/passwd"`, "image/png"), ".. .. etc passwd.png");
});

test("a public PDF is fetched with its type and name", async () => {
  const { impl } = fakeFetch({ "https://school.org/p1.pdf": ok(PDF, { "content-type": "text/plain" }) });
  const file = await fetchLinkedFile("https://school.org/p1.pdf", { fetchImpl: impl });
  assert.equal(file.type, "application/pdf");
  assert.equal(file.name, "p1.pdf");
  assert.deepEqual([...file.bytes], [...PDF]);
});

test("redirects are followed by hand and every hop is checked again", async () => {
  const { impl, asked } = fakeFetch({
    "https://bit.example.org/x": redirect("https://cdn.school.org/p.pdf"),
    "https://cdn.school.org/p.pdf": ok(PDF),
  });
  assert.equal((await fetchLinkedFile("https://bit.example.org/x", { fetchImpl: impl })).finalUrl, "https://cdn.school.org/p.pdf");
  assert.deepEqual(asked, ["https://bit.example.org/x", "https://cdn.school.org/p.pdf"]);

  const toPrivate = fakeFetch({ "https://bit.example.org/x": redirect("https://169.254.169.254/latest/meta-data") });
  await refusedAsync(fetchLinkedFile("https://bit.example.org/x", { fetchImpl: toPrivate.impl }), "link_ip_literal");
  assert.deepEqual(toPrivate.asked, ["https://bit.example.org/x"], "the private address is never requested");

  const toHttp = fakeFetch({ "https://bit.example.org/x": redirect("http://school.org/p.pdf") });
  await refusedAsync(fetchLinkedFile("https://bit.example.org/x", { fetchImpl: toHttp.impl }), "link_not_https");
});

test("a redirect loop stops after three hops", async () => {
  const { impl, asked } = fakeFetch({
    "https://a.example.org/": redirect("https://b.example.org/"),
    "https://b.example.org/": redirect("https://a.example.org/"),
  });
  await refusedAsync(fetchLinkedFile("https://a.example.org/", { fetchImpl: impl }), "link_too_many_redirects");
  assert.equal(asked.length, 4);
});

test("a file over the cap is refused, by header and while reading", async () => {
  const big = new Uint8Array(2048); big.set(PDF);
  const declared = fakeFetch({ "https://s.org/a.pdf": ok(big, { "content-length": "2048" }) });
  await refusedAsync(fetchLinkedFile("https://s.org/a.pdf", { fetchImpl: declared.impl, maxBytes: 1024 }), "link_too_large");
  // No Content-Length: the stream is counted and cut off.
  const streamed = fakeFetch({ "https://s.org/a.pdf": () => new Response(new ReadableStream({
    start(c) { c.enqueue(big.subarray(0, 1000)); c.enqueue(big.subarray(1000)); c.close(); },
  })) });
  await refusedAsync(fetchLinkedFile("https://s.org/a.pdf", { fetchImpl: streamed.impl, maxBytes: 1024 }), "link_too_large");
});

test("a sign-in page, a sign-in status and a non-paper file each say why", async () => {
  const page = fakeFetch({ "https://drive.google.com/uc?export=download&id=1AbCdEfGhIjKlMnOp": ok(HTML, { "content-type": "application/pdf" }) });
  await refusedAsync(fetchLinkedFile("https://drive.google.com/file/d/1AbCdEfGhIjKlMnOp/view", { fetchImpl: page.impl }), "link_is_a_page");
  const status = fakeFetch({ "https://s.org/a.pdf": () => new Response("no", { status: 403 }) });
  await refusedAsync(fetchLinkedFile("https://s.org/a.pdf", { fetchImpl: status.impl }), "link_needs_sign_in");
  const exe = fakeFetch({ "https://s.org/a.pdf": ok(new TextEncoder().encode("MZ\x90\x00binary")) });
  await refusedAsync(fetchLinkedFile("https://s.org/a.pdf", { fetchImpl: exe.impl }), "link_unsupported_type");
  const empty = fakeFetch({ "https://s.org/a.pdf": ok(new Uint8Array()) });
  await refusedAsync(fetchLinkedFile("https://s.org/a.pdf", { fetchImpl: empty.impl }), "link_empty");
});

test("an unreachable or slow host is a refusal, not a crash", async () => {
  const none = fakeFetch({});
  await refusedAsync(fetchLinkedFile("https://gone.example.org/a.pdf", { fetchImpl: none.impl }), "link_unreachable");
  const slow = (async (_: RequestInfo | URL, init?: RequestInit) => new Promise<Response>((_, reject) => {
    // Node unrefs AbortSignal.timeout's timer; hold the event loop open until it fires.
    const hold = setTimeout(() => {}, 5000);
    init?.signal?.addEventListener("abort", () => { clearTimeout(hold); reject(Object.assign(new Error("timed out"), { name: "TimeoutError" })); });
  })) as typeof fetch;
  await refusedAsync(fetchLinkedFile("https://slow.example.org/a.pdf", { fetchImpl: slow, timeoutMs: 20 }), "link_timeout");
});
