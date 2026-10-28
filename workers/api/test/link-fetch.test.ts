import { beforeEach, expect, test, vi } from "vitest";

// AXO-295: the /link-fetch route. The fetch rules themselves are tested in
// shared/src/__tests__/link_fetch.test.ts; this covers the route around them.
const f = vi.hoisted(() => ({ clientFor: vi.fn(), getUser: vi.fn() }));
vi.mock("@mastery/shared/http.js", async (orig) => ({
  ...await orig<any>(),
  clientFor: f.clientFor,
  readJson: async (req: Request) => { try { return await req.json(); } catch { return null; } },
}));

import worker, { linkFetch } from "../src/index.js";

const PDF = new TextEncoder().encode("%PDF-1.7\n");
const post = (body: unknown, auth = true) => new Request("https://mastery-api.example/link-fetch", {
  method: "POST",
  headers: { "content-type": "application/json", ...(auth ? { authorization: "Bearer jwt" } : {}), origin: "https://axonstudy.online" },
  body: JSON.stringify(body),
});
const remote = (bytes: Uint8Array, status = 200) => vi.fn(async () => new Response(bytes, { status }));

beforeEach(() => {
  f.getUser.mockReset().mockResolvedValue({ data: { user: { id: "guardian" } }, error: null });
  f.clientFor.mockReset().mockImplementation((req: Request) => (req.headers.get("authorization") ? { auth: { getUser: f.getUser } } : null));
});

test("only a signed-in caller can use the fetcher", async () => {
  const fetchImpl = remote(PDF);
  expect((await linkFetch(post({ url: "https://school.org/p.pdf" }, false), {} as any, fetchImpl)).status).toBe(401);
  f.getUser.mockResolvedValue({ data: { user: null }, error: { message: "bad jwt" } });
  expect((await linkFetch(post({ url: "https://school.org/p.pdf" }), {} as any, fetchImpl)).status).toBe(401);
  expect(fetchImpl).not.toHaveBeenCalled();
});

test("a fetched PDF comes back as bytes with its type and a readable file name", async () => {
  const res = await linkFetch(post({ url: "https://school.org/Paper%201.pdf" }), {} as any, remote(PDF));
  expect(res.status).toBe(200);
  expect(res.headers.get("content-type")).toBe("application/pdf");
  expect(decodeURIComponent(res.headers.get("x-axon-filename")!)).toBe("Paper 1.pdf");
  expect(res.headers.get("access-control-expose-headers")).toContain("X-Axon-Filename");
  expect(new Uint8Array(await res.arrayBuffer())).toEqual(PDF);
});

test("a refused link answers with the reason and a machine code, without touching the network", async () => {
  const fetchImpl = remote(PDF);
  const res = await linkFetch(post({ url: "https://169.254.169.254/latest" }), {} as any, fetchImpl);
  expect(res.status).toBe(422);
  const body = await res.json() as any;
  expect(body.detail.code).toBe("link_ip_literal");
  expect(body.error).toMatch(/address rather than a website/);
  expect(fetchImpl).not.toHaveBeenCalled();
});

test("a missing or oversized url is a plain bad request", async () => {
  expect((await linkFetch(post({}), {} as any, remote(PDF))).status).toBe(400);
  expect((await linkFetch(post({ url: "https://s.org/" + "a".repeat(3000) }), {} as any, remote(PDF))).status).toBe(400);
});

test("the route is wired into the worker", async () => {
  // Unauthenticated, so it stops at the auth check; reaching it at all proves the route exists.
  const res = await worker.fetch(post({ url: "https://school.org/p.pdf" }, false), {} as any);
  expect(res.status).toBe(401);
});
