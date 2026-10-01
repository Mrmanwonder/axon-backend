import { beforeEach, describe, expect, test, vi } from "vitest";

const fixture = vi.hoisted(() => ({
  clientFor: vi.fn(),
  serviceClient: vi.fn(),
  sendBatch: vi.fn(async () => undefined),
}));

vi.mock("@mastery/shared/http.js", () => {
  const CORS = { "Access-Control-Allow-Origin": "https://axonstudy.online" };
  const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), {
    status,
    headers: { ...CORS, "Content-Type": "application/json" },
  });
  return {
    CORS,
    corsFor: () => CORS,
    withCors: (_req: Request, response: Response) => response,
    json,
    failure: (message: string, status = 400, detail?: unknown) =>
      json({ error: message, detail: detail ?? null }, status),
    clientFor: fixture.clientFor,
    serviceClient: fixture.serviceClient,
    readJson: async (req: Request) => {
      try { return await req.json(); } catch { return null; }
    },
  };
});

vi.mock("@mastery/shared/r2.js", () => ({
  presignPut: vi.fn(),
  headObject: vi.fn(),
  signAssetUrl: vi.fn(),
  verifyAssetSignature: vi.fn(),
  objectKey: vi.fn(),
  BUCKET_FOR: {},
}));

vi.mock("@mastery/shared/contract.js", () => ({
  CAPTURE: { MAX_PAGES: 60, UPLOAD_EXTENSIONS: {} },
  PIPELINE_VERSION: "test",
  SAFE_OBJECT_NAME: /^[A-Za-z0-9._-]+$/,
}));

import worker from "../src/index.js";

function request(body: unknown) {
  return new Request("https://api.test/explain-retry", {
    method: "POST",
    headers: { authorization: "Bearer t", "content-type": "application/json" },
    body: JSON.stringify(body),
  });
}

function userWithRun(run: unknown) {
  const b: any = {};
  b.select = vi.fn(() => b);
  b.eq = vi.fn(() => b);
  b.maybeSingle = vi.fn(async () => ({ data: run, error: null }));
  return { from: vi.fn(() => b) };
}

const env: any = { EXPLAIN_QUEUE: { sendBatch: fixture.sendBatch } };

describe("POST /explain-retry", () => {
  beforeEach(() => { vi.clearAllMocks(); });

  test("401 without a session", async () => {
    fixture.clientFor.mockReturnValue(null);
    const res = await worker.fetch(request({ run_id: "r1" }), env, {} as any);
    expect(res.status).toBe(401);
  });

  test("403 when the run is not visible to the user", async () => {
    fixture.clientFor.mockReturnValue(userWithRun(null));
    const rpc = vi.fn();
    fixture.serviceClient.mockReturnValue({ rpc });
    const res = await worker.fetch(request({ run_id: "r1" }), env, {} as any);
    expect(res.status).toBe(403);
    expect(rpc).not.toHaveBeenCalled();
  });

  test("re-queues the regions the RPC returns", async () => {
    fixture.clientFor.mockReturnValue(userWithRun({ id: "r1" }));
    const rpc = vi.fn(async () => ({ data: { queued: 2, region_ids: ["a", "b"] }, error: null }));
    fixture.serviceClient.mockReturnValue({ rpc });
    const res = await worker.fetch(request({ run_id: "r1" }), env, {} as any);
    expect(res.status).toBe(200);
    expect(await res.json()).toEqual({ run_id: "r1", retrying: 2 });
    expect(rpc).toHaveBeenCalledWith("retry_failed_explanations", { p_run_id: "r1" });
    expect(fixture.sendBatch).toHaveBeenCalledTimes(1);
    expect(fixture.sendBatch.mock.calls[0][0]).toEqual([
      { body: { run_id: "r1", region_id: "a" } },
      { body: { run_id: "r1", region_id: "b" } },
    ]);
  });

  test("sends nothing when no region failed", async () => {
    fixture.clientFor.mockReturnValue(userWithRun({ id: "r1" }));
    fixture.serviceClient.mockReturnValue({ rpc: vi.fn(async () => ({ data: { queued: 0, region_ids: [] }, error: null })) });
    const res = await worker.fetch(request({ run_id: "r1" }), env, {} as any);
    expect(await res.json()).toEqual({ run_id: "r1", retrying: 0 });
    expect(fixture.sendBatch).not.toHaveBeenCalled();
  });
});
