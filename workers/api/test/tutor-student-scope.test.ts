import { beforeEach, describe, expect, test, vi } from "vitest";

const fixture = vi.hoisted(() => ({
  clientFor: vi.fn(),
  serviceClient: vi.fn(),
  intelligenceFetch: vi.fn(),
  trace: [] as string[],
}));

vi.mock("@mastery/shared/http.js", () => {
  const CORS = { "Access-Control-Allow-Origin": "https://axonstudy.online" };
  const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), {
    status,
    headers: { ...CORS, "Content-Type": "application/json" },
  });
  return {
    CORS,
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
  CAPTURE: { MAX_PAGES: 60 },
  PIPELINE_VERSION: "test",
  SAFE_OBJECT_NAME: /^[a-zA-Z0-9._-]+$/,
}));

import worker from "../src/index.js";

const A = "student-a";
const B = "student-b";

function request(studentId = A) {
  return new Request("https://api.test/tutor", {
    method: "POST",
    headers: {
      authorization: "Bearer test-jwt",
      "content-type": "application/json",
    },
    body: JSON.stringify({ studentId, message: "Explain this step." }),
  });
}

function userWithScope(scope: unknown, scopeError: unknown = null) {
  const maybeSingle = vi.fn(async () => {
    fixture.trace.push("student-lookup");
    return { data: { id: A }, error: null };
  });
  const eq = vi.fn(() => ({ maybeSingle }));
  const select = vi.fn(() => ({ eq }));
  const from = vi.fn(() => {
    fixture.trace.push("from-student");
    return { select };
  });
  const rpc = vi.fn(async (name: string) => {
    fixture.trace.push(`rpc:${name}`);
    return { data: scope, error: scopeError };
  });
  return { rpc, from, maybeSingle };
}

function env() {
  return {
    AXON_INTERNAL_TOKEN: "internal-token",
    INTELLIGENCE: { fetch: fixture.intelligenceFetch },
  } as any;
}

beforeEach(() => {
  vi.clearAllMocks();
  fixture.trace.length = 0;
  fixture.intelligenceFetch.mockImplementation(async () => {
    fixture.trace.push("intelligence");
    return new Response(JSON.stringify({ ok: true }), {
      status: 200,
      headers: { "content-type": "application/json" },
    });
  });
});

describe("AXO-63 Tutor Student Mode abuse boundary", () => {
  test("active Student A scope cannot target owned sibling B", async () => {
    const user = userWithScope({ active: true, student_id: A, remaining_seconds: 600 });
    fixture.clientFor.mockReturnValue(user);

    const response = await worker.fetch(request(B), env());

    expect(response.status).toBe(403);
    expect(await response.json()).toMatchObject({ error: "Switch to that student first." });
    expect(user.rpc).toHaveBeenCalledWith("student_scope_state");
    expect(user.from).not.toHaveBeenCalled();
    expect(fixture.intelligenceFetch).not.toHaveBeenCalled();
    expect(fixture.trace).toEqual(["rpc:student_scope_state"]);
  });

  test("revoked or expired Student Mode cannot reach student lookup or model work", async () => {
    const user = userWithScope({
      active: false,
      student_id: null,
      remaining_seconds: 0,
      reason: "expired_or_revoked",
    });
    fixture.clientFor.mockReturnValue(user);

    const response = await worker.fetch(request(A), env());

    expect(response.status).toBe(403);
    expect(user.from).not.toHaveBeenCalled();
    expect(fixture.intelligenceFetch).not.toHaveBeenCalled();
    expect(fixture.trace).toEqual(["rpc:student_scope_state"]);
  });

  test("scope verification failure fails closed before resource or model work", async () => {
    const user = userWithScope(null, { message: "database unavailable" });
    fixture.clientFor.mockReturnValue(user);

    const response = await worker.fetch(request(A), env());

    expect(response.status).toBe(503);
    expect(await response.json()).toMatchObject({ error: "We could not verify the active student." });
    expect(user.from).not.toHaveBeenCalled();
    expect(fixture.intelligenceFetch).not.toHaveBeenCalled();
    expect(fixture.trace).toEqual(["rpc:student_scope_state"]);
  });

  test("matching active scope proceeds through ownership lookup and Intelligence", async () => {
    const user = userWithScope({ active: true, student_id: A, remaining_seconds: 600 });
    fixture.clientFor.mockReturnValue(user);

    const response = await worker.fetch(request(A), env());

    expect(response.status).toBe(200);
    expect(user.from).toHaveBeenCalledWith("student");
    expect(fixture.intelligenceFetch).toHaveBeenCalledTimes(1);
    const [url, init] = fixture.intelligenceFetch.mock.calls[0]!;
    expect(url).toBe("https://axon-intelligence.internal/v1/tutor");
    expect((init as RequestInit).headers).toMatchObject({
      authorization: "Bearer internal-token",
      "x-axon-client-id": A,
    });
    expect(JSON.parse(String((init as RequestInit).body))).toMatchObject({
      studentId: A,
      message: "Explain this step.",
    });
    expect(fixture.trace).toEqual([
      "rpc:student_scope_state",
      "from-student",
      "student-lookup",
      "intelligence",
    ]);
  });
});
