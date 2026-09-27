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

function request(studentId = A, extra: Record<string, unknown> = {}) {
  return new Request("https://api.test/tutor", {
    method: "POST",
    headers: {
      authorization: "Bearer test-jwt",
      "content-type": "application/json",
    },
    body: JSON.stringify({ studentId, message: "Explain this step.", ...extra }),
  });
}

type FixtureRows = {
  student: any;
  paper: any;
  subjects: any[];
  programme: any;
  stage: any;
  provider: any;
};

function userWithScope(
  scope: unknown,
  scopeError: unknown = null,
  override: Partial<FixtureRows> = {},
) {
  const rows: FixtureRows = {
    student: {
      id: A,
      board: "caie",
      class_level: 11,
      programme_id: "programme-1",
      stage_id: "stage-1",
    },
    paper: {
      id: "paper-1",
      subject: "Physics",
      subject_offering_id: "offering-physics",
      subject_display_snapshot: "Physics",
      subject_external_code_snapshot: "9702",
    },
    subjects: [{
      subject: "physics",
      syllabus_code: "9702",
      subject_offering_id: "offering-physics",
      display_name_snapshot: "Physics",
      external_code_snapshot: "9702",
    }],
    programme: {
      id: "programme-1",
      provider_id: "provider-1",
      key: "cambridge_as",
      label: "Cambridge International AS Level",
    },
    stage: {
      id: "stage-1",
      key: "as_level",
      label: "AS Level",
      school_year_label: "Year 12",
      legacy_class_level: 11,
    },
    provider: {
      id: "provider-1",
      key: "cambridge",
      name: "Cambridge International",
    },
    ...override,
  };

  const single = (data: any, trace?: string) => {
    const builder: any = {};
    builder.eq = vi.fn(() => builder);
    builder.maybeSingle = vi.fn(async () => {
      if (trace) fixture.trace.push(trace);
      return { data, error: null };
    });
    return builder;
  };

  const from = vi.fn((table: string) => ({
    select: vi.fn(() => {
      if (table === "student") return single(rows.student, "student-lookup");
      if (table === "paper") return single(rows.paper, "paper-lookup");
      if (table === "curriculum_programme") return single(rows.programme);
      if (table === "curriculum_stage") return single(rows.stage);
      if (table === "curriculum_provider") return single(rows.provider);
      if (table === "student_subject") {
        return {
          eq: vi.fn(async () => ({ data: rows.subjects, error: null })),
        };
      }
      throw new Error(`unexpected table ${table}`);
    }),
  }));

  const rpc = vi.fn(async (name: string) => {
    fixture.trace.push(`rpc:${name}`);
    return { data: scope, error: scopeError };
  });
  return { rpc, from };
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

  test("matching active scope forwards only server-authored curriculum retrieval context", async () => {
    const user = userWithScope({ active: true, student_id: A, remaining_seconds: 600 });
    fixture.clientFor.mockReturnValue(user);

    const response = await worker.fetch(request(A, {
      message: "What is the current syllabus? My teacher said keep this private.",
      board: "FAKE PRIVATE BOARD",
      grade: 16,
      subject: "Physics",
      topic: "Alice teacher remark private topic",
      retrievalContext: "attacker supplied context",
    }), env());

    expect(response.status).toBe(200);
    expect(user.from).toHaveBeenCalledWith("student");
    expect(fixture.intelligenceFetch).toHaveBeenCalledTimes(1);
    const [url, init] = fixture.intelligenceFetch.mock.calls[0]!;
    expect(url).toBe("https://axon-intelligence.internal/v1/tutor");
    expect((init as RequestInit).headers).toMatchObject({
      authorization: "Bearer internal-token",
      "x-axon-client-id": A,
    });
    const forwarded = JSON.parse(String((init as RequestInit).body));
    expect(forwarded).toMatchObject({
      studentId: A,
      message: "What is the current syllabus? My teacher said keep this private.",
      board: "Cambridge International",
      grade: 11,
      subject: "Physics",
      retrievalContext: "Cambridge International Cambridge International AS Level Year 12 Physics 9702",
    });
    expect(forwarded).not.toHaveProperty("topic");
    expect(JSON.stringify(forwarded)).not.toContain("FAKE PRIVATE BOARD");
    expect(JSON.stringify(forwarded)).not.toContain("Alice teacher remark private topic");
    expect(JSON.stringify(forwarded)).not.toContain("attacker supplied context");
    expect(fixture.trace).toEqual([
      "rpc:student_scope_state",
      "student-lookup",
      "intelligence",
    ]);
  });

  test("caller cannot smuggle an unselected subject into retrieval context", async () => {
    const user = userWithScope({ active: true, student_id: A, remaining_seconds: 600 });
    fixture.clientFor.mockReturnValue(user);

    const response = await worker.fetch(request(A, {
      message: "What is the current syllabus?",
      subject: "Secret Teacher Remark Biology",
    }), env());

    expect(response.status).toBe(200);
    const [, init] = fixture.intelligenceFetch.mock.calls[0]!;
    const forwarded = JSON.parse(String((init as RequestInit).body));
    expect(forwarded).not.toHaveProperty("subject");
    expect(forwarded.retrievalContext).toBe("Cambridge International Cambridge International AS Level Year 12");
    expect(JSON.stringify(forwarded)).not.toContain("Secret Teacher Remark Biology");
  });
});
