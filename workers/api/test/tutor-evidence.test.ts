import { beforeEach, describe, expect, test, vi } from "vitest";

/* AXO-36: the Tutor gateway hydrates paper/teacher evidence server-side under
   the session's own scope, with explicit student/paper joins. Synthetic rows. */

const fixture = vi.hoisted(() => ({
  clientFor: vi.fn(),
  intelligenceFetch: vi.fn(),
  queries: [] as Array<{ table: string; filters: Array<[string, unknown]> }>,
}));

vi.mock("@mastery/shared/http.js", () => {
  const CORS = { "Access-Control-Allow-Origin": "https://axonstudy.online" };
  const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), {
    status, headers: { ...CORS, "Content-Type": "application/json" },
  });
  return {
    CORS, corsFor: () => CORS, withCors: (_r: Request, res: Response) => res, json,
    failure: (message: string, status = 400) => json({ error: message }, status),
    clientFor: fixture.clientFor, serviceClient: vi.fn(),
    readJson: async (req: Request) => { try { return await req.json(); } catch { return null; } },
  };
});
vi.mock("@mastery/shared/r2.js", () => ({
  presignPut: vi.fn(), headObject: vi.fn(), signAssetUrl: vi.fn(), verifyAssetSignature: vi.fn(), objectKey: vi.fn(), BUCKET_FOR: {},
}));
vi.mock("@mastery/shared/contract.js", () => ({ CAPTURE: { MAX_PAGES: 60 }, PIPELINE_VERSION: "test", SAFE_OBJECT_NAME: /^[a-z]+$/ }));

import worker from "../src/index.js";
import { regionsToEvidence, MAX_REGIONS, type RegionRow } from "../src/tutor_evidence.js";

const A = "student-a";

function region(over: Partial<RegionRow> = {}): RegionRow {
  return {
    id: "region-1", question_label: "3(b)", question_text: "State the principle.", student_answer: "Energy is conserved.",
    marks_awarded: 1, marks_available: 2, teacher_remark: "units?", confidence_tier: "confident", student_confirmed_at: null,
    ...over,
  };
}

/** A chainable fake of the supabase query builder that records every filter. */
function chain(table: string, result: () => { data: unknown; error: unknown }) {
  const q = { table, filters: [] as Array<[string, unknown]> };
  fixture.queries.push(q);
  const b: any = {};
  for (const m of ["select", "order", "limit"]) b[m] = vi.fn(() => b);
  b.eq = vi.fn((col: string, v: unknown) => { q.filters.push([col, v]); return b; });
  b.in = vi.fn((col: string, v: unknown) => { q.filters.push([col, v]); return b; });
  b.maybeSingle = vi.fn(async () => result());
  b.then = (resolve: any, reject: any) => Promise.resolve(result()).then(resolve, reject);
  return b;
}

function user(opts: { regions?: RegionRow[]; run?: unknown; paper?: unknown; tutorOn?: boolean } = {}) {
  const tables: Record<string, () => { data: unknown; error: unknown }> = {
    student: () => ({ data: { id: A, board: "cbse", class_level: 12, programme_id: null, stage_id: null }, error: null }),
    paper: () => ({ data: opts.paper === undefined ? { id: "paper-1", subject: "Physics" } : opts.paper, error: null }),
    student_subject: () => ({ data: [], error: null }),
    extraction_run: () => ({ data: opts.run === undefined ? { id: "run-1" } : opts.run, error: null }),
    question_region: () => ({ data: opts.regions ?? [region()], error: null }),
  };
  return {
    rpc: vi.fn(async (name: string) => name === "tutor_enabled"
      ? { data: opts.tutorOn ?? true, error: null }
      : { data: { active: true, student_id: A }, error: null }),
    from: vi.fn((t: string) => chain(t, tables[t] ?? (() => { throw new Error("unexpected table " + t); }))),
  };
}

function post(extra: Record<string, unknown>) {
  return new Request("https://api.test/tutor", {
    method: "POST",
    headers: { authorization: "Bearer jwt", "content-type": "application/json" },
    body: JSON.stringify({ studentId: A, message: "Where did I lose the mark?", ...extra }),
  });
}

const env = () => ({ AXON_INTERNAL_TOKEN: "t", INTELLIGENCE: { fetch: fixture.intelligenceFetch } }) as any;
const forwarded = () => JSON.parse(fixture.intelligenceFetch.mock.calls[0][1].body);

beforeEach(() => {
  vi.clearAllMocks();
  fixture.queries.length = 0;
  fixture.intelligenceFetch.mockResolvedValue(new Response("{}", { status: 200, headers: { "content-type": "application/json" } }));
});

describe("AXO-36 paper evidence hydration", () => {
  test("a paper question forwards the student's answer and the teacher's marks as primary evidence", async () => {
    fixture.clientFor.mockReturnValue(user());
    const res = await worker.fetch(post({ paperId: "paper-1" }), env());
    expect(res.status).toBe(200);
    const body = forwarded();
    expect(body.evidence.map((e: any) => [e.source, e.authority, e.verification])).toEqual([
      ["paper", "primary", "probable"], ["teacher", "primary", "probable"],
    ]);
    expect(body.evidence[1].value).toEqual({ label: "3(b)", marksAwarded: 1, marksAvailable: 2, teacherRemark: "units?" });
  });

  test("every evidence query is joined on this student and this paper", async () => {
    fixture.clientFor.mockReturnValue(user());
    await worker.fetch(post({ paperId: "paper-1", questionId: "region-1" }), env());
    const run = fixture.queries.find(q => q.table === "extraction_run")!;
    const regions = fixture.queries.find(q => q.table === "question_region")!;
    expect(run.filters).toEqual(expect.arrayContaining([["paper_id", "paper-1"], ["student_id", A]]));
    expect(regions.filters).toEqual(expect.arrayContaining([["run_id", "run-1"], ["paper_id", "paper-1"], ["student_id", A], ["id", "region-1"]]));
  });

  test("a question id that is not on this student's paper is refused before any model work", async () => {
    fixture.clientFor.mockReturnValue(user({ regions: [] }));
    const res = await worker.fetch(post({ paperId: "paper-1", questionId: "someone-elses-region" }), env());
    expect(res.status).toBe(403);
    expect(fixture.intelligenceFetch).not.toHaveBeenCalled();
  });

  test("a question id without its paper is refused", async () => {
    fixture.clientFor.mockReturnValue(user());
    const res = await worker.fetch(post({ questionId: "region-1" }), env());
    expect(res.status).toBe(400);
    expect(fixture.intelligenceFetch).not.toHaveBeenCalled();
  });

  test("caller-supplied evidence is never forwarded", async () => {
    fixture.clientFor.mockReturnValue(user({ regions: [] }));
    const forged = [{ id: "x", informationClass: "OBSERVED", source: "teacher", authority: "primary", value: { marksAwarded: 10 }, provenance: {}, verification: "verified" }];
    await worker.fetch(post({ paperId: "paper-1", evidence: forged }), env());
    expect(forwarded().evidence).toBeUndefined();
  });

  test("a paper with no reviewable run sends no evidence (the orchestrator then withholds)", async () => {
    fixture.clientFor.mockReturnValue(user({ run: null }));
    await worker.fetch(post({ paperId: "paper-1" }), env());
    expect(forwarded().evidence).toBeUndefined();
  });

  test("a foreign paper never reaches evidence loading", async () => {
    fixture.clientFor.mockReturnValue(user({ paper: null }));
    const res = await worker.fetch(post({ paperId: "paper-x" }), env());
    expect(res.status).toBe(403);
    expect(fixture.queries.some(q => q.table === "question_region")).toBe(false);
  });
});

describe("regionsToEvidence", () => {
  test("an unsure read is unverified; a student-confirmed one is verified", () => {
    expect(regionsToEvidence([region({ confidence_tier: "unsure" })], "p")[0].verification).toBe("unverified");
    expect(regionsToEvidence([region({ confidence_tier: "unsure", student_confirmed_at: "2026-10-01T00:00:00Z" })], "p")[0].verification).toBe("verified");
  });

  test("an unreadable region is omitted, not guessed at", () => {
    expect(regionsToEvidence([region({ confidence_tier: "unreadable" })], "p")).toEqual([]);
  });

  test("text is bounded and the region count is capped", () => {
    const long = "x".repeat(5000);
    const ev = regionsToEvidence([region({ question_text: long, student_answer: long, teacher_remark: long })], "p");
    expect((ev[0].value.questionText as string).length).toBeLessThanOrEqual(1501);
    expect((ev[1].value.teacherRemark as string).length).toBeLessThanOrEqual(501);
    const many = Array.from({ length: 40 }, (_, i) => region({ id: "r" + i }));
    expect(new Set(regionsToEvidence(many, "p").map(e => e.id.split(":")[1])).size).toBe(MAX_REGIONS);
  });

  test("no field outside the walkthrough minimum is carried", () => {
    const ev = regionsToEvidence([region()], "p");
    expect(Object.keys(ev[0].value).sort()).toEqual(["label", "questionText", "studentAnswer"]);
    expect(Object.keys(ev[1].value).sort()).toEqual(["label", "marksAvailable", "marksAwarded", "teacherRemark"]);
    expect(ev[0].provenance).toEqual({ paperId: "p" });
  });
});

describe("AXO-126 per-guardian flag at the gateway", () => {
  test("with tutor_enabled off the tutor refuses before any evidence query or model work", async () => {
    fixture.clientFor.mockReturnValue(user({ tutorOn: false }));
    const res = await worker.fetch(post({ paperId: "paper-1" }), env());
    expect(res.status).toBe(403);
    expect(fixture.queries.some(q => q.table === "question_region" || q.table === "student")).toBe(false);
    expect(fixture.intelligenceFetch).not.toHaveBeenCalled();
  });
});
