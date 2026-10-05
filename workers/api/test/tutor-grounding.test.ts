import { beforeEach, describe, expect, test, vi } from "vitest";

/* AXO-36 / AXO-12: verified scheme and syllabus grounding for the Tutor.
   Synthetic rows only; no student content. */

const fixture = vi.hoisted(() => ({
  clientFor: vi.fn(),
  serviceClient: vi.fn(),
  intelligenceFetch: vi.fn(),
  resolve: vi.fn(),
}));

vi.mock("@mastery/shared/http.js", () => {
  const CORS = { "Access-Control-Allow-Origin": "https://axonstudy.online" };
  const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), {
    status, headers: { ...CORS, "Content-Type": "application/json" },
  });
  return {
    CORS, corsFor: () => CORS, withCors: (_r: Request, res: Response) => res, json,
    failure: (message: string, status = 400) => json({ error: message }, status),
    clientFor: fixture.clientFor, serviceClient: fixture.serviceClient,
    readJson: async (req: Request) => { try { return await req.json(); } catch { return null; } },
  };
});
vi.mock("@mastery/shared/r2.js", () => ({
  presignPut: vi.fn(), headObject: vi.fn(), signAssetUrl: vi.fn(), verifyAssetSignature: vi.fn(), objectKey: vi.fn(), BUCKET_FOR: {},
}));
vi.mock("@mastery/shared/contract.js", () => ({ CAPTURE: { MAX_PAGES: 60 }, PIPELINE_VERSION: "test", SAFE_OBJECT_NAME: /^[a-z]+$/ }));
vi.mock("@mastery/shared/assessment.js", () => ({ resolveSchemeEvidence: fixture.resolve }));

import worker from "../src/index.js";
import { loadSchemeGrounding, loadSyllabusGrounding, schemeToEvidence } from "../src/tutor_grounding.js";
import type { RegionRow } from "../src/tutor_evidence.js";

const A = "student-a";
const ASSESSMENT = "assessment-1";
const DOC = "scheme-doc-1";

function region(over: Partial<RegionRow> = {}): RegionRow {
  return {
    id: "region-1", question_label: "3(b)", question_text: "State the principle.", student_answer: "Energy is conserved.",
    marks_awarded: 1, marks_available: 2, teacher_remark: null, confidence_tier: "confident", student_confirmed_at: null,
    ...over,
  };
}

const scheme = (over: Record<string, unknown> = {}) => ({
  canonicalQuestionId: "cq-3", questionLabel: "3", retrievalMode: "ancestor_label", markingScheme: "1 mark: principle stated; 1 mark: applied.",
  source: "CBSE", version: "2026-27", sourceUrl: "https://cbseacademic.nic.in/web_material/SQP/ClassXII_2026_27/PhysicsMS.pdf",
  schemeDocumentId: DOC, assessmentIdentityId: ASSESSMENT, ...over,
});

type Rec = { table: string; filters: Array<[string, string, unknown]> };

/** Chainable fake query builder recording every filter with its operator. */
function db(tables: Record<string, (filters: Rec["filters"]) => { data: unknown; error: unknown }>, log: Rec[] = []) {
  return {
    log,
    rpc: vi.fn(async (name: string) => name === "tutor_enabled"
      ? { data: true, error: null }
      : { data: { active: true, student_id: A }, error: null }),
    from: vi.fn((table: string) => {
      const rec: Rec = { table, filters: [] };
      log.push(rec);
      const result = () => {
        const handler = tables[table];
        if (!handler) throw new Error("unexpected table " + table);
        return handler(rec.filters);
      };
      const b: any = {};
      for (const m of ["select", "order", "limit"]) b[m] = vi.fn(() => b);
      for (const op of ["eq", "in", "is", "neq"]) b[op] = vi.fn((col: string, v: unknown) => { rec.filters.push([op, col, v]); return b; });
      b.maybeSingle = vi.fn(async () => result());
      b.then = (resolve: any, reject: any) => Promise.resolve().then(result).then(resolve, reject);
      return b;
    }),
  };
}

const has = (rec: Rec | undefined, op: string, col: string, v: unknown) =>
  !!rec?.filters.some(([o, c, val]) => o === op && c === col && JSON.stringify(val) === JSON.stringify(v));

beforeEach(() => {
  vi.clearAllMocks();
  fixture.intelligenceFetch.mockResolvedValue(new Response("{}", { status: 200, headers: { "content-type": "application/json" } }));
});

describe("scheme grounding", () => {
  const admin = (paper: unknown) => db({ paper: () => ({ data: paper, error: null }) });

  test("a bound paper yields verified scheme evidence with exact provenance", async () => {
    const a = admin({ id: "paper-1", assessment_identity_id: ASSESSMENT });
    fixture.resolve.mockResolvedValue(scheme());
    const out = await loadSchemeGrounding(a, { studentId: A, paperId: "paper-1", regions: [region()] }, fixture.resolve);
    expect(out).toHaveLength(1);
    expect(out[0]).toMatchObject({
      id: "scheme:cq-3", source: "axon_db", informationClass: "VERIFIED_EXTERNAL", authority: "primary", verification: "verified",
      provenance: { paperId: "paper-1", url: scheme().sourceUrl, artifactHash: `scheme_document:${DOC}` },
      value: { kind: "official_marking_scheme", questionLabel: "3(b)", schemeQuestionLabel: "3", schemeVersion: "2026-27" },
    });
  });

  test("the service-role read is joined on this student and this paper", async () => {
    const a = admin({ id: "paper-1", assessment_identity_id: ASSESSMENT });
    fixture.resolve.mockResolvedValue(null);
    await loadSchemeGrounding(a, { studentId: A, paperId: "paper-1", regions: [region()] }, fixture.resolve);
    const paperRead = a.log.find((r) => r.table === "paper");
    expect(has(paperRead, "eq", "id", "paper-1")).toBe(true);
    expect(has(paperRead, "eq", "student_id", A)).toBe(true);
  });

  test("a paper that is not this student's never reaches the resolver", async () => {
    const a = admin(null);
    const out = await loadSchemeGrounding(a, { studentId: A, paperId: "someone-elses", regions: [region()] }, fixture.resolve);
    expect(out).toEqual([]);
    expect(fixture.resolve).not.toHaveBeenCalled();
  });

  test("an unbound paper (no exact assessment) gets no scheme — Tier 1 only", async () => {
    const a = admin({ id: "paper-1", assessment_identity_id: null });
    const out = await loadSchemeGrounding(a, { studentId: A, paperId: "paper-1", regions: [region()] }, fixture.resolve);
    expect(out).toEqual([]);
    expect(fixture.resolve).not.toHaveBeenCalled();
  });

  test("evidence from a different assessment than the paper's is refused", async () => {
    const a = admin({ id: "paper-1", assessment_identity_id: ASSESSMENT });
    fixture.resolve.mockResolvedValue(scheme({ assessmentIdentityId: "another-assessment" }));
    expect(await loadSchemeGrounding(a, { studentId: A, paperId: "paper-1", regions: [region()] }, fixture.resolve)).toEqual([]);
  });

  test("an empty or unverified registry produces no official scheme context", async () => {
    const a = admin({ id: "paper-1", assessment_identity_id: ASSESSMENT });
    fixture.resolve.mockResolvedValue(null); // resolver's fail-closed answer for revoked/pending/metadata-only
    expect(await loadSchemeGrounding(a, { studentId: A, paperId: "paper-1", regions: [region(), region({ id: "r2", question_label: "4" })] }, fixture.resolve)).toEqual([]);
  });

  test("a resolver error withholds grounding instead of failing the answer", async () => {
    const a = admin({ id: "paper-1", assessment_identity_id: ASSESSMENT });
    fixture.resolve.mockRejectedValue(new Error("db down"));
    expect(await loadSchemeGrounding(a, { studentId: A, paperId: "paper-1", regions: [region()] }, fixture.resolve)).toEqual([]);
  });

  test("no service client means no scheme evidence", async () => {
    expect(await loadSchemeGrounding(null, { studentId: A, paperId: "paper-1", regions: [region()] }, fixture.resolve)).toEqual([]);
  });

  test("two parts sharing one parent scheme row are cited once", async () => {
    const a = admin({ id: "paper-1", assessment_identity_id: ASSESSMENT });
    fixture.resolve.mockResolvedValue(scheme());
    const out = await loadSchemeGrounding(a, { studentId: A, paperId: "paper-1", regions: [region(), region({ id: "r2", question_label: "3(c)" })] }, fixture.resolve);
    expect(out.map((e) => e.id)).toEqual(["scheme:cq-3"]);
  });

  test("a non-https source is not presented as a citation", () => {
    expect(schemeToEvidence(scheme({ sourceUrl: "http://example.test/x.pdf" }) as any, region(), "paper-1")).toBeNull();
  });
});

describe("syllabus grounding", () => {
  const tables = (over: { docStatus?: string; tags?: unknown[] } = {}) => ({
    region_topic: () => ({ data: over.tags ?? [{ region_id: "region-1", topic_id: "t-1", is_primary: true }], error: null }),
    syllabus_topic: () => ({ data: [{ id: "t-1", document_id: "d-1", code: "1.2", kind: "objective", title: "Kinematics", objective_text: "use equations of motion for constant acceleration" }], error: null }),
    syllabus_document: (filters: Rec["filters"]) => {
      const row = { id: "d-1", provider_key: "cambridge", syllabus_code: "9702", version_label: "2025-2027", source_url: "https://www.cambridgeinternational.org/9702.pdf", status: over.docStatus ?? "verified" };
      const wantsVerified = filters.some(([o, c, v]) => o === "eq" && c === "status" && v === "verified");
      return { data: wantsVerified && row.status !== "verified" ? [] : [row], error: null };
    },
  });

  test("a tagged question carries the verbatim objective from a verified syllabus", async () => {
    const u = db(tables());
    const out = await loadSyllabusGrounding(u, { studentId: A, paperId: "paper-1", regions: [region()] });
    expect(out).toHaveLength(1);
    expect(out[0]).toMatchObject({
      id: "syllabus:t-1", source: "axon_db", verification: "verified",
      value: { kind: "syllabus_objective", questionLabels: ["3(b)"], code: "1.2", objectiveText: "use equations of motion for constant acceleration", syllabus: "9702 (2025-2027)" },
      provenance: { paperId: "paper-1", url: "https://www.cambridgeinternational.org/9702.pdf", artifactHash: "syllabus_document:d-1" },
    });
  });

  test("tags are read for this student only, excluding rejected and unsure tags", async () => {
    const u = db(tables());
    await loadSyllabusGrounding(u, { studentId: A, paperId: "paper-1", regions: [region()] });
    const tags = u.log.find((r) => r.table === "region_topic");
    expect(has(tags, "eq", "student_id", A)).toBe(true);
    expect(has(tags, "in", "region_id", ["region-1"])).toBe(true);
    expect(has(tags, "is", "student_rejected_at", null)).toBe(true);
    expect(has(tags, "neq", "confidence", "unsure")).toBe(true);
  });

  test("a draft syllabus never grounds an answer", async () => {
    const u = db(tables({ docStatus: "draft" }));
    expect(await loadSyllabusGrounding(u, { studentId: A, paperId: "paper-1", regions: [region()] })).toEqual([]);
  });

  test("an untagged question gets no syllabus evidence", async () => {
    const u = db(tables({ tags: [] }));
    expect(await loadSyllabusGrounding(u, { studentId: A, paperId: "paper-1", regions: [region()] })).toEqual([]);
  });

  test("a tag on a region outside this request is ignored", async () => {
    const u = db(tables({ tags: [{ region_id: "other-region", topic_id: "t-1", is_primary: true }] }));
    expect(await loadSyllabusGrounding(u, { studentId: A, paperId: "paper-1", regions: [region()] })).toEqual([]);
  });
});

describe("gateway forwards grounding with the paper evidence", () => {
  function sessionDb() {
    return db({
      student: () => ({ data: { id: A, board: "cbse", class_level: 12, programme_id: null, stage_id: null }, error: null }),
      paper: () => ({ data: { id: "paper-1", subject: "Physics" }, error: null }),
      student_subject: () => ({ data: [], error: null }),
      extraction_run: () => ({ data: { id: "run-1" }, error: null }),
      question_region: () => ({ data: [region()], error: null }),
      region_topic: () => ({ data: [], error: null }),
    });
  }
  const env = () => ({ AXON_INTERNAL_TOKEN: "t", INTELLIGENCE: { fetch: fixture.intelligenceFetch } }) as any;
  const post = () => new Request("https://api.test/tutor", {
    method: "POST", headers: { authorization: "Bearer jwt", "content-type": "application/json" },
    body: JSON.stringify({ studentId: A, message: "What did the scheme want here?", paperId: "paper-1" }),
  });

  test("bound paper: teacher evidence first, then the exact scheme row", async () => {
    fixture.clientFor.mockReturnValue(sessionDb());
    fixture.serviceClient.mockReturnValue(db({ paper: () => ({ data: { id: "paper-1", assessment_identity_id: ASSESSMENT }, error: null }) }));
    fixture.resolve.mockResolvedValue(scheme());
    const res = await worker.fetch(post(), env());
    expect(res.status).toBe(200);
    const body = JSON.parse(fixture.intelligenceFetch.mock.calls[0][1].body);
    expect(body.evidence.map((e: any) => e.id)).toEqual(["paper:region-1", "teacher:region-1", "scheme:cq-3"]);
  });

  test("a missing service key still answers, just without the scheme", async () => {
    fixture.clientFor.mockReturnValue(sessionDb());
    fixture.serviceClient.mockImplementation(() => { throw new Error("SUPABASE_SERVICE_ROLE_KEY is not set"); });
    const res = await worker.fetch(post(), env());
    expect(res.status).toBe(200);
    const body = JSON.parse(fixture.intelligenceFetch.mock.calls[0][1].body);
    expect(body.evidence.map((e: any) => e.source)).toEqual(["paper", "teacher"]);
  });
});
