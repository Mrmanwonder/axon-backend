import { beforeEach, describe, expect, test, vi } from "vitest";

const fixture = vi.hoisted(() => ({
  clientFor: vi.fn(),
  rpc: vi.fn(),
  requestedKeys: [] as string[],
  paperOwned: true,
  serviceClient: vi.fn(),
  presignPut: vi.fn(),
  headObject: vi.fn(),
  signAssetUrl: vi.fn(),
  insert: vi.fn(),
  cleanupInsert: vi.fn(),
  deleteObject: vi.fn(),
  update: vi.fn(),
  pages: [] as any[],
  ledger: null as any[] | null,
  dbUpdateErrorKey: null as string | null,
  headErrorKey: null as string | null,
  presignActive: 0,
  presignMax: 0,
  headActive: 0,
  headMax: 0,
  dbActive: 0,
  dbMax: 0,
  signActive: 0,
  signMax: 0,
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
  presignPut: fixture.presignPut,
  sealUpload: fixture.headObject,
  signAssetUrl: fixture.signAssetUrl,
  verifyAssetSignature: vi.fn(),
  objectKey: ({ studentId, paperId, kind, name, extension }: any) =>
    `${studentId}/${paperId}/${kind}/${name}.${extension}`,
  deleteObject: fixture.deleteObject,
  stagingKey: (key: string) => key + ".pending",
  BUCKET_FOR: {
    upload: "originals",
    raw: "originals",
    page: "derived",
    crop: "derived",
    mask: "derived",
    thumb: "derived",
  },
}));

vi.mock("@mastery/shared/contract.js", () => ({
  CAPTURE: {
    MAX_PAGES: 60,
    UPLOAD_EXTENSIONS: {
      "image/jpeg": "jpg",
      "image/png": "png",
      "image/webp": "webp",
      "application/pdf": "pdf",
    },
  },
  PIPELINE_VERSION: "test",
  SAFE_OBJECT_NAME: /^[A-Za-z0-9._-]+$/,
}));

import worker, { IO_CONCURRENCY, mapLimit } from "../src/index.js";

const STUDENT = "student-a";
const PAPER = "paper-a";

const sleep = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));

function userClient() {
  return {
    auth: { getUser: vi.fn(async () => ({ data: { user: { id: "owner" } }, error: null })) },
    from: vi.fn((table: string) => {
      if (table === "paper") {
        const builder: any = {};
        builder.select = vi.fn(() => builder);
        builder.eq = vi.fn(() => builder);
        builder.maybeSingle = vi.fn(async () => ({
          data: fixture.paperOwned ? { id: PAPER, student_id: STUDENT } : null,
          error: null,
        }));
        return builder;
      }
      if (table === "paper_page") {
        const builder: any = {};
        builder.select = vi.fn(() => builder);
        builder.eq = vi.fn(() => builder);
        builder.in = vi.fn(async () => ({ data: fixture.pages, error: null }));
        return builder;
      }
      throw new Error(`unexpected user table ${table}`);
    }),
  };
}

function adminClient() {
  const from = vi.fn((table: string) => {
    if (table === "r2_deletion") return { insert: fixture.cleanupInsert };
    if (table !== "upload") throw new Error(`unexpected admin table ${table}`);
    return {
      insert: fixture.insert,
      update: fixture.update,
      select: () => {
        const rows = () => fixture.ledger ?? Array.from({ length: 60 }, (_, i) => ({ r2_bucket: "originals", r2_key: uploadClaim(i + 1).key, bytes: 100 }));
        const b: any = { eq: () => b, in: (_column: string, keys: string[]) => { fixture.requestedKeys = keys; return b; },
          maybeSingle: async () => ({ data: rows()[0] ?? null, error: null }),
          then: (yes: any) => Promise.resolve({ data: rows(), error: null }).then(yes) };
        return b;
      },
    };
  });
  return { from, rpc: fixture.rpc };
}

function request(path: string, body: unknown) {
  return new Request(`https://api.test${path}`, {
    method: "POST",
    headers: {
      authorization: "Bearer test-jwt",
      "content-type": "application/json",
    },
    body: JSON.stringify(body),
  });
}

function uploadObject(i: number, overrides: Record<string, unknown> = {}) {
  return {
    kind: "upload",
    name: String(i),
    content_type: "image/jpeg",
    bytes: 100,
    ...overrides,
  };
}

function uploadClaim(i: number) {
  return {
    bucket: "originals",
    key: `${STUDENT}/${PAPER}/upload/${i}.jpg`,
    bytes: 100,
    sha256: `client-${i}`,
  };
}

beforeEach(() => {
  vi.clearAllMocks();
  fixture.pages = [];
  fixture.requestedKeys=[]; fixture.paperOwned=true;
  fixture.rpc.mockResolvedValue({data:{attached:[{page_number:1,key:STUDENT+'/'+PAPER+'/raw/p1-original-x.jpg'}]},error:null});
  fixture.ledger = null;
  fixture.dbUpdateErrorKey = null;
  fixture.headErrorKey = null;
  fixture.presignActive = fixture.presignMax = 0;
  fixture.headActive = fixture.headMax = 0;
  fixture.dbActive = fixture.dbMax = 0;
  fixture.signActive = fixture.signMax = 0;

  fixture.cleanupInsert.mockResolvedValue({ error: null });
  fixture.deleteObject.mockResolvedValue(undefined);
  fixture.clientFor.mockReturnValue(userClient());
  fixture.serviceClient.mockReturnValue(adminClient());

  fixture.presignPut.mockImplementation(async (_env: unknown, _bucket: string, key: string) => {
    fixture.presignActive++;
    fixture.presignMax = Math.max(fixture.presignMax, fixture.presignActive);
    await sleep(5);
    fixture.presignActive--;
    return `https://upload.test/${encodeURIComponent(key)}`;
  });

  fixture.headObject.mockImplementation(async (_env: unknown, _bucket: string, key: string) => {
    fixture.headActive++;
    fixture.headMax = Math.max(fixture.headMax, fixture.headActive);
    await sleep(5);
    fixture.headActive--;
    if (fixture.headErrorKey === key) throw new Error("R2 unavailable");
    return { bytes: 100, etag: `etag-${key}` };
  });

  fixture.signAssetUrl.mockImplementation(async (_env: unknown, _bucket: string, key: string) => {
    fixture.signActive++;
    fixture.signMax = Math.max(fixture.signMax, fixture.signActive);
    await sleep(5);
    fixture.signActive--;
    return `https://asset.test/${encodeURIComponent(key)}`;
  });

  fixture.insert.mockImplementation((rows: any[]) => ({
    select: vi.fn(async () => ({
      data: rows.map((row, index) => ({ id: `upload-${index}`, r2_key: row.r2_key })),
      error: null,
    })),
  }));

  fixture.update.mockImplementation((_values: unknown) => {
    let key = "";
    const builder: any = {};
    builder.eq = vi.fn((column: string, value: string) => { if (column === "r2_key") key = value; return builder; });
    builder.select = vi.fn(async () => {
      fixture.dbActive++;
      fixture.dbMax = Math.max(fixture.dbMax, fixture.dbActive);
      await sleep(5);
      fixture.dbActive--;
      return { data: [{ id: "upload" }], error: fixture.dbUpdateErrorKey === key ? { message: "database update failed" } : null };
    });
    return builder;
  });
});

describe("AXO-110 bounded API/R2 I/O", () => {
  test("late invalid upload object creates no partial ledger rows or signed URLs", async () => {
    const response = await worker.fetch(request("/upload-intent", {
      student_id: STUDENT,
      paper_id: PAPER,
      objects: [uploadObject(1), uploadObject(2, { name: "../escape" })],
    }), {} as any);

    expect(response.status).toBe(400);
    expect(fixture.presignPut).not.toHaveBeenCalled();
    expect(fixture.insert).not.toHaveBeenCalled();
  });

  test("upload intent bulk-inserts ledger rows once and bounds URL signing", async () => {
    const objects = Array.from({ length: 24 }, (_, i) => uploadObject(i + 1));
    const response = await worker.fetch(request("/upload-intent", {
      student_id: STUDENT,
      paper_id: PAPER,
      objects,
    }), {} as any);

    expect(response.status).toBe(200);
    const body = await response.json() as any;
    expect(body.objects).toHaveLength(24);
    expect(fixture.insert).toHaveBeenCalledTimes(1);
    expect(fixture.insert.mock.calls[0]![0]).toHaveLength(24);
    expect(fixture.presignMax).toBeLessThanOrEqual(IO_CONCURRENCY);
    expect(fixture.presignMax).toBeGreaterThan(1);
  });

  test("upload intent reports ledger failure instead of returning unsigned authority", async () => {
    fixture.insert.mockImplementationOnce(() => ({
      select: vi.fn(async () => ({ data: null, error: { message: "database unavailable" } })),
    }));

    const response = await worker.fetch(request("/upload-intent", {
      student_id: STUDENT,
      paper_id: PAPER,
      objects: [uploadObject(1)],
    }), {} as any);

    expect(response.status).toBe(500);
    expect(await response.json()).toMatchObject({
      error: "We could not prepare those files for upload. Nothing was uploaded yet.",
    });
  });

  test("upload completion bounds R2 HEAD and database confirmation concurrency", async () => {
    const uploads = Array.from({ length: 24 }, (_, i) => uploadClaim(i + 1));
    const response = await worker.fetch(request("/upload-complete", {
      paper_id: PAPER,
      uploads,
    }), {} as any);

    expect(response.status).toBe(200);
    const body = await response.json() as any;
    expect(body.confirmed).toHaveLength(24);
    expect(body.missing).toEqual([]);
    expect(fixture.headMax).toBeLessThanOrEqual(IO_CONCURRENCY);
    expect(fixture.dbMax).toBeLessThanOrEqual(IO_CONCURRENCY);
    expect(fixture.headMax).toBeGreaterThan(1);
    expect(fixture.dbMax).toBeGreaterThan(1);
  });

  test("upload completion keeps R2 failures explicit and confirms only verified claims", async () => {
    const good = uploadClaim(1);
    const bad = uploadClaim(2);
    fixture.headErrorKey = bad.key;

    const response = await worker.fetch(request("/upload-complete", {
      paper_id: PAPER,
      uploads: [good, bad],
    }), {} as any);

    expect(response.status).toBe(409);
    const body = await response.json() as any;
    expect(body.confirmed).toEqual([good.key]);
    expect(body.missing).toHaveLength(1);
    expect(body.missing[0]).toMatchObject({ key: bad.key });
    expect(body.missing[0].reason).toContain("we could not check that file");
  });

  test("database confirmation failure can never produce a false confirmed response", async () => {
    const uploads = [uploadClaim(1), uploadClaim(2)];
    fixture.dbUpdateErrorKey = uploads[1]!.key;

    const response = await worker.fetch(request("/upload-complete", {
      paper_id: PAPER,
      uploads,
    }), {} as any);

    expect(response.status).toBe(500);
    const body = await response.json() as any;
    expect(body).toMatchObject({ error: "We could not confirm those uploaded files. Try again." });
    expect(body).not.toHaveProperty("confirmed");
  });

  test("upload completion enforces the existing maximum object count before R2 work", async () => {
    const uploads = Array.from({ length: 61 }, (_, i) => uploadClaim(i + 1));
    const response = await worker.fetch(request("/upload-complete", {
      paper_id: PAPER,
      uploads,
    }), {} as any);

    expect(response.status).toBe(400);
    expect(fixture.headObject).not.toHaveBeenCalled();
    expect(fixture.update).not.toHaveBeenCalled();
  });

  test("page and mask signing stays bounded across a large requested page set", async () => {
    fixture.pages = Array.from({ length: 24 }, (_, i) => ({
      page_number: i + 1,
      r2_bucket: "derived",
      student_id: STUDENT,
      paper_id: PAPER,
      r2_key: `${STUDENT}/${PAPER}/page/${i + 1}`,
      mask_key: `${STUDENT}/${PAPER}/mask/${i + 1}`,
    }));

    const response = await worker.fetch(request("/page-asset-urls", {
      paper_id: PAPER,
      page_numbers: fixture.pages.map((page) => page.page_number),
    }), {} as any);

    expect(response.status).toBe(200);
    const body = await response.json() as any;
    expect(Object.keys(body.urls)).toHaveLength(24);
    // Four page pairs at once, with two signatures per pair.
    expect(fixture.signMax).toBeLessThanOrEqual(IO_CONCURRENCY);
    expect(fixture.signMax).toBeGreaterThan(2);
  });

  test("bounded helper preserves order and never exceeds its configured remote limit", async () => {
    let active = 0;
    let maxActive = 0;
    const input = Array.from({ length: 31 }, (_, index) => index);
    const output = await mapLimit(input, 5, async (value) => {
      active++;
      maxActive = Math.max(maxActive, active);
      await sleep(2);
      active--;
      return value * 2;
    });

    expect(output).toEqual(input.map((value) => value * 2));
    expect(maxActive).toBeLessThanOrEqual(5);
    expect(maxActive).toBeGreaterThan(1);
  });

  test("controlled I/O latency fixture records serial versus bounded 1/N/max-object behavior", async () => {
    const results: Array<{ count: number; serialMs: number; boundedMs: number }> = [];
    for (const count of [1, 16, 60]) {
      const items = Array.from({ length: count }, (_, index) => index);
      const serialStart = performance.now();
      for (const item of items) {
        void item;
        await sleep(4);
      }
      const serialMs = performance.now() - serialStart;

      const boundedStart = performance.now();
      await mapLimit(items, IO_CONCURRENCY, async () => { await sleep(4); });
      const boundedMs = performance.now() - boundedStart;
      results.push({ count, serialMs, boundedMs });

      if (count > 1) expect(boundedMs).toBeLessThan(serialMs * 0.7);
    }
    console.info("[AXO-110 controlled I/O benchmark]", JSON.stringify(results));
  });
});

describe("AXO-170/172 storage authority", () => {
  test.each([undefined, 0, -1, 1.5, 25 * 1024 * 1024 + 1])("rejects invalid declared size %s before issuing authority", async bytes => {
    const response = await worker.fetch(request("/upload-intent", { student_id: STUDENT, paper_id: PAPER, objects: [uploadObject(1, { bytes })] }), {} as any);
    expect(response.status).toBe(400);
    expect(fixture.insert).not.toHaveBeenCalled();
    expect(fixture.presignPut).not.toHaveBeenCalled();
  });
  test("rejects an oversized measured object even when the client omits its byte claim", async () => {
    fixture.headObject.mockResolvedValue({ bytes: 25 * 1024 * 1024 + 1, etag: "oversized" });
    const { bytes: _bytes, ...claim } = uploadClaim(1);
    const response = await worker.fetch(request("/upload-complete", { paper_id: PAPER, uploads: [claim] }), {} as any);
    expect(response.status).toBe(409);
    expect(fixture.update).not.toHaveBeenCalled();
    expect(await response.json()).toMatchObject({ confirmed: [] });
  });
  test("requires a server-issued intent before checking an object", async () => {
    fixture.ledger = [];
    const response = await worker.fetch(request("/upload-complete", { paper_id: PAPER, uploads: [uploadClaim(1)] }), {} as any);
    expect(response.status).toBe(409);
    expect(fixture.headObject).not.toHaveBeenCalled();
  });
  test("refuses to sign a victim key stored on an otherwise visible page", async () => {
    fixture.pages = [{ page_number: 1, student_id: STUDENT, paper_id: PAPER, r2_bucket: "derived", r2_key: "victim/paper/page.jpg", mask_key: null }];
    const response = await worker.fetch(request("/page-asset-urls", { paper_id: PAPER, page_numbers: [1] }), {} as any);
    expect(response.status).toBe(403);
    expect(fixture.signAssetUrl).not.toHaveBeenCalled();
  });
  test("returns a signed derived thumb_url per page, and null when a page has no thumb", async () => {
    fixture.pages = [
      { page_number: 1, student_id: STUDENT, paper_id: PAPER, r2_bucket: "derived", r2_key: `${STUDENT}/${PAPER}/page/p1.jpg`, mask_key: null, thumb_key: `${STUDENT}/${PAPER}/thumb/p1-thumb.jpg` },
      { page_number: 2, student_id: STUDENT, paper_id: PAPER, r2_bucket: "derived", r2_key: `${STUDENT}/${PAPER}/page/p2.jpg`, mask_key: null, thumb_key: null },
    ];
    const response = await worker.fetch(request("/page-asset-urls", { paper_id: PAPER, page_numbers: [1, 2] }), {} as any);
    expect(response.status).toBe(200);
    const body = await response.json() as any;
    expect(body.urls["1"].thumb_url).toBe(`https://asset.test/${encodeURIComponent(`${STUDENT}/${PAPER}/thumb/p1-thumb.jpg`)}`);
    expect(body.urls["2"].thumb_url).toBeNull();
    const thumbCalls = fixture.signAssetUrl.mock.calls.filter((call: any[]) => String(call[2]).includes("/thumb/"));
    expect(thumbCalls).toHaveLength(1);
    expect(thumbCalls[0]![1]).toBe("derived");
  });
  test("refuses to sign a victim thumb key stored on an otherwise visible page", async () => {
    fixture.pages = [{ page_number: 1, student_id: STUDENT, paper_id: PAPER, r2_bucket: "derived", r2_key: `${STUDENT}/${PAPER}/page/p1.jpg`, mask_key: null, thumb_key: "victim/paper/thumb.jpg" }];
    const response = await worker.fetch(request("/page-asset-urls", { paper_id: PAPER, page_numbers: [1] }), {} as any);
    expect(response.status).toBe(403);
    expect(fixture.signAssetUrl).not.toHaveBeenCalled();
  });
  test("creates ledger entries for derived pages and masks as well as originals", async () => {
    const response = await worker.fetch(request("/upload-intent", { student_id: STUDENT, paper_id: PAPER, objects: [uploadObject(1, { kind: "page" }), uploadObject(2, { kind: "mask", content_type: "image/png" })] }), {} as any);
    expect(response.status).toBe(200);
    expect(fixture.insert.mock.calls[0]![0]).toHaveLength(2);
  });
});

test("issued PUT capabilities get cleanup after their expiry, and cleanup failure withholds URLs", async () => {
  const before = Date.now();
  const response = await worker.fetch(request("/upload-intent", { student_id: STUDENT, paper_id: PAPER, objects: [uploadObject(1)] }), {} as any, {} as any);
  expect(response.status).toBe(200);
  const row = fixture.cleanupInsert.mock.calls[0][0][0];
  expect(row.key).toBe(uploadClaim(1).key + ".pending");
  expect(new Date(row.not_before).getTime()).toBeGreaterThanOrEqual(before + 20 * 60 * 1000);
  fixture.cleanupInsert.mockResolvedValueOnce({ error: { message: "unavailable" } });
  const failed = await worker.fetch(request("/upload-intent", { student_id: STUDENT, paper_id: PAPER, objects: [uploadObject(2)] }), {} as any, {} as any);
  expect(failed.status).toBe(503);
  expect(await failed.json()).not.toHaveProperty("objects");
});

test("canonical cleanup remains armed for unconfirmed upload intents", async () => {
  const response = await worker.fetch(request("/upload-intent", { student_id: STUDENT, paper_id: PAPER, objects: [uploadObject(1)] }), {} as any, {} as any);
  expect(response.status).toBe(200);
  const rows = fixture.cleanupInsert.mock.calls[0][0];
  expect(rows[1]).toMatchObject({ key: uploadClaim(1).key, unconfirmed_upload_id: "upload-0" });
});
test("erasure during promotion removes the orphan canonical object", async () => {
  fixture.update.mockImplementation(() => {
    const b: any = { eq: () => b, select: async () => ({ data: [], error: null }) };
    return b;
  });
  const response = await worker.fetch(request("/upload-complete", { paper_id: PAPER, uploads: [uploadClaim(1)] }), {} as any, {} as any);
  expect(response.status).toBe(500);
  expect(fixture.deleteObject).toHaveBeenCalledWith(expect.anything(), "originals", uploadClaim(1).key);
  expect(fixture.deleteObject).toHaveBeenCalledWith(expect.anything(), "originals", uploadClaim(1).key + ".pending");
});

describe("AXO-188 upload recovery authority",()=>{
  test("confirmation reads only the requested issued keys and rejects duplicate claims before R2",async()=>{
    const claims=[uploadClaim(1),uploadClaim(2)];
    const response=await worker.fetch(request("/upload-complete",{paper_id:PAPER,uploads:claims}),{} as any);
    expect(response.status).toBe(200);expect(fixture.requestedKeys).toEqual(claims.map(c=>c.key));
    fixture.headObject.mockClear();
    const duplicate=await worker.fetch(request("/upload-complete",{paper_id:PAPER,uploads:[claims[0],claims[0]]}),{} as any);
    expect(duplicate.status).toBe(400);expect(fixture.headObject).not.toHaveBeenCalled();
  });
  test("rollout fails closed and originals cannot exceed the batch cohort",async()=>{
    const off=await worker.fetch(request("/upload-policy",{}),{} as any);expect(await off.json()).toEqual({batch_percent:0,originals_percent:0});
    const clipped=await worker.fetch(request("/upload-policy",{}),{UPLOAD_BATCH_PERCENT:"5",UPLOAD_ORIGINALS_PERCENT:"100"} as any);
    expect(await clipped.json()).toEqual({batch_percent:5,originals_percent:5});
    const invalid=await worker.fetch(request("/upload-policy",{}),{UPLOAD_BATCH_PERCENT:"NaN",UPLOAD_ORIGINALS_PERCENT:"100"} as any);
    expect(await invalid.json()).toEqual({batch_percent:0,originals_percent:0});
  });
  test("legacy grouped page/raw intents get a server-issued capture binding",async()=>{
    const response=await worker.fetch(request("/upload-intent",{student_id:STUDENT,paper_id:PAPER,objects:[
      uploadObject(1,{kind:"page",name:"p1",page_number:1,page_revision:"revision"}),
      uploadObject(1,{kind:"raw",name:"p1-original",page_number:1,page_revision:"revision"}),
    ]}),{} as any);
    expect(response.status).toBe(200);
    const rows=fixture.insert.mock.calls[0][0];
    expect(rows[1].page_key).toBe(rows[0].r2_key);expect(rows[1].asset_kind).toBe("raw");expect(rows[1].page_revision).toBe("revision");
  });
  test("malformed capture metadata and unbound raw uploads are rejected before capabilities",async()=>{
    for(const overrides of [{kind:"raw",name:"p1-original",page_number:1,page_revision:"revision"},
      {kind:"page",name:"p2",page_number:1,page_revision:"revision"},{kind:"raw",name:"p1-original",page_number:1,page_revision:"../bad"}]){
      const response=await worker.fetch(request("/upload-intent",{student_id:STUDENT,paper_id:PAPER,objects:[uploadObject(1,overrides)]}),{} as any);
      expect(response.status).toBeGreaterThanOrEqual(400);
    }
    expect(fixture.insert).not.toHaveBeenCalled();expect(fixture.presignPut).not.toHaveBeenCalled();
  });
  test("original attachment uses user paper scope before service RPC and propagates rejection",async()=>{
    const body={student_id:STUDENT,paper_id:PAPER,pages:[{page_number:1,page_revision:"revision",
      page_key:STUDENT+"/"+PAPER+"/page/p1-x.jpg",original_key:STUDENT+"/"+PAPER+"/raw/p1-original-x.jpg"}]};
    fixture.paperOwned=false;
    expect((await worker.fetch(request("/paper-originals",body),{} as any)).status).toBe(403);
    expect(fixture.rpc).not.toHaveBeenCalled();
    fixture.paperOwned=true;
    expect((await worker.fetch(request("/paper-originals",body),{} as any)).status).toBe(200);
    expect(fixture.rpc).toHaveBeenCalledWith("attach_paper_originals",{p_student_id:STUDENT,p_paper_id:PAPER,p_pages:body.pages});
    fixture.rpc.mockResolvedValue({error:{code:"42501"},data:null});
    expect((await worker.fetch(request("/paper-originals",body),{} as any)).status).toBe(409);
  });
  test("foreign keys and duplicate pages never reach original attachment RPC",async()=>{
    const page={page_number:1,page_revision:"revision",page_key:"other/"+PAPER+"/page/p1-x.jpg",original_key:STUDENT+"/"+PAPER+"/raw/p1-original-x.jpg"};
    expect((await worker.fetch(request("/paper-originals",{student_id:STUDENT,paper_id:PAPER,pages:[page]}),{} as any)).status).toBe(400);
    page.page_key=STUDENT+"/"+PAPER+"/page/p1-x.jpg";
    expect((await worker.fetch(request("/paper-originals",{student_id:STUDENT,paper_id:PAPER,pages:[page,page]}),{} as any)).status).toBe(400);
    expect(fixture.rpc).not.toHaveBeenCalled();
  });
});

test("legacy confirmed page can bind a missing original without granting arbitrary revision aliases",async()=>{
  const key=STUDENT+"/"+PAPER+"/page/p1-old.jpg";
  fixture.ledger=[{r2_key:key,asset_kind:null}];
  const body={student_id:STUDENT,paper_id:PAPER,objects:[uploadObject(1,{kind:"raw",name:"p1-original",page_number:1,page_revision:"legacy-1",page_key:key})]};
  expect((await worker.fetch(request("/upload-intent",body),{} as any)).status).toBe(200);
  expect(fixture.insert.mock.calls[0][0][0].page_key).toBe(key);
  fixture.insert.mockClear();body.objects[0].page_revision="new-capture";
  expect((await worker.fetch(request("/upload-intent",body),{} as any)).status).toBe(409);
  expect(fixture.insert).not.toHaveBeenCalled();
});
