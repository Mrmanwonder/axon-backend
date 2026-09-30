import { beforeEach, describe, expect, test, vi } from "vitest";

const fixture = vi.hoisted(() => ({
  clientFor: vi.fn(),
  serviceClient: vi.fn(),
  presignPut: vi.fn(),
  headObject: vi.fn(),
  signAssetUrl: vi.fn(),
  insert: vi.fn(),
  update: vi.fn(),
  pages: [] as any[],
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
  headObject: fixture.headObject,
  signAssetUrl: fixture.signAssetUrl,
  verifyAssetSignature: vi.fn(),
  objectKey: ({ studentId, paperId, kind, name, extension }: any) =>
    `${studentId}/${paperId}/${kind}/${name}.${extension}`,
  BUCKET_FOR: {
    upload: "originals",
    raw: "originals",
    page: "derived",
    crop: "derived",
    mask: "derived",
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
    from: vi.fn((table: string) => {
      if (table === "paper") {
        const builder: any = {};
        builder.select = vi.fn(() => builder);
        builder.eq = vi.fn(() => builder);
        builder.maybeSingle = vi.fn(async () => ({
          data: { id: PAPER, student_id: STUDENT },
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
    if (table !== "upload") throw new Error(`unexpected admin table ${table}`);
    return {
      insert: fixture.insert,
      update: fixture.update,
    };
  });
  return { from };
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
  fixture.dbUpdateErrorKey = null;
  fixture.headErrorKey = null;
  fixture.presignActive = fixture.presignMax = 0;
  fixture.headActive = fixture.headMax = 0;
  fixture.dbActive = fixture.dbMax = 0;
  fixture.signActive = fixture.signMax = 0;

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
    let eqCount = 0;
    const builder: any = {};
    builder.eq = vi.fn((_column: string, value: string) => {
      eqCount++;
      if (eqCount === 1) return builder;
      return (async () => {
        fixture.dbActive++;
        fixture.dbMax = Math.max(fixture.dbMax, fixture.dbActive);
        await sleep(5);
        fixture.dbActive--;
        return {
          error: fixture.dbUpdateErrorKey === value
            ? { message: "database update failed" }
            : null,
        };
      })();
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
      r2_key: `page-${i + 1}`,
      mask_key: `mask-${i + 1}`,
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
