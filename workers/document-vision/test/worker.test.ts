import { describe, expect, it } from "vitest";
import { Value } from "@sinclair/typebox/value";
import type { RgbaImage } from "@mastery/shared/crop.js";
import { VisionAnalysisSchema as ConsumerVisionAnalysisSchema } from "../../intelligence/src/document/vision/provider";
import { analyzeDocument } from "../src/core";
import { handleRequestWith } from "../src/handler";
import type { PageLayoutReader, TargetedRegionContext, TargetedRegionReader } from "../src/readers";
import type { ReaderOutput, TargetedRegionRead, VisionAnalysis } from "../src/schema";

function pngHeader(width: number, height: number): Uint8Array {
  const bytes = new Uint8Array(24);
  bytes.set([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a], 0);
  bytes.set([0x49, 0x48, 0x44, 0x52], 12);
  bytes[16] = (width >>> 24) & 0xff;
  bytes[17] = (width >>> 16) & 0xff;
  bytes[18] = (width >>> 8) & 0xff;
  bytes[19] = width & 0xff;
  bytes[20] = (height >>> 24) & 0xff;
  bytes[21] = (height >>> 16) & 0xff;
  bytes[22] = (height >>> 8) & 0xff;
  bytes[23] = height & 0xff;
  return bytes;
}

function asBase64(bytes: Uint8Array): string {
  let binary = "";
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary);
}

const layoutOutput: ReaderOutput = {
  orientationDegrees: 0,
  perspectiveDegrees: 0,
  cropCompleteness: 1,
  regions: [
    { class: "question_number", box: { x: 0.1, y: 0.1, width: 0.1, height: 0.1 }, confidence: 0.98, text: "1", layer: "PRINTED" },
    { class: "diagram", box: { x: 0.4, y: 0.4, width: 0.2, height: 0.2 }, confidence: 0.9, text: null, layer: "PRINTED" }
  ]
};

const targetedOutput: TargetedRegionRead = {
  class: "question_number",
  layer: "PRINTED",
  confidence: 0.97,
  value: "1",
  alternatives: [],
  status: "read"
};

function probeAnalysis(readerIds: string[] = ["layout", "targeted"]): VisionAnalysis {
  const box = { x: 0.1, y: 0.1, width: 0.1, height: 0.1 };
  return {
    qualityMetrics: { blur: 0, glareFraction: 0, perspectiveDegrees: 0, resolution: 1, compression: 0, cropCompleteness: 1, shadowFraction: 0 },
    orientationDegrees: 0,
    regions: [{ id: "r-0001", class: "question_number", box, confidence: 0.95, text: "1" }],
    reads: [{ regionId: "r-0001", reads: readerIds.map((readerId) => ({ value: "1", alternatives: [], status: "read", region: box, readerIds: [readerId] })) }]
  };
}

describe("document-vision Worker", () => {
  it("fails closed before parsing or inference when privacy is unattested", async () => {
    const env = { GEMINI_PRIVACY_MODE: "unverified", WORKERS_AI_PRIVACY_MODE: "unverified", AXON_VISION_VERSION: "test" } as unknown as Env;
    const response = await handleRequestWith(new Request("https://vision.test/v1/analyze", {
      method: "POST",
      headers: { "x-axon-contract-version": "axon-document-vision.v1" },
      body: "not-json"
    }), env, () => Promise.reject(new Error("must not run")));
    expect(response.status).toBe(503);
    await expect(response.json()).resolves.toEqual({ error: "PRIVACY_ATTESTATION_REQUIRED" });
  });

  it("does not report ready when privacy flags exist but the Gemini secret is absent", async () => {
    const env = { GEMINI_PRIVACY_MODE: "zdr", WORKERS_AI_PRIVACY_MODE: "zdr", AXON_VISION_VERSION: "test", AI: {} } as unknown as Env;
    const response = await handleRequestWith(new Request("https://vision.test/health"), env, () => Promise.reject(new Error("must not run")));
    expect(response.status).toBe(503);
    await expect(response.json()).resolves.toMatchObject({ status: "not_ready", privacyReady: false });
  });

  it("requires the capability probe to prove both independent readers ran", async () => {
    const env = {
      GEMINI_PRIVACY_MODE: "zdr",
      WORKERS_AI_PRIVACY_MODE: "zdr",
      AXON_VISION_VERSION: "vision-test",
      GOOGLE_API_KEY: "test-key",
      AI: {}
    } as unknown as Env;
    let inputPageId: string | undefined;
    const response = await handleRequestWith(new Request("https://vision.test/v1/probe", {
      method: "POST",
      headers: { "x-axon-contract-version": "axon-document-vision.v1" }
    }), env, (input) => {
      inputPageId = input.pageId;
      return Promise.resolve(probeAnalysis());
    });
    expect(inputPageId).toBe("synthetic-capability-probe");
    expect(response.status).toBe(200);
    await expect(response.json()).resolves.toEqual({
      status: "passed",
      contractVersion: "axon-document-vision.v1",
      version: "vision-test",
      regionCount: 1,
      readGroupCount: 1,
      readerCount: 2
    });

    const failed = await handleRequestWith(new Request("https://vision.test/v1/probe", {
      method: "POST",
      headers: { "x-axon-contract-version": "axon-document-vision.v1" }
    }), env, () => Promise.resolve(probeAnalysis(["layout"])));
    expect(failed.status).toBe(502);
    await expect(failed.json()).resolves.toEqual({ status: "failed", error: "VISION_READER_FAILURE" });
  });

  it("discovers the full page once and sends only readable crops to the targeted reader", async () => {
    const decoded: RgbaImage = { data: new Uint8ClampedArray(100 * 100 * 4).fill(240), width: 100, height: 100 };
    const pageUrls: string[] = [];
    const cropUrls: string[] = [];
    const contexts: TargetedRegionContext[] = [];
    const encodedSizes: Array<{ width: number; height: number }> = [];
    const layoutReader: PageLayoutReader = {
      id: "layout",
      discoverPage: (dataUrl) => { pageUrls.push(dataUrl); return Promise.resolve(layoutOutput); }
    };
    const targetedReader: TargetedRegionReader = {
      id: "targeted",
      readRegion: (dataUrl, context) => { cropUrls.push(dataUrl); contexts.push(context); return Promise.resolve(targetedOutput); }
    };
    const pageBase64 = asBase64(pngHeader(100, 100));
    const analysis = await analyzeDocument({
      contractVersion: "axon-document-vision.v1",
      pageId: "page-1",
      mimeType: "image/png",
      dataBase64: pageBase64
    }, layoutReader, targetedReader, () => Promise.resolve(decoded), (image) => {
      encodedSizes.push({ width: image.width, height: image.height });
      return Promise.resolve(new Uint8Array([1, 2, 3]));
    });
    expect(pageUrls).toEqual([`data:image/png;base64,${pageBase64}`]);
    expect(cropUrls).toEqual(["data:image/webp;base64,AQID"]);
    expect(encodedSizes).toHaveLength(1);
    expect(encodedSizes[0].width).toBeLessThan(decoded.width);
    expect(encodedSizes[0].height).toBeLessThan(decoded.height);
    expect(contexts[0]).toMatchObject({ expectedClass: "question_number", expectedLayer: "PRINTED", pageBox: layoutOutput.regions[0].box });
    expect(analysis.regions).toHaveLength(2);
    expect(analysis.reads[0]?.reads.map((read) => read.readerIds[0])).toEqual(["layout", "targeted"]);
    expect(Value.Check(ConsumerVisionAnalysisSchema, analysis)).toBe(true);
  });

  it("rejects a declared type that does not match the bytes before either reader runs", async () => {
    const layoutReader: PageLayoutReader = { id: "layout", discoverPage: () => Promise.reject(new Error("must not run")) };
    const targetedReader: TargetedRegionReader = { id: "targeted", readRegion: () => Promise.reject(new Error("must not run")) };
    await expect(analyzeDocument({
      contractVersion: "axon-document-vision.v1",
      pageId: "page-1",
      mimeType: "image/jpeg",
      dataBase64: asBase64(pngHeader(4, 4))
    }, layoutReader, targetedReader, () => Promise.reject(new Error("must not decode")), () => Promise.reject(new Error("must not encode")))).rejects.toThrow("MIME_TYPE_MISMATCH");
  });
});
