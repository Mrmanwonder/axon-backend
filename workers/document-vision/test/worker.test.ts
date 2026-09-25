import { describe, expect, it } from "vitest";
import { Value } from "@sinclair/typebox/value";
import type { RgbaImage } from "@mastery/shared/crop.js";
import { VisionAnalysisSchema as ConsumerVisionAnalysisSchema } from "../../intelligence/src/document/vision/provider";
import { analyzeDocument } from "../src/core";
import { handleRequestWith } from "../src/handler";
import type { ReaderOutput } from "../src/schema";
import type { VisionReader } from "../src/readers";

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

const readerOutput: ReaderOutput = {
  orientationDegrees: 0,
  perspectiveDegrees: 0,
  cropCompleteness: 1,
  regions: [{ class: "question_number", box: { x: 0.1, y: 0.1, width: 0.1, height: 0.1 }, confidence: 0.98, text: "1", layer: "PRINTED" }]
};

const reader = (id: string): VisionReader => ({ id, read: () => Promise.resolve(readerOutput) });

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

  it("runs the private capability probe through the production analysis callback", async () => {
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
      return Promise.resolve({
        qualityMetrics: { blur: 0, glareFraction: 0, perspectiveDegrees: 0, resolution: 1, compression: 0, cropCompleteness: 1, shadowFraction: 0 },
        orientationDegrees: 0,
        regions: [],
        reads: []
      });
    });
    expect(inputPageId).toBe("synthetic-capability-probe");
    expect(response.status).toBe(200);
    await expect(response.json()).resolves.toEqual({
      status: "passed",
      contractVersion: "axon-document-vision.v1",
      version: "vision-test",
      regionCount: 0,
      readGroupCount: 0
    });
  });

  it("builds a consumer-valid analysis from two readers", async () => {
    const decoded: RgbaImage = { data: new Uint8ClampedArray(4 * 4 * 4).fill(240), width: 4, height: 4 };
    const analysis = await analyzeDocument({
      contractVersion: "axon-document-vision.v1",
      pageId: "page-1",
      mimeType: "image/png",
      dataBase64: asBase64(pngHeader(4, 4))
    }, [reader("gemini"), reader("moondream")], () => Promise.resolve(decoded), () => Promise.resolve(new Uint8Array([1])));
    expect(analysis.reads[0]?.reads).toHaveLength(2);
    expect(Value.Check(ConsumerVisionAnalysisSchema, analysis)).toBe(true);
  });

  it("rejects a declared type that does not match the bytes", async () => {
    await expect(analyzeDocument({
      contractVersion: "axon-document-vision.v1",
      pageId: "page-1",
      mimeType: "image/jpeg",
      dataBase64: asBase64(pngHeader(4, 4))
    }, [reader("a"), reader("b")], () => Promise.reject(new Error("must not decode")), () => Promise.reject(new Error("must not encode")))).rejects.toThrow("MIME_TYPE_MISMATCH");
  });
});
