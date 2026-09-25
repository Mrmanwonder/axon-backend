import webpDecFactory from "@jsquash/webp/codec/dec/webp_dec.js";
import webpEncFactory from "@jsquash/webp/codec/enc/webp_enc.js";
import jpegDecFactory from "@jsquash/jpeg/codec/dec/mozjpeg_dec.js";
import { defaultOptions as WEBP_DEFAULTS } from "@jsquash/webp/meta.js";
import initPng, { decode as pngDecode } from "@jsquash/png/codec/pkg/squoosh_png.js";
import WEBP_DEC_WASM from "@jsquash/webp/codec/dec/webp_dec.wasm";
import WEBP_ENC_WASM from "@jsquash/webp/codec/enc/webp_enc.wasm";
import JPEG_DEC_WASM from "@jsquash/jpeg/codec/dec/mozjpeg_dec.wasm";
// @ts-expect-error The package describes raw wasm-bindgen exports; Wrangler binds a WebAssembly.Module.
import PNG_WASM_RAW from "@jsquash/png/codec/pkg/squoosh_png_bg.wasm";
import { imageFormat, type RgbaImage } from "@mastery/shared/crop.js";

const PNG_WASM = PNG_WASM_RAW as unknown as WebAssembly.Module;

function instantiate<T>(factory: (options?: Record<string, unknown>) => Promise<T>, wasm: WebAssembly.Module): Promise<T> {
  return factory({
    noInitialRun: true,
    instantiateWasm: (imports: WebAssembly.Imports, callback: (instance: WebAssembly.Instance) => void) => {
      const instance = new WebAssembly.Instance(wasm, imports);
      callback(instance);
      return instance.exports;
    }
  });
}

let webpDec: ReturnType<typeof instantiate<Awaited<ReturnType<typeof webpDecFactory>>>> | null = null;
let webpEnc: ReturnType<typeof instantiate<Awaited<ReturnType<typeof webpEncFactory>>>> | null = null;
let jpegDec: ReturnType<typeof instantiate<Awaited<ReturnType<typeof jpegDecFactory>>>> | null = null;
let pngReady: Promise<unknown> | null = null;

export async function decodeImage(bytes: Uint8Array): Promise<RgbaImage> {
  const format = imageFormat(bytes);
  if (!format) throw new Error("UNSUPPORTED_IMAGE_CODEC");
  if (format === "webp") {
    webpDec ??= instantiate(webpDecFactory, WEBP_DEC_WASM);
    const decoded = (await webpDec).decode(bytes);
    if (!decoded) throw new Error("IMAGE_DECODE_FAILED");
    return decoded;
  }
  if (format === "jpeg") {
    jpegDec ??= instantiate(jpegDecFactory, JPEG_DEC_WASM);
    const decoded = (await jpegDec).decode(bytes, false);
    if (!decoded) throw new Error("IMAGE_DECODE_FAILED");
    return decoded;
  }
  pngReady ??= initPng(PNG_WASM);
  await pngReady;
  const decoded = pngDecode(bytes);
  if (!decoded) throw new Error("IMAGE_DECODE_FAILED");
  return decoded;
}

export async function encodeWebp(image: RgbaImage): Promise<Uint8Array> {
  webpEnc ??= instantiate(webpEncFactory, WEBP_ENC_WASM);
  const encoded = (await webpEnc).encode(image.data, image.width, image.height, {
    ...WEBP_DEFAULTS,
    quality: 92,
    method: 4
  });
  if (!encoded) throw new Error("IMAGE_ENCODE_FAILED");
  return encoded;
}
