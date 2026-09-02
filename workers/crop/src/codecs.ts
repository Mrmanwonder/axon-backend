// Decoding and encoding images inside a Worker.
//
// ── why the codecs are reached at this depth ──────────────────────────────
//
// jSquash's own `decode`/`encode` entry points cannot be used here, and it is
// not a packaging problem, it is a Workers one:
//
// · `@jsquash/webp/encode.js` calls `simd()` from `wasm-feature-detect`, which
//   compiles a small probe module from bytes at runtime. Workers forbid
//   `WebAssembly.compile` on dynamic bytes — only modules the bundler compiled
//   ahead of time can be instantiated — so that call throws before any image is
//   touched. The non-SIMD encoder is imported directly instead, which is the
//   same encoder that branch would have chosen anyway.
//
// · The decoders would otherwise try to locate their `.wasm` beside themselves
//   and fetch it. Passing the module in explicitly is exactly what jSquash's
//   `initEmscriptenModule` was modified upstream to support, and what wrangler's
//   `.wasm` imports produce.
//
// Everything is instantiated once per isolate and kept, because instantiation
// is the expensive part and a Worker handling a booklet will be asked for
// several pages in a row.

import webpDecFactory from "@jsquash/webp/codec/dec/webp_dec.js";
import webpEncFactory from "@jsquash/webp/codec/enc/webp_enc.js";
import jpegDecFactory from "@jsquash/jpeg/codec/dec/mozjpeg_dec.js";
import { defaultOptions as WEBP_DEFAULTS } from "@jsquash/webp/meta.js";
import initPng, { decode as pngDecode } from "@jsquash/png/codec/pkg/squoosh_png.js";

import WEBP_DEC_WASM from "@jsquash/webp/codec/dec/webp_dec.wasm";
import WEBP_ENC_WASM from "@jsquash/webp/codec/enc/webp_enc.wasm";
import JPEG_DEC_WASM from "@jsquash/jpeg/codec/dec/mozjpeg_dec.wasm";
// The other three `.wasm` imports pick up the wildcard declaration in
// wasm.d.ts. This one does not: @jsquash/png ships a real `.d.ts` beside its
// wasm describing the module's raw wasm-bindgen exports (`memory`, `decode`,
// ...), which resolves first and has no default. What wrangler actually binds
// is a `WebAssembly.Module`, the same as the other three, so the runtime is
// right and only the type is wrong.
// @ts-expect-error — see above: the shipped .d.ts describes the wasm's exports, not wrangler's module binding.
import PNG_WASM_RAW from "@jsquash/png/codec/pkg/squoosh_png_bg.wasm";
const PNG_WASM = PNG_WASM_RAW as unknown as WebAssembly.Module;

import { imageFormat, type RgbaImage } from "@mastery/shared/crop.js";

/** jSquash's own helper, inlined: it is nine lines and importing it would mean
    declaring types for one more untyped deep path. `instantiateWasm` is how an
    emscripten module is handed a `WebAssembly.Module` instead of being left to
    find its own. */
function instantiate<T>(factory: (options?: Record<string, unknown>) => Promise<T>, wasm: WebAssembly.Module): Promise<T> {
  return factory({
    noInitialRun: true,
    instantiateWasm: (imports: WebAssembly.Imports, callback: (instance: WebAssembly.Instance) => void) => {
      const instance = new WebAssembly.Instance(wasm, imports);
      callback(instance);
      return instance.exports;
    },
  });
}

let webpDec: ReturnType<typeof instantiate<Awaited<ReturnType<typeof webpDecFactory>>>> | null = null;
let webpEnc: ReturnType<typeof instantiate<Awaited<ReturnType<typeof webpEncFactory>>>> | null = null;
let jpegDec: ReturnType<typeof instantiate<Awaited<ReturnType<typeof jpegDecFactory>>>> | null = null;
let pngReady: Promise<unknown> | null = null;

/**
 * Quality for a stored crop.
 *
 * Higher than the page's own 0.92-equivalent would suggest is needed, and
 * deliberately so: this is a second lossy generation on top of the page's
 * first, and the thing that has to survive two of them is a thin red pen
 * stroke. 90 in libwebp's scale, with the encoder left in its default
 * (lossy, no alpha work) configuration.
 */
const CROP_QUALITY = 90;

/** Decode whatever the bytes turn out to be. Format is sniffed rather than
    taken from a column: the page is WebP or JPEG depending on what the
    student's browser could write, and a mismatch would be a decode error with
    nothing useful to say about it. */
export async function decodeImage(bytes: Uint8Array): Promise<RgbaImage> {
  const format = imageFormat(bytes);
  if (!format) throw new Error("crop: unrecognised image format");

  if (format === "webp") {
    webpDec ??= instantiate(webpDecFactory, WEBP_DEC_WASM);
    const decoded = (await webpDec).decode(bytes);
    if (!decoded) throw new Error("crop: webp decode returned nothing");
    return decoded;
  }

  if (format === "jpeg") {
    jpegDec ??= instantiate(jpegDecFactory, JPEG_DEC_WASM);
    const decoded = (await jpegDec).decode(bytes, false);
    if (!decoded) throw new Error("crop: jpeg decode returned nothing");
    return decoded;
  }

  pngReady ??= initPng(PNG_WASM);
  await pngReady;
  const decoded = pngDecode(bytes);
  if (!decoded) throw new Error("crop: png decode returned nothing");
  return decoded;
}

/** WebP, always, whatever the page was. One format downstream means one thing
    for `imageRef` to serve and one thing for the model to receive. */
export async function encodeWebp(image: RgbaImage): Promise<Uint8Array> {
  webpEnc ??= instantiate(webpEncFactory, WEBP_ENC_WASM);
  // The whole options struct, not just the two fields that differ from the
  // defaults. libwebp's embind binding requires every field to be present and
  // throws `Missing field: "lossless"` on a partial object — a failure that
  // only appears at the moment a real image is encoded, which is why it was
  // found by running the codec rather than by building it. jSquash's own
  // `encode.js` merges these the same way; this file cannot use that entry
  // point (see the note at the top) but it can use its defaults.
  const encoded = (await webpEnc).encode(image.data, image.width, image.height, {
    ...WEBP_DEFAULTS,
    quality: CROP_QUALITY,
    method: 4,
    // Nothing else is touched. The encoder has thirty knobs and tuning them
    // against no measurement is how the old build ended up walking JPEG
    // quality downward while measuring how much red survived.
  });
  if (!encoded) throw new Error("crop: webp encode returned nothing");
  return encoded;
}
