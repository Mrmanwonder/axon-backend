// A runtime check of the codec path, run by hand:  npm run smoke -w @mastery/crop
//
// It is not part of `npm test`, because it needs the fixtures that live in the
// Axon-Site repo rather than this one. It is checked in anyway, because it
// earned its place: `wrangler deploy --dry-run` proved the worker *bundles*,
// and the bundle was still wrong — libwebp's embind binding requires the whole
// options struct and throws `Missing field: "lossless"` on a partial one, a
// failure that appears only at the moment a real image is encoded. Nothing
// short of actually running the codec would have found it before production
// did.
//
// What it checks, in order: the format sniff, a real 2250x3301 submitted page
// decoding, the band coming out full-width, the cut being *byte-identical* to
// the source rows (a transposed or offset copy would still look like it
// worked), the WebP encode, and the round trip back through the decoder.
import { readFile, writeFile } from 'node:fs/promises';
import webpDecFactory from '@jsquash/webp/codec/dec/webp_dec.js';
import webpEncFactory from '@jsquash/webp/codec/enc/webp_enc.js';
import initPng, { decode as pngDecode } from '@jsquash/png/codec/pkg/squoosh_png.js';
import jpegDecFactory from '@jsquash/jpeg/codec/dec/mozjpeg_dec.js';
import { defaultOptions as WEBP_DEFAULTS } from '@jsquash/webp/meta.js';
import { bandForRegion, cutRegion, imageFormat } from '@mastery/shared/crop.js';

// The same instantiation the Worker uses: hand the codec a compiled
// WebAssembly.Module rather than letting it go looking for its own .wasm.
// Without this the emscripten glue calls fetch() — which is what a Worker
// forbids, and what Node cannot do for a file: URL either.
async function instantiate(factory, wasmPath) {
  const wasm = await WebAssembly.compile(await readFile(new URL(wasmPath, import.meta.url)));
  return factory({
    noInitialRun: true,
    instantiateWasm: (imports, callback) => {
      const instance = new WebAssembly.Instance(wasm, imports);
      callback(instance);
      return instance.exports;
    },
  });
}

const FIX = '/home/user/Axon-Site/bench/fixtures/';

async function main() {
  // 1. PNG: the real production-scale submitted page.
  const pngBytes = new Uint8Array(await readFile(FIX + 'glare-blown-background-2.png'));
  console.log('sniff png ->', imageFormat(pngBytes));
  await initPng(await readFile(new URL('../../node_modules/@jsquash/png/codec/pkg/squoosh_png_bg.wasm', import.meta.url)));
  const page = pngDecode(pngBytes);
  console.log('decoded page', page.width + 'x' + page.height);

  // 2. A band, the way the worker would compute one for a mid-page question.
  const band = bandForRegion(
    [{ page: 1, box: { x: 300, y: Math.round(page.height * 0.35), w: Math.round(page.width * 0.7), h: Math.round(page.height * 0.18) } }],
    1, page.width, page.height,
  );
  console.log('band', JSON.stringify(band), '-> full width?', band.x === 0 && band.w === page.width);

  const cut = cutRegion(page, band);
  console.log('cut', cut.width + 'x' + cut.height, 'pixels', (cut.width * cut.height / 1e6).toFixed(2) + 'M',
              'vs page', (page.width * page.height / 1e6).toFixed(2) + 'M',
              '->', (page.width * page.height / (cut.width * cut.height)).toFixed(1) + 'x fewer');

  // The copy must be exact: compare a row of the cut against the source row it
  // came from. A transposed or offset copy would still "work" and be wrong.
  const row = 17;
  const srcOffset = ((band.y + row) * page.width + band.x) * 4;
  const same = Buffer.compare(
    Buffer.from(cut.data.buffer, cut.data.byteOffset + row * cut.width * 4, cut.width * 4),
    Buffer.from(page.data.buffer, page.data.byteOffset + srcOffset, cut.width * 4),
  ) === 0;
  console.log('row', row, 'is byte-identical to its source row:', same);
  if (!same) process.exitCode = 1;

  // 3. Encode as WebP, the non-SIMD encoder the Worker is pinned to.
  const enc = await instantiate(webpEncFactory, '../../node_modules/@jsquash/webp/codec/enc/webp_enc.wasm');
  const encoded = enc.encode(cut.data, cut.width, cut.height, { ...WEBP_DEFAULTS, quality: 90, method: 4 });
  console.log('encoded webp bytes', encoded.length, '(' + (encoded.length / 1024).toFixed(0) + ' KiB)');
  await writeFile('/tmp/crop-smoke.webp', Buffer.from(encoded));

  // 4. Round-trip it back through the decoder the worker uses.
  const dec = await instantiate(webpDecFactory, '../../node_modules/@jsquash/webp/codec/dec/webp_dec.wasm');
  const back = dec.decode(encoded.buffer.slice(encoded.byteOffset, encoded.byteOffset + encoded.length));
  console.log('re-decoded', back.width + 'x' + back.height,
              'matches cut:', back.width === cut.width && back.height === cut.height);
  console.log('sniff encoded ->', imageFormat(new Uint8Array(encoded)));

  // 5. JPEG decoder, since a page can be either.
  const jpegBytes = new Uint8Array(await readFile(FIX + 'page-tilted.jpg'));
  const jdec = await instantiate(jpegDecFactory, '../../node_modules/@jsquash/jpeg/codec/dec/mozjpeg_dec.wasm');
  const jpg = jdec.decode(jpegBytes.buffer.slice(jpegBytes.byteOffset, jpegBytes.byteOffset + jpegBytes.length), false);
  console.log('sniff jpeg ->', imageFormat(jpegBytes), '| decoded', jpg.width + 'x' + jpg.height);
}
main();
