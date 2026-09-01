// The jSquash packages ship types for their entry points but not for the deep
// codec paths, and nothing types a `.wasm` import at all. Both are declared
// here rather than suppressed at each import with a comment that would say
// less.
//
// Wrangler compiles an imported `.wasm` file into a `WebAssembly.Module` and
// binds it as a module — no fetch, no filesystem — which is what lets the
// emscripten and wasm-bindgen glue below be instantiated inside a Worker at
// all. See src/index.ts for why the codecs are reached at this depth instead of
// through the packages' own `decode`/`encode` entry points.
declare module "*.wasm" {
  const module: WebAssembly.Module;
  export default module;
}

declare module "@jsquash/webp/codec/dec/webp_dec.js" {
  const factory: (options?: Record<string, unknown>) => Promise<{
    decode(buffer: ArrayBuffer | Uint8Array): { data: Uint8ClampedArray; width: number; height: number } | null;
  }>;
  export default factory;
}

declare module "@jsquash/webp/codec/enc/webp_enc.js" {
  const factory: (options?: Record<string, unknown>) => Promise<{
    encode(
      data: Uint8Array | Uint8ClampedArray,
      width: number,
      height: number,
      options: Record<string, unknown>,
    ): Uint8Array | null;
  }>;
  export default factory;
}

declare module "@jsquash/jpeg/codec/dec/mozjpeg_dec.js" {
  const factory: (options?: Record<string, unknown>) => Promise<{
    decode(buffer: ArrayBuffer | Uint8Array, preserveOrientation?: boolean): { data: Uint8ClampedArray; width: number; height: number } | null;
  }>;
  export default factory;
}

declare module "@jsquash/webp/meta.js" {
  export const defaultOptions: Record<string, unknown>;
}

declare module "@jsquash/png/codec/pkg/squoosh_png.js" {
  export default function init(module?: WebAssembly.Module): Promise<unknown>;
  export function decode(data: Uint8Array): { data: Uint8ClampedArray; width: number; height: number } | null;
}
