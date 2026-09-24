export async function readBoundedBytes(body: ReadableStream<Uint8Array> | null, maximumBytes: number): Promise<ArrayBuffer> {
  if (!body) throw new Error("Missing request body");
  const reader = body.getReader();
  const chunks: Uint8Array[] = [];
  let total = 0;
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      total += value.byteLength;
      if (total > maximumBytes) throw new Error(`Request body exceeds ${maximumBytes} bytes`);
      chunks.push(value);
    }
  } finally {
    reader.releaseLock();
  }
  if (total === 0) throw new Error("Missing request body");
  const output = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) { output.set(chunk, offset); offset += chunk.byteLength; }
  return output.buffer;
}
