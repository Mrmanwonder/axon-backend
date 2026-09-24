import { decodeImage, encodeWebp } from "./codecs";
import { analyzeDocument } from "./core";
import { handleRequestWith } from "./handler";
import { GeminiVisionReader, MoondreamVisionReader } from "./readers";

async function handleRequest(request: Request, env: Env): Promise<Response> {
  return handleRequestWith(request, env, (input) => analyzeDocument(input, [
    new GeminiVisionReader(env.GOOGLE_API_KEY, env.GEMINI_MODEL),
    new MoondreamVisionReader(env.AI)
  ], decodeImage, encodeWebp));
}

export default { fetch: handleRequest } satisfies ExportedHandler<Env>;
