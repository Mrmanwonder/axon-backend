import { decodeImage, encodeWebp } from "./codecs";
import { analyzeDocument } from "./core";
import { handleRequestWith } from "./handler";
import { GeminiTargetedRegionReader, MoondreamPageLayoutReader } from "./readers";

async function handleRequest(request: Request, env: Env): Promise<Response> {
  return handleRequestWith(request, env, (input) => analyzeDocument(
    input,
    new MoondreamPageLayoutReader(env.AI),
    new GeminiTargetedRegionReader(env.GOOGLE_API_KEY, env.GEMINI_MODEL),
    decodeImage,
    encodeWebp
  ));
}

export default { fetch: handleRequest } satisfies ExportedHandler<Env>;
