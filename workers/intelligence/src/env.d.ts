// Keep this declaration complete and checked in: Wrangler's generated
// worker-configuration.d.ts is deliberately ignored, so clean CI installs cannot
// rely on it. Secrets remain absent from wrangler.jsonc.
interface IntelligenceBindings {
  DB: D1Database;
  PIPELINE_CACHE: KVNamespace;
  PAPER_ARTIFACTS: R2Bucket;
  PAPER_QUEUE: Queue;
  AXON_DEPLOYMENT_SHA: string;
  AXON_CONFIG_REVISION: string;
  AXON_PIPELINE_VERSION: string;
  GEMINI_PRIVACY_MODE: string;
  TAVILY_API_BASE: string;
  SUPABASE_URL: string;
  GOOGLE_API_KEY: string;
  SUPABASE_SERVICE_ROLE_KEY: string;
  AXON_INTERNAL_TOKEN: string;
  AXON_ADMIN_TOKEN: string;
  TAVILY_API_KEY?: string;
  AXON_VISION_API_BASE?: string;
  AXON_VISION_TOKEN?: string;
  AXON_VISION_PRIVACY_MODE?: string;
  AXON_SHADOW_MODEL?: string;
  AXON_SHADOW_CONFIG_REVISION?: string;
  AXON_PSEUDONYM_KEY: string;
  AXON_RELEASE_CERTIFIED?: string;
  GEMINI_INPUT_USD_PER_MILLION?: string;
  GEMINI_OUTPUT_USD_PER_MILLION?: string;
  TEST_MIGRATIONS?: Array<{ name: string; queries: string[] }>;
}

interface Env extends IntelligenceBindings {}

declare namespace Cloudflare {
  interface GlobalProps {
    mainModule: typeof import("./index");
  }
  interface Env extends IntelligenceBindings {}
}
