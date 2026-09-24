// Classic Worker secrets are intentionally absent from wrangler.jsonc and therefore
// cannot be emitted by `wrangler types`. This declaration augments only secret keys;
// every configured binding remains generated from wrangler.jsonc.
interface Env {
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

declare namespace Cloudflare {
  interface Env {
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
}
