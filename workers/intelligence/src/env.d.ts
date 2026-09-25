// Non-secret bindings and variables are generated from wrangler.jsonc into
// generated-bindings.d.ts. Only secret or deployment-only bindings belong here.
interface IntelligenceSecretBindings {
  GOOGLE_API_KEY: string;
  SUPABASE_SERVICE_ROLE_KEY: string;
  AXON_INTERNAL_TOKEN: string;
  AXON_ADMIN_TOKEN: string;
  TAVILY_API_KEY?: string;
  AXON_SHADOW_MODEL?: string;
  AXON_SHADOW_CONFIG_REVISION?: string;
  AXON_PSEUDONYM_KEY: string;
  AXON_RELEASE_CERTIFIED?: string;
  GEMINI_INPUT_USD_PER_MILLION?: string;
  GEMINI_OUTPUT_USD_PER_MILLION?: string;
  TEST_MIGRATIONS?: Array<{ name: string; queries: string[] }>;
}

type Env = IntelligenceBindings & IntelligenceSecretBindings;

declare namespace Cloudflare {
  // Declaration merging is required so cloudflare:test receives secret bindings.
  // eslint-disable-next-line @typescript-eslint/no-empty-object-type
  interface Env extends IntelligenceSecretBindings {}
}
