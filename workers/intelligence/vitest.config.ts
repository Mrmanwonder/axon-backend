import { cloudflareTest, readD1Migrations } from "@cloudflare/vitest-plugin";
import { defineConfig } from "vitest/config";
import path from "node:path";

export default defineConfig({
  plugins: [cloudflareTest(async () => ({
    wrangler: { configPath: "./wrangler.jsonc" },
    miniflare: {
      bindings: { TEST_MIGRATIONS: await readD1Migrations(path.join(import.meta.dirname, "migrations")), AXON_INTERNAL_TOKEN: "test-token", AXON_ADMIN_TOKEN: "test-admin-token", AXON_PSEUDONYM_KEY: "test-pseudonym-key", GOOGLE_API_KEY: "test-only", SUPABASE_SECRET_KEY: "test-only" },
      serviceBindings: { DOCUMENT_VISION: { network: {} } }
    }
  }))],
  test: { setupFiles: ["./test/apply-migrations.ts"] }
});
