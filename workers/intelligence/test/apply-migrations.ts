import { beforeAll } from "vitest";
import { env } from "cloudflare:workers";
import { applyD1Migrations } from "cloudflare:test";

beforeAll(async () => {
  if (env.TEST_MIGRATIONS) await applyD1Migrations(env.DB, env.TEST_MIGRATIONS);
});
