import { describe, expect, it } from "vitest";
import { env } from "cloudflare:workers";
import worker, { TUTOR_PROFILE_ROUTES } from "../src/index";

/* AXO-126: `wrangler deploy --env tutor` ships without the paper pipeline's
   bindings; every paper route must be absent there, not half-working. */
describe("tutor-only deploy profile", () => {
  const ctx = { waitUntil: () => undefined, passThroughOnException: () => undefined } as unknown as ExecutionContext;
  const tutorEnv = { ...env, AXON_PROFILE: "tutor", DOCUMENT_VISION: undefined, PAPER_QUEUE: undefined, PAPER_ARTIFACTS: undefined } as unknown as Env;
  const call = (path: string, method = "POST") =>
    worker.fetch(new Request("https://axon.test" + path, { method, headers: { authorization: "Bearer test-token", "content-type": "application/json" }, body: method === "POST" ? "{}" : undefined }), tutorEnv, ctx);

  it("serves only health, the tutor and provider health", () => {
    expect([...TUTOR_PROFILE_ROUTES].sort()).toEqual(["/health", "/v1/admin/provider-health", "/v1/tutor"]);
  });

  for (const path of ["/v1/papers/ingest", "/v1/corrections", "/v1/insights/observations", "/v1/admin/capabilities/probe", "/v1/admin/readiness"]) {
    it("answers 404 for " + path, async () => {
      const response = await call(path, path.endsWith("readiness") ? "GET" : "POST");
      expect(response.status).toBe(404);
    });
  }

  it("still answers health", async () => {
    const response = await call("/health", "GET");
    expect(response.status).toBe(200);
  });
});
