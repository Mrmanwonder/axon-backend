import { expect, test } from "vitest";
import worker from "../src/index.js";
const env = { SUPABASE_URL: "https://example.test", SUPABASE_ANON_KEY: "fixture" } as any;
for (const path of ["/paper-submit", "/paper-retry", "/upload-intent", "/upload-complete", "/review-complete", "/explain-retry", "/page-asset-urls", "/paper-originals", "/tutor"]) {
  test(`${path}: oversized JSON has bounded 413 with CORS and no echoed payload`, async () => {
    const res = await worker.fetch(new Request(`https://api.test${path}`, {
      method: "POST", headers: { authorization: "Bearer fixture", "content-type": "application/json", "content-length": "1", origin: "https://www.axonstudy.online" },
      body: JSON.stringify({ data: "x".repeat(270_000) }),
    }), env);
    expect(res.status).toBe(413);
    expect(res.headers.get("access-control-allow-origin")).toBe("https://www.axonstudy.online");
    expect((await res.text()).length).toBeLessThan(150);
  });
}
test("tutor's 100KB contract also applies without Content-Length", async () => {
  const res = await worker.fetch(new Request("https://api.test/tutor", {
    method: "POST", headers: { authorization: "Bearer fixture", "content-type": "application/json" }, body: JSON.stringify({ data: "x".repeat(110_000) }),
  }), env);
  expect(res.status).toBe(413);
});
