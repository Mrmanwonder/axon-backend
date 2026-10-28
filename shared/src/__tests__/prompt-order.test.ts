import { test } from "node:test";
import assert from "node:assert/strict";
import { callModel } from "../model-client.js";
import { SYSTEM as TRIAGE } from "../prompts/triage.v1.js";
import { SYSTEM as STRUCTURE } from "../prompts/structure.v1.js";
import { SYSTEM as CONTENT } from "../prompts/content.v1.js";
import { SYSTEM as STRUCTURE_V2 } from "../prompts/structure.v2.js";
import { SYSTEM as CONTENT_V2 } from "../prompts/content.v2.js";
import { SYSTEM as ADJUDICATE } from "../prompts/adjudicate.v1.js";
import { SYSTEM as EXPLAIN_T1 } from "../prompts/explain_tier1.v2.js";
import { SYSTEM as EXPLAIN_T2 } from "../prompts/explain_tier2.v1.js";

// Implicit prefix caching only works when everything that repeats across calls comes
// first and everything that is specific to this paper comes last. Pin that shape.

test("every stage's system prompt is a static string, not built from per-paper data", () => {
  for (const system of [TRIAGE, STRUCTURE, CONTENT, STRUCTURE_V2, CONTENT_V2, ADJUDICATE, EXPLAIN_T1, EXPLAIN_T2]) {
    assert.equal(typeof system, "string");
    assert.ok(system.length > 200, "a system prompt this short is not carrying the stage's instructions");
    // A per-paper value can never appear in a module-level constant, but a template
    // placeholder left behind by a refactor would.
    assert.doesNotMatch(system, /\$\{|\{\{/);
  }
});

test("the request sends the static system prompt first and the per-paper text then images last", async (t) => {
  const route = {
    stage: "order-test", primary_model: "gemini-3.8-flash", provider: "ai_studio", fallbacks: [], temperature: 0,
    max_tokens: 8192, prompt_version: "t.v1", thinking_level: "medium", allow_training: false, enabled: true,
  };
  const sb = {
    from(table: string) {
      if (table === "model_route") {
        const q = { select: () => q, eq: () => q, maybeSingle: async () => ({ data: route, error: null }) };
        return q;
      }
      return { insert: async () => ({ error: null }) };
    },
  };
  let body: any;
  t.mock.method(globalThis, "fetch", async (_u: string | URL | Request, init?: RequestInit) => {
    body = JSON.parse(String(init?.body));
    return Response.json({
      model: "gemini-3.8-flash",
      choices: [{ finish_reason: "stop", message: { content: '{"ok":true}' } }],
      usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
    });
  });

  await callModel({
    env: { GOOGLE_API_KEY: "k" },
    sb: sb as never,
    stage: "order-test",
    system: "STATIC SYSTEM",
    instruction: "PER-PAPER INSTRUCTION",
    images: [{ url: "https://example.test/page-1", key: "k1", detail: "high" }],
    schema: { name: "s", schema: { type: "object" } },
    validate: (v) => v as { ok: boolean },
  });

  assert.equal(body.messages[0].role, "system");
  assert.equal(body.messages[0].content, "STATIC SYSTEM");
  const parts = body.messages[1].content as Array<{ type: string }>;
  assert.deepEqual(parts.map((p) => p.type), ["text", "image_url"]);
  assert.equal(body.model, "gemini-3.8-flash");
  assert.ok(!Object.hasOwn(body, "temperature"), "no sampling override on Gemini 3.x");
});
