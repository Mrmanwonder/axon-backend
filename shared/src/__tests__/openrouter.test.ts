import { test } from "node:test";
import assert from "node:assert/strict";
import { callModel, ModelError } from "../openrouter.js";

test("Gemini 3.5 uses the route thinking level and omits legacy temperature", async (t) => {
  const inserted: Array<Record<string, unknown>> = [];
  const route = {
    stage: "thinking-test",
    primary_model: "gemini-3.5-flash-lite",
    fallbacks: [],
    temperature: 0,
    max_tokens: 512,
    prompt_version: "test.v1",
    thinking_level: "high",
    allow_training: false,
    enabled: true,
  };
  const sb = {
    from(table: string) {
      if (table === "model_route") {
        const query = {
          select: () => query,
          eq: () => query,
          maybeSingle: async () => ({ data: route, error: null }),
        };
        return query;
      }
      return {
        insert: async (row: Record<string, unknown>) => {
          inserted.push(row);
          return { error: null };
        },
      };
    },
  };
  let requestBody: Record<string, unknown> | undefined;
  t.mock.method(globalThis, "fetch", async (_url: string | URL | Request, init?: RequestInit) => {
    requestBody = JSON.parse(String(init?.body)) as Record<string, unknown>;
    return Response.json({
      model: "gemini-3.5-flash-lite",
      choices: [{ message: { content: JSON.stringify({ answer: "ok" }) } }],
      usage: { prompt_tokens: 3, completion_tokens: 2 },
    });
  });

  const result = await callModel({
    env: { GOOGLE_API_KEY: "test" },
    sb: sb as never,
    stage: "thinking-test",
    system: "system",
    instruction: "instruction",
    schema: { name: "test", schema: { type: "object" } },
    validate: (value) => value as { answer: string },
  });

  assert.equal(result.model, "gemini-3.5-flash-lite");
  assert.equal(result.requestedModel, "gemini-3.5-flash-lite");
  assert.equal(requestBody?.reasoning_effort, "high");
  assert.ok(!Object.hasOwn(requestBody ?? {}, "temperature"));
  assert.equal(inserted.at(-1)?.thinking_level, "high");
  assert.equal(inserted.at(-1)?.schema_valid, true);
  assert.equal(inserted.at(-1)?.verification_status, "transport_only");
  assert.equal(inserted.at(-1)?.answer_status, "pending_verification");
  assert.deepEqual(inserted.at(-1)?.verification_failures, []);
  assert.deepEqual(inserted.at(-1)?.tool_calls, []);
  assert.equal(inserted.at(-1)?.grounding_used, false);
  assert.equal(inserted.at(-1)?.repair_attempted, false);
});

test("an unexpected configured route is rejected before student data reaches Gemini", async (t) => {
  const inserted: Array<Record<string, unknown>> = [];
  const route = {
    stage: "route-drift-test", primary_model: "gemini-unapproved", fallbacks: [], temperature: 0,
    max_tokens: 512, prompt_version: "test.v1", thinking_level: "low" as const, allow_training: false, enabled: true
  };
  const sb = {
    from(table: string) {
      if (table === "model_route") {
        const query = { select: () => query, eq: () => query, maybeSingle: async () => ({ data: route, error: null }) };
        return query;
      }
      return { insert: async (row: Record<string, unknown>) => { inserted.push(row); return { error: null }; } };
    }
  };
  let fetched = false;
  t.mock.method(globalThis, "fetch", async () => { fetched = true; return Response.json({}); });

  await assert.rejects(callModel({
    env: { GOOGLE_API_KEY: "test" }, sb: sb as never, stage: route.stage, system: "system", instruction: "student input",
    schema: { name: "test", schema: { type: "object" } }, validate: (value) => value,
    expectedModel: "gemini-3.5-flash-lite"
  }), (error: unknown) => error instanceof ModelError && error.code === "route_model_mismatch");

  assert.equal(fetched, false);
  assert.equal(inserted.at(-1)?.error_code, "route_model_mismatch");
});

test("a provider response from a different model is withheld and recorded", async (t) => {
  const inserted: Array<Record<string, unknown>> = [];
  const route = {
    stage: "served-drift-test", primary_model: "gemini-3.5-flash-lite", fallbacks: [], temperature: 0,
    max_tokens: 512, prompt_version: "test.v1", thinking_level: "low" as const, allow_training: false, enabled: true
  };
  const sb = {
    from(table: string) {
      if (table === "model_route") {
        const query = { select: () => query, eq: () => query, maybeSingle: async () => ({ data: route, error: null }) };
        return query;
      }
      return { insert: async (row: Record<string, unknown>) => { inserted.push(row); return { error: null }; } };
    }
  };
  t.mock.method(globalThis, "fetch", async () => Response.json({
    model: "gemini-unapproved",
    choices: [{ message: { content: JSON.stringify({ answer: "must not escape" }) } }],
    usage: { prompt_tokens: 3, completion_tokens: 2 }
  }));

  await assert.rejects(callModel({
    env: { GOOGLE_API_KEY: "test" }, sb: sb as never, stage: route.stage, system: "system", instruction: "student input",
    schema: { name: "test", schema: { type: "object" } }, validate: (value) => value,
    expectedModel: "gemini-3.5-flash-lite"
  }), (error: unknown) => error instanceof ModelError && error.code === "served_model_mismatch");

  assert.equal(inserted.at(-1)?.model_id, "gemini-unapproved");
  assert.equal(inserted.at(-1)?.answer_status, "controlled_failure");
});
