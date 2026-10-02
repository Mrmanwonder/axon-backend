import { test } from "node:test";
import assert from "node:assert/strict";
import { callModel, ModelError } from "../model-client.js";

test("Gemini 3.5 uses the route thinking level and omits legacy temperature", async (t) => {
  const inserted: Array<Record<string, unknown>> = [];
  const route = {
    stage: "thinking-test",
    primary_model: "gemini-3.8-flash",
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
      model: "gemini-3.8-flash",
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

  assert.equal(result.model, "gemini-3.8-flash");
  assert.equal(result.requestedModel, "gemini-3.8-flash");
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
    expectedModel: "gemini-3.8-flash"
  }), (error: unknown) => error instanceof ModelError && error.code === "route_model_mismatch");

  assert.equal(fetched, false);
  assert.equal(inserted.at(-1)?.error_code, "route_model_mismatch");
});

test("a provider response from a different model is withheld and recorded", async (t) => {
  const inserted: Array<Record<string, unknown>> = [];
  const route = {
    stage: "served-drift-test", primary_model: "gemini-3.8-flash", fallbacks: [], temperature: 0,
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
    expectedModel: "gemini-3.8-flash"
  }), (error: unknown) => error instanceof ModelError && error.code === "served_model_mismatch");

  assert.equal(inserted.at(-1)?.model_id, "gemini-unapproved");
  assert.equal(inserted.at(-1)?.answer_status, "controlled_failure");
});

function harness(stage: string, maxTokens: number) {
  const inserted: Array<Record<string, unknown>> = [];
  const route = {
    stage,
    primary_model: "gemini-3.1-flash-lite",
    fallbacks: [],
    temperature: 0,
    max_tokens: maxTokens,
    prompt_version: "test.v1",
    thinking_level: "low",
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
  const run = (extra: Record<string, unknown> = {}) =>
    callModel({
      env: { GOOGLE_API_KEY: "test" },
      sb: sb as never,
      stage,
      system: "system",
      instruction: "instruction",
      schema: { name: "test", schema: { type: "object" } },
      validate: (value) => {
        const v = value as { answer?: string };
        if (typeof v.answer !== "string") throw new Error("answer must be a string");
        return v as { answer: string };
      },
      ...extra,
    });
  return { inserted, run };
}

test("a truncated answer is repaired once with double the budget", async (t) => {
  const { inserted, run } = harness("truncate-repair", 1000);
  const bodies: Array<Record<string, unknown>> = [];
  t.mock.method(globalThis, "fetch", async (_u: string | URL | Request, init?: RequestInit) => {
    bodies.push(JSON.parse(String(init?.body)) as Record<string, unknown>);
    return bodies.length === 1
      ? Response.json({
          model: "gemini-3.1-flash-lite",
          choices: [{ finish_reason: "length", message: { content: '{"answer": "cut' } }],
          usage: { prompt_tokens: 10, completion_tokens: 75, completion_tokens_details: { reasoning_tokens: 70 } },
        })
      : Response.json({
          model: "gemini-3.1-flash-lite",
          choices: [{ finish_reason: "stop", message: { content: '{"answer":"ok"}' } }],
          usage: { prompt_tokens: 10, completion_tokens: 40, completion_tokens_details: { reasoning_tokens: 20 } },
        });
  });

  const result = await run();
  assert.equal(result.parsed.answer, "ok");
  assert.equal(bodies.length, 2);
  assert.equal(bodies[0].max_tokens, 1000);
  assert.equal(bodies[1].max_tokens, 2000);
  assert.equal(result.reasoningTokens, 90);
  assert.equal(inserted.length, 1);
  assert.equal(inserted[0].ok, true);
  assert.equal(inserted[0].repair_attempted, true);
  assert.equal(inserted[0].reasoning_tokens, 90);
});

test("a still-truncated answer fails once, diagnosably, with no student text", async (t) => {
  const { inserted, run } = harness("truncate-fail", 1000);
  let calls = 0;
  t.mock.method(globalThis, "fetch", async () => {
    calls += 1;
    return Response.json({
      model: "gemini-3.1-flash-lite",
      choices: [{ finish_reason: "length", message: { content: '{"answer": "student wrote SECRET' } }],
      usage: { prompt_tokens: 10, completion_tokens: 75, completion_tokens_details: { reasoning_tokens: 70 } },
    });
  });

  await assert.rejects(run(), (err: unknown) => err instanceof ModelError && err.code === "bad_shape");
  assert.equal(calls, 2);
  const row = inserted.at(-1)!;
  assert.equal(row.ok, false);
  assert.equal(row.repair_attempted, true);
  assert.match(String(row.error_detail), /finish_reason=length/);
  assert.match(String(row.error_detail), /reasoning_tokens=140/);
  assert.doesNotMatch(String(row.error_detail), /SECRET/);
});

test("a schema mismatch on a normal stop logs the validator message and is not repaired", async (t) => {
  const { inserted, run } = harness("shape-mismatch", 1000);
  let calls = 0;
  t.mock.method(globalThis, "fetch", async () => {
    calls += 1;
    return Response.json({
      model: "gemini-3.1-flash-lite",
      choices: [{ finish_reason: "stop", message: { content: '{"answer": 5}' } }],
      usage: { prompt_tokens: 10, completion_tokens: 8 },
    });
  });

  await assert.rejects(run(), (err: unknown) => err instanceof ModelError && err.code === "bad_shape");
  assert.equal(calls, 1);
  const row = inserted.at(-1)!;
  assert.equal(row.repair_attempted, false);
  assert.match(String(row.error_detail), /finish_reason=stop/);
  assert.match(String(row.error_detail), /validator=answer must be a string/);
});

test("serviceTier flex is sent to the provider, and the standard tier sends none", async (t) => {
  const bodies: Array<Record<string, unknown>> = [];
  t.mock.method(globalThis, "fetch", async (_u: string | URL | Request, init?: RequestInit) => {
    bodies.push(JSON.parse(String(init?.body)) as Record<string, unknown>);
    return Response.json({
      model: "gemini-3.1-flash-lite",
      choices: [{ finish_reason: "stop", message: { content: '{"answer":"ok"}' } }],
      usage: { prompt_tokens: 1, completion_tokens: 1 },
    });
  });

  const flex = harness("tier-flex", 1000);
  await flex.run({ serviceTier: "flex" });
  const standard = harness("tier-standard", 1000);
  await standard.run();

  assert.equal(bodies[0].service_tier, "flex");
  assert.ok(!Object.hasOwn(bodies[1], "service_tier"));
  // The tier the call was made on is logged so the database prices it correctly.
  assert.equal(flex.inserted.at(-1)?.service_tier, "flex");
  assert.equal(standard.inserted.at(-1)?.service_tier, "standard");
});

test("usage is logged for pricing: cached tokens and billed output including thinking", async (t) => {
  const { inserted, run } = harness("usage-pricing", 1000);
  t.mock.method(globalThis, "fetch", async () =>
    Response.json({
      model: "gemini-3.1-flash-lite",
      choices: [{ finish_reason: "stop", message: { content: '{"answer":"ok"}' } }],
      usage: {
        prompt_tokens: 10000,
        completion_tokens: 1200,
        total_tokens: 12000,
        prompt_tokens_details: { cached_tokens: 4000 },
        completion_tokens_details: { reasoning_tokens: 800 },
      },
    }));

  await run();
  const row = inserted.at(-1)!;
  assert.equal(row.input_tokens, 10000);
  assert.equal(row.cached_tokens, 4000);
  // total - prompt = 2000: completion 1200 + thinking 800, billed as output.
  assert.equal(row.billed_output_tokens, 2000);
  assert.equal(row.reasoning_tokens, 800);
  // The database prices the call; the worker never invents a cost.
  assert.equal(row.cost_usd, null);
});
