// Despite the filename (a holdover from before the Gemini migration — see
// AXON_FIX_BRIEF.md §F1/§F3), this calls Gemini directly via its
// OpenAI-compatible endpoint. Renaming the file is a one-line import change
// away whenever someone gets to it; not done here to keep this reconstruction
// a faithful port of the live bundle, not a drive-by rename.
import type { SupabaseClient } from "@supabase/supabase-js";
import type { Env } from "./env.js";
import { TAVILY_TOOLS, WEB_TOOL_SYSTEM_GUARD, runTavilyTool, type TavilyToolCall } from "./tavily.js";

const OR_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions";

export class ModelError extends Error {
  code: string;
  status: number;
  retryable: boolean;
  constructor(code: string, message: string, status = 0, retryable = false) {
    super(message);
    this.name = "ModelError";
    this.code = code;
    this.status = status;
    this.retryable = retryable;
  }
}

export interface ModelRoute {
  stage: string;
  primary_model: string;
  fallbacks: string[] | null;
  temperature: number;
  max_tokens: number;
  prompt_version: string;
  allow_training: boolean;
  enabled: boolean;
}

export interface RouteOverride {
  primary_model?: string;
  fallbacks?: string[];
  temperature?: number;
  max_tokens?: number;
  prompt_version?: string;
}

const ROUTE_TTL_MS = 60_000;
const routes = new Map<string, { at: number; route: ModelRoute }>();

export async function getRoute(sb: SupabaseClient, stage: string): Promise<ModelRoute> {
  const cached = routes.get(stage);
  if (cached && Date.now() - cached.at < ROUTE_TTL_MS) return cached.route;
  const { data, error } = await sb
    .from("model_route")
    .select("stage, primary_model, fallbacks, temperature, max_tokens, prompt_version, allow_training, enabled")
    .eq("stage", stage)
    .maybeSingle();
  if (error) throw new ModelError("route_lookup_failed", `could not read the route for ${stage}: ${error.message}`);
  if (!data) throw new ModelError("no_route", `no model route is configured for ${stage}`);
  if (!data.enabled) throw new ModelError("route_disabled", `the ${stage} route is switched off`);
  const route = data as ModelRoute;
  routes.set(stage, { at: Date.now(), route });
  return route;
}

function applyOverride(route: ModelRoute, override?: RouteOverride | null): ModelRoute {
  if (!override) return route;
  return {
    ...route,
    ...(override.primary_model ? { primary_model: override.primary_model } : {}),
    ...(Array.isArray(override.fallbacks) ? { fallbacks: override.fallbacks } : {}),
    ...(typeof override.temperature === "number" ? { temperature: override.temperature } : {}),
    ...(typeof override.max_tokens === "number" ? { max_tokens: override.max_tokens } : {}),
    ...(override.prompt_version ? { prompt_version: override.prompt_version } : {}),
    // A route override can swap models or tune sampling, but never opt a
    // stage into training on its own — that stays a route-level decision.
    allow_training: route.allow_training,
  };
}

function classify(status: number, body: string): ModelError {
  const lower = body.toLowerCase();
  if (status === 404 && (lower.includes("no endpoints") || lower.includes("no allowed providers"))) {
    return new ModelError(
      "no_compliant_provider",
      "No provider for this model meets the zero-data-retention policy. Either pick a model with a compliant endpoint, or decide deliberately that this stage may be trained on.",
      status,
      false
    );
  }
  if (status === 401) return new ModelError("invalid_key", "Gemini rejected GOOGLE_API_KEY (401).", status, false);
  if (status === 403) return new ModelError("forbidden", "Gemini rejected the request (403): " + body.slice(0, 300), status, false);
  if (status === 402) return new ModelError("billing", "Gemini rejected the request for billing/quota reasons", status, false);
  if (status === 429) return new ModelError("rate_limited", "rate limited by Gemini", status, true);
  if (status >= 500) return new ModelError("provider_error", `provider returned ${status}`, status, true);
  return new ModelError("bad_request", body.slice(0, 500) || `request failed with ${status}`, status, false);
}

export interface ImageRefResult {
  url: string;
  key: string;
  detail: "low" | "high";
}

export interface JsonSchema {
  name: string;
  schema: Record<string, unknown>;
}

export interface CallModelOptions<T> {
  env: Env;
  sb: SupabaseClient;
  stage: string;
  system: string;
  instruction: string;
  images?: ImageRefResult[];
  schema: JsonSchema;
  validate: (parsed: unknown) => T;
  runId?: string | null;
  paperId?: string | null;
  regionId?: string | null;
  studentId?: string | null;
  attempt?: number;
  routeOverride?: RouteOverride | null;
  timeoutMs?: number;
  /** Give Gemini the Tavily Search/Extract tools for this call. Off by default. */
  webTools?: boolean;
}

export interface CallModelResult<T> {
  parsed: T;
  model: string;
  promptVersion: string;
  inputTokens: number | null;
  outputTokens: number | null;
  costUsd: number | null;
  /** Public URLs consulted by Tavily during this model call. */
  webSources: string[];
  latencyMs: number;
}

// Transient failures get exactly one inline retry before the call throws and
// falls back to the queue's own retry (see shared/worker.ts). Timeouts are
// deliberately excluded — they have already burned the timeout budget once,
// so retrying inline just burns it again for no better odds.
const TRANSIENT_HTTP = new Set([408, 409, 425, 429, 500, 502, 503, 504]);
const INLINE_TRIES = 2;

export async function callModel<T>(opts: CallModelOptions<T>): Promise<CallModelResult<T>> {
  const key = opts.env.GOOGLE_API_KEY;
  if (!key) throw new ModelError("no_key", "GOOGLE_API_KEY is not set for this worker", 0, false);

  const route = applyOverride(await getRoute(opts.sb, opts.stage), opts.routeOverride);
  const attempt = opts.attempt ?? 1;
  const started = Date.now();

  const content: Array<{ type: string; text?: string; image_url?: { url: string } }> = [
    { type: "text", text: opts.instruction },
  ];
  for (const image of opts.images ?? []) {
    content.push({ type: "image_url", image_url: { url: image.url } });
  }

  const system = opts.webTools
    ? `${opts.system}\n\n${WEB_TOOL_SYSTEM_GUARD}`
    : opts.system;
  const messages: Record<string, unknown>[] = [
    { role: "system", content: system },
    { role: "user", content },
  ];

  const log = (patch: Record<string, unknown>) =>
    logCall(opts.sb, {
      run_id: opts.runId ?? null,
      paper_id: opts.paperId ?? null,
      region_id: opts.regionId ?? null,
      student_id: opts.studentId ?? null,
      stage: opts.stage,
      requested_model: route.primary_model,
      prompt_version: route.prompt_version,
      attempt,
      latency_ms: Date.now() - started,
      image_keys: (opts.images ?? []).map((i) => i.key),
      ...patch,
    });

  let inputTokens: number | null = null;
  let outputTokens: number | null = null;
  let costUsd: number | null = null;
  let served = route.primary_model;
  let toolCallsUsed = 0;
  const webSources = new Set<string>();

  const add = (current: number | null, value: number | undefined): number | null =>
    typeof value === "number" && Number.isFinite(value) ? (current ?? 0) + value : current;

  const request = async (body: Record<string, unknown>): Promise<any> => {
    let res: Response | undefined;
    for (let tryNo = 1; ; tryNo++) {
      let caught: ModelError | null = null;
      try {
        res = await fetch(OR_URL, {
          method: "POST",
          headers: {
            Authorization: `Bearer ${key}`,
            "Content-Type": "application/json",
          },
          body: JSON.stringify(body),
          signal: AbortSignal.timeout(opts.timeoutMs ?? 90_000),
        });
      } catch (cause) {
        const timedOut = cause instanceof DOMException && cause.name === "TimeoutError";
        caught = new ModelError(
          timedOut ? "timeout" : "network",
          timedOut ? "the model did not answer in time" : String(cause),
          0,
          true
        );
      }

      if (!caught && res!.ok) return await res!.json();

      const rawBody = caught ? "" : await res!.text();
      const err = caught ?? classify(res!.status, rawBody);
      const retryHere =
        tryNo < INLINE_TRIES &&
        (caught ? caught.code === "network" : TRANSIENT_HTTP.has(res!.status));

      if (!retryHere) {
        await log(
          caught
            ? { model_id: served, ok: false, error_code: err.code }
            : {
                model_id: served,
                ok: false,
                error_code: err.code,
                http_status: res!.status,
                error_detail: rawBody.slice(0, 500),
              }
        );
        throw err;
      }

      console.warn(
        "transient Gemini error, retrying in-process",
        opts.stage,
        caught ? err.code : res!.status,
        "try",
        tryNo
      );
      await new Promise((sleep) => setTimeout(sleep, 800 * tryNo + Math.floor(Math.random() * 400)));
    }
  };

  // Three Tavily calls allows search -> extract -> one refinement while keeping
  // the model/tool loop bounded. No current pipeline worker opts in implicitly.
  const maxRounds = opts.webTools ? 4 : 1;

  for (let round = 0; round < maxRounds; round++) {
    const body: Record<string, unknown> = {
      model: route.primary_model,
      temperature: route.temperature,
      max_tokens: route.max_tokens,
      messages,
      response_format: {
        type: "json_schema",
        json_schema: { name: opts.schema.name, schema: opts.schema.schema },
      },
      ...(opts.webTools ? { tools: TAVILY_TOOLS, tool_choice: "auto" } : {}),
    };

    const data = await request(body);
    served = data.model ?? served;
    inputTokens = add(inputTokens, data.usage?.prompt_tokens);
    outputTokens = add(outputTokens, data.usage?.completion_tokens);
    costUsd = add(costUsd, data.usage?.cost);

    const message = data.choices?.[0]?.message;
    if (data.error || !message) {
      const err = new ModelError("empty_response", data.error?.message ?? "Gemini returned nothing", 200, true);
      await log({
        model_id: served,
        ok: false,
        error_code: err.code,
        input_tokens: inputTokens,
        output_tokens: outputTokens,
        cost_usd: costUsd,
      });
      throw err;
    }

    const rawCalls = Array.isArray(message.tool_calls) ? message.tool_calls as TavilyToolCall[] : [];
    if (rawCalls.length) {
      if (!opts.webTools) {
        const err = new ModelError("unexpected_tool_call", "Gemini requested a tool on a tool-free call", 200, false);
        await log({ model_id: served, ok: false, error_code: err.code });
        throw err;
      }
      if (round === maxRounds - 1) {
        const err = new ModelError("tool_loop_limit", "Gemini exhausted the live-web round budget", 200, false);
        await log({ model_id: served, ok: false, error_code: err.code });
        throw err;
      }

      const calls = rawCalls.map((call, index) => ({
        ...call,
        id: call.id || `${call.function.name}-${round}-${index}`,
      }));

      // Gemini's OpenAI-compatibility layer expects the assistant tool-call
      // message to carry non-empty content on the follow-up request.
      messages.push({
        role: "assistant",
        content: typeof message.content === "string" && message.content
          ? message.content
          : "Using live web reference tools.",
        tool_calls: calls,
      });

      for (const call of calls) {
        if (toolCallsUsed >= 3) {
          const err = new ModelError("tool_loop_limit", "Gemini exceeded the live-web tool-call limit", 200, false);
          await log({ model_id: served, ok: false, error_code: err.code });
          throw err;
        }
        toolCallsUsed += 1;

        const result = await runTavilyTool(opts.env, call);
        for (const source of result.sources) webSources.add(source);

        messages.push({
          role: "tool",
          tool_call_id: call.id,
          name: call.function.name,
          content: result.content,
        });
      }
      continue;
    }

    const raw = message.content;
    if (typeof raw !== "string" || raw.length === 0) {
      const err = new ModelError("empty_response", "Gemini returned no final content", 200, true);
      await log({
        model_id: served,
        ok: false,
        error_code: err.code,
        input_tokens: inputTokens,
        output_tokens: outputTokens,
        cost_usd: costUsd,
      });
      throw err;
    }

    let parsed: T;
    try {
      parsed = opts.validate(JSON.parse(raw));
    } catch (cause) {
      const err = new ModelError("bad_shape", `the model's answer did not fit the schema: ${cause}`, 200, true);
      await log({
        model_id: served,
        ok: false,
        error_code: err.code,
        input_tokens: inputTokens,
        output_tokens: outputTokens,
        cost_usd: costUsd,
      });
      throw err;
    }

    await log({
      model_id: served,
      ok: true,
      input_tokens: inputTokens,
      output_tokens: outputTokens,
      cost_usd: costUsd,
    });

    return {
      parsed,
      model: served,
      promptVersion: route.prompt_version,
      inputTokens,
      outputTokens,
      costUsd,
      webSources: [...webSources],
      latencyMs: Date.now() - started,
    };
  }

  const err = new ModelError("tool_loop_limit", "Gemini did not finish within the live-web tool budget", 200, false);
  await log({ model_id: served, ok: false, error_code: err.code });
  throw err;
}

async function logCall(sb: SupabaseClient, row: Record<string, unknown>): Promise<void> {
  const { error } = await sb.from("model_call").insert(row);
  if (error) console.error("model_call insert failed", error.message, row.stage, row.error_code ?? "ok");
}
