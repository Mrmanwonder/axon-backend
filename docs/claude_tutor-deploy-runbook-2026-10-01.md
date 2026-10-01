# Tutor deploy runbook (AXO-126)

Prepared 2026-10-01. The code, workflow and tests for a staged Tutor rollout are in this branch. Four owner actions remain: secrets, two repository variables, and the internal user list. They can't be done from an agent session, because secret values must never pass through one.

## What ships

- **`axon-intelligence --env tutor`.** Tutor routes, D1 traces/rate limits/idempotency, the KV Tavily cache, and the hourly prune cron. No `DOCUMENT_VISION`, no paper queue, no paper routes (they 404; `test/tutor-profile.test.ts`). The mastery pipeline stays the only paper pipeline.
- **`mastery-api /tutor`.** Student Mode scope check → rollout switch → server-side paper/teacher evidence (AXO-36) → service call. The `INTELLIGENCE` binding is appended at deploy time only when the tutor Worker deployed in the same run (`deploy-bindings.test.mjs` pins this).
- **D1 schema.** The `axon-intelligence` database exists but has **0 tables** (checked 2026-10-01). The tutor job applies `workers/intelligence/migrations/0001…0010` before deploying.

## 1 · Secrets (once; values never in chat, logs or Git)

```sh
cd workers/intelligence
npx wrangler secret put GOOGLE_API_KEY       --env tutor   # paid-tier Google AI project
npx wrangler secret put SUPABASE_SECRET_KEY  --env tutor
npx wrangler secret put TAVILY_API_KEY       --env tutor
npx wrangler secret put AXON_ADMIN_TOKEN     --env tutor   # random 32+ bytes
npx wrangler secret put AXON_PSEUDONYM_KEY   --env tutor   # random 32+ bytes; never rotate casually (re-keys pseudonyms)
npx wrangler secret put AXON_INTERNAL_TOKEN  --env tutor   # random 32+ bytes, SAME value as below
cd ../api
npx wrangler secret put AXON_INTERNAL_TOKEN                # same value as above
npx wrangler secret put TUTOR_INTERNAL_USERS               # comma-separated Supabase auth user ids for the internal stage
```

To generate a token: `openssl rand -base64 48`.

`wrangler secret put --env tutor` creates the `axon-intelligence` script if it doesn't exist yet. That's harmless, because nothing binds to it until step 2.

Optional cost logging: `GEMINI_INPUT_USD_PER_MILLION` / `GEMINI_OUTPUT_USD_PER_MILLION` for the routed model.

`GEMINI_PRIVACY_MODE` stays `unverified` until Google confirms zero data retention in writing. Don't flip it to `zdr` on assumption.

## 2 · Repository variables (GitHub → Settings → Secrets and variables → Actions → Variables)

| Variable | Value | Effect |
|---|---|---|
| `TUTOR_DEPLOY_ENABLED` | `true` | Next push to `main` runs D1 migrations, deploys the tutor Worker, then deploys `mastery-api` bound to it |
| `TUTOR_ROLLOUT` | `internal` | Only `TUTOR_INTERNAL_USERS` can use `/tutor`; everyone else gets 503 "not available yet" |

Then push or re-run the `main` workflow.

## 3 · Stage gates

| Stage | `TUTOR_ROLLOUT` | Gate |
|---|---|---|
| Internal | `internal` | Steps 1–2 done; smoke below passes |
| Beta | *(not implemented: needs an opt-in flag)* | AXO-40 60-case golden set (20 Cambridge / 20 CBSE / 20 IBDP) passes AXO-44 thresholds with **0 critical hallucinations and 0 teacher-mark contradictions** |
| GA | `ga` | 500-case set, rollback drill, canary (AXO-44) |

Any value other than `internal` or `ga` is treated as **off**.

## 4 · Smoke (internal stage)

1. Sign in as an internal user, enter Student Mode, open a committed paper, and ask about one question.
2. Expect HTTP 200 from `/tutor` and an answer that cites the paper/teacher evidence.
3. Query D1 `axon-intelligence`: there should be a new trace row with model, prompt hash and latency, and **no raw student text** (AXO-106).
4. Sign in as a non-listed user: `/tutor` → 503.

## 5 · Rollback

- **Immediate** (seconds): `cd workers/api && npx wrangler secret delete AXON_INTERNAL_TOKEN`. `/tutor` returns 503 "not available yet" for everyone, with no deploy.
- **Config**: set `TUTOR_ROLLOUT=off` and re-run the deploy.
- **Full**: set `TUTOR_DEPLOY_ENABLED=false` and re-run. `mastery-api` redeploys without the binding. The tutor Worker stays deployed but unreachable; delete it with `npx wrangler delete --name axon-intelligence` only if wanted.
