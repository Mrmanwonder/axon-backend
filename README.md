# axon-backend

The `mastery-*` Cloudflare Workers pipeline behind [Axon](https://github.com/Mrmanwonder/Axon-Site),
plus the fail-closed `axon-intelligence` tutor and `axon-document-vision` service.

The paper pipeline runs as a chain of queue consumers:

```
triage → structure → crop → content → reconcile → adjudicate → explain
```

`mastery-api` is the student-facing HTTP worker. `mastery-sweep` runs on three crons:
every minute it dispatches `private.stage_outbox` (AXO-225); every 15 minutes it
recovers stuck runs and garbage-collects R2; Sunday 12:30 UTC it sends the weekly
parent digest (AXO-263).

**Read `docs/README.md` first.** This repository exists because these workers used to
be hand-edited in production with no source at all — see there, and
`docs/AXON_FIX_BRIEF.md`, before changing anything.

## Layout

```
shared/                 the shared library, one copy for every worker
  src/
    env.ts              Env bindings interface
    http.ts             CORS, JSON responses, Supabase clients
    model-client.ts     callModel(): the one Gemini client (AI Studio or Vertex) + retry/tool loop
    tavily.ts           Tavily Search/Extract tools for opt-in live web grounding
    worker.ts           consumeQueue(): the queue-consumer harness
    queue_tuning.ts     per-stage batch size and concurrency (must match each wrangler.toml)
    r2.ts               R2 reads/writes, presigned URLs, asset signing
    contract.ts         box-provenance / PIPELINE_VERSION
    attribution.ts      ink-mark → question-region attribution
    reconcile.ts        paper-total reconciliation
    confidence.ts       the confidence-tier assess() logic
    quality_floor.ts    the "do this next" quality gate
    prompts.ts          stage system prompts, schemas, validators (with prompts/*.ts)
    __tests__/          unit tests for the shared logic
workers/
  triage/ structure/ crop/ content/ reconcile/ adjudicate/ explain/
  sweep/ api/ intelligence/ document-vision/
    src/index.ts        wrangler.toml (or wrangler.jsonc)        package.json
eval/                   prompt-manifest.json and committed eval-run evidence for the prompt gate
scripts/                CI gates, registries and one-off maintenance scripts
db/migrations/          reference snapshot of RPCs + migrations written from this repo; not
                        wired into a migration runner (see db/migrations/README.md)
docs/                   AXON_FIX_BRIEF.md, security notes, and why this repo exists
```

## Working here

```bash
npm install
npm run typecheck                # every workspace
npm test                         # shared/ and every worker
npm run dry-run                  # bundle-check every worker without deploying
npm run check:contract-parity    # contract.ts against Axon-Site
```

Every worker has a suite in `workers/<name>/test/` (AXO-296). The fake Supabase client they
share is `shared/src/__tests__/support/fake-db.ts`.

To work on one worker: `cd workers/<name> && npm run dev` (or `wrangler dev`).

Secrets aren't in the repo. They are bound on the live Cloudflare workers;
`wrangler deploy` preserves them, but verify per worker after a deploy per
`AXON_FIX_BRIEF.md` §5.3.5. A non-secret setting that exists only in the dashboard is
wiped by the next `wrangler deploy`, so non-secret vars live in `wrangler.toml`.

Gemini live-web grounding is enabled only for `mastery-explain`. Set `TAVILY_API_KEY`
as a Cloudflare Worker secret on that worker (and optionally `TAVILY_PROJECT` for usage
attribution). The model cannot author the Tavily search query: `mastery-explain`
supplies public-only subject/question context and the adapter restricts extraction to
URLs returned by that search.

Do not put the Tavily key in Axon-Site, Supabase Edge Function secrets, Wrangler
plaintext vars, or any browser-exposed variable.

## Things this repo's CI and `wrangler.toml`s deliberately hold the line on

- Queue batch sizes are set per stage in `shared/src/queue_tuning.ts` (AXO-211) and must
  equal `max_batch_size` in that stage's `wrangler.toml`; a test fails if they drift.
  The limits are sized to stay under a 50-subrequest invocation ceiling, because the
  subrequest-ceiling bug is what took down `mastery-content` — see `AXON_FIX_BRIEF.md`
  §3.3 and §10 before raising any of them. Every queue has a dead-letter queue.
- `wrangler deploy` (not a raw bundle upload) for every deploy, so secrets are never at
  risk of being wiped.
- A changed prompt file needs a real, measured eval run in `eval/` before CI accepts it
  (`scripts/check-prompt-gate.mjs`).

See `docs/AXON_FIX_BRIEF.md` §10 for the full "do not do these" list.
