# axon-backend

The `mastery-*` Cloudflare Workers pipeline behind [Axon](https://github.com/Mrmanwonder/Axon-Site):
triage → structure → content → reconcile → adjudicate → explain, plus
`mastery-sweep` (a `*/15 * * * *` cron for stuck-run recovery and R2
garbage collection) and `mastery-api` (the student-facing HTTP worker).

**Read `docs/README.md` first.** This repository's whole reason for
existing is that these workers used to be hand-edited in production with no
source at all — see there, and `docs/AXON_FIX_BRIEF.md`, before changing
anything.

## Layout

```
shared/                 the actually-shared library — one copy, not six
  src/
    env.ts               Env bindings interface
    http.ts               CORS, JSON responses, Supabase clients
    openrouter.ts         callModel(): the model call + retry logic
    worker.ts             consumeQueue(): the queue-consumer harness
    r2.ts                  R2 reads/writes, presigned URLs, asset signing
    contract.ts             box-provenance / PIPELINE_VERSION
    attribution.ts           ink-mark → question-region attribution
    reconcile.ts              paper-total reconciliation
    confidence.ts             the confidence-tier assess() logic
    quality_floor.ts          the "do this next" quality gate
    prompts.ts + prompts/*.ts  stage system prompts, schemas, validators
    __tests__/                unit tests for the pure logic above
workers/
  triage/ structure/ content/ reconcile/ adjudicate/ explain/ sweep/ api/
    src/index.ts    wrangler.toml    package.json
db/migrations/       reference snapshot of RPCs + migration history
docs/                 AXON_FIX_BRIEF.md, and why this repo exists
```

## Working here

```bash
npm install
npm run typecheck   # every workspace
npm test            # shared/'s unit tests
npm run dry-run      # bundle-check every worker without deploying
```

To work on one worker: `cd workers/<name> && npm run dev` (or
`wrangler dev`). Secrets aren't in the repo — they're bound on the live
Cloudflare workers already; `wrangler deploy` preserves them, but verify
per worker after a deploy per `AXON_FIX_BRIEF.md` §5.3.5.

## Things this repo's CI and `wrangler.toml`s deliberately hold the line on

- Every queue consumer's `max_batch_size = 1`. Raising it reintroduces the
  subrequest-ceiling bug that took down `mastery-content` — see
  `AXON_FIX_BRIEF.md` §3.3 and §10.
- `wrangler deploy` (not a raw bundle upload) for every deploy, so secrets
  are never at risk of being wiped.

See `docs/AXON_FIX_BRIEF.md` §10 for the full "do not do these" list.
