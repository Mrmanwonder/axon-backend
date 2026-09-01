# Why this repository exists

Until 2026-09-01, the `mastery-*` Cloudflare Workers that make up Axon's
extraction pipeline existed **only as deployed bundles**. There was no
source repository. Every fix — including the four real bugs fixed on
2026-08-31 (see `AXON_FIX_BRIEF.md` §3) — was made by:

1. Pulling the deployed bundle down via the Cloudflare API.
2. Grepping through ~750KB of bundled, minified JavaScript to find the
   worker's own ~4-8KB of code (everything past the `// src/index.ts`
   marker).
3. Hand-editing that JavaScript.
4. Re-uploading it as a new Worker version.

This is how three different indentation variants of the same shared
`callModel` retry logic ended up living independently inside six different
deployed bundles (`AXON_FIX_BRIEF.md` §4.D4) — and part of how a stale
`mastery-reconcile` deploy ended up missing a `SELF_QUEUE` binding fix that
every sibling worker already had (§4.D2). None of that was a decision
anyone made; it's what happens when "the codebase" is a live production
bundle nobody can diff, review, or type-check.

## What this repository is

The reconstructed source, mined back out of those live bundles on
2026-09-01 (see `AXON_FIX_BRIEF.md` §2 and §5.3 for the method) and
organized into the `shared/` + `workers/*` structure that brief lays out.
Every module boundary here (`shared/openrouter.ts`, `shared/worker.ts`,
`shared/r2.ts`, `shared/prompts/*.ts`, etc.) is the **real** file boundary
recovered from esbuild's own `// ../shared/foo.ts` source-comments in the
bundles — not a guess at how the code should have been organized.

Where a worker's deployed copy of shared code had drifted from the others
(reconcile's stale `worker.ts`, without the manual re-enqueue retry path;
the differing 401/403 error message text between `mastery-triage` and the
other five), the canonical version checked in here is `mastery-triage`'s —
per `AXON_FIX_BRIEF.md` §5.3.2 — and the drift itself is called out in a
comment at the point it was found, not silently resolved.

## Do not go back to hand-editing deployed bundles

If you're about to pull a bundle via the Cloudflare API to patch something
in production: stop. Make the change here, run `npm run typecheck`, run
`npm test`, dry-run the affected worker (`wrangler deploy --dry-run` from
`workers/<name>/`), then deploy through CI or `wrangler deploy` from a
checkout of this repo. `AXON_FIX_BRIEF.md` §10 has the full list of things
not to do while you're in here — most of them were learned the hard way,
on 2026-08-31, in production.

## Before the first deploy of `mastery-crop`

`workers/crop` is the one worker in here that has never existed in
production, so unlike the other eight it has no queues waiting for it.
Deploying it against a Cloudflare account that has not been prepared will
fail at bind time, and — worse — deploying `mastery-structure` *first*
would leave every run stalled in `cropping` with nothing consuming the
queue. Create the queues before either deploy:

```sh
npx wrangler queues create crop-queue
npx wrangler queues create crop-dlq
```

Then deploy in this order, which is the order that keeps a run able to
finish at every point in between:

1. `workers/crop` — the consumer exists, and nothing produces to it yet.
2. `workers/structure` — starts producing to it.

Deploying them the other way round is what strands papers.

The database side is already applied (`db/migrations/20260901120000_*`
through `20260901120200_*`): the `cropping` run status, `paper_page.
crop_status`, `advance_after_crop`, `apply_region_crops`, and the sweep
clauses that close out a crop left running under a swept run. A structure
worker deployed before those migrations would call an
`advance_after_structure` that does not know about cropping — but since the
migrations are live, that ordering problem no longer exists.

`workers/crop` also carries a runtime check the other workers do not:

```sh
npm run smoke -w @mastery/crop
```

It decodes, cuts and re-encodes a real page through the exact codec path
the Worker uses. It exists because `wrangler deploy --dry-run` proved the
worker bundles and the bundle was still wrong — libwebp's embind binding
requires the complete options struct and throws only when a real image is
encoded. Run it after any change to `workers/crop/src/codecs.ts`. It reads
fixtures from a checkout of the Axon-Site repo beside this one, which is
why it is not in `npm test`.
