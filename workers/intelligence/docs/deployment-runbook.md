# Deployment and rollback runbook

## Before a canary

1. Confirm the provisioned D1 database, R2 bucket, KV namespace, processing queue, and dead-letter queue still match the checked-in resource IDs in `wrangler.jsonc`.
2. Set secrets with Wrangler: `GOOGLE_API_KEY`, `SUPABASE_SERVICE_ROLE_KEY`, `TAVILY_API_KEY`, separate `AXON_INTERNAL_TOKEN` and `AXON_ADMIN_TOKEN` values, `AXON_PSEUDONYM_KEY`, and, when document automation is enabled, `AXON_VISION_TOKEN`.
3. Obtain explicit zero-data-retention attestations for the Gemini and document endpoints. Only then set their privacy modes to `zdr` in the deployment environment.
4. Set the current audited Gemini input/output USD-per-million-token rates; AXON will not report itself ready when cost telemetry is unconfigured.
5. Apply D1 migrations remotely, run the authenticated capability probe, and retain its rows as deployment evidence.
6. Populate `certification/release.json` from reviewed benchmark and rollback evidence. Never copy the example values without evidence.
7. Run `npm ci`, `npm run types`, `npm run check`, `npm run eval`, `npm audit`, and `npm run release:preflight`.

## Rollout

Promote through benchmark, internal, 1%, 5%, 25%, 50%, and full stages. At every stage compare correction rate, mark-attribution accuracy, unsupported-claim escape rate, p95 latency, provider failures, and cost against the current immutable revision. Halt on any threshold implemented in `src/deployment/rollout.ts`. Shadow outputs are never shown and store only verification metrics plus a hash.

## Rollback

Select a previously recorded immutable config revision and deployment SHA. Do not edit the current revision in place. Deploy the prior Worker/config pair, set its rollout state to `ROLLED_BACK`, verify `/health`, run one synthetic tutor request and one synthetic document job, and confirm D1 trace provenance points to the restored revision. Student-visible fields already committed remain immutable; corrections are additive events.

## Incident rules

- Open provider circuit: stop model traffic; do not relax privacy or switch to an unapproved model.
- Retrieval outage: withhold current claims; stable canonical knowledge may continue.
- Vision outage or unreadable page: queue student review/rescan; do not infer missing text.
- Repeated document job failure: after three attempts the page is moved to review; infrastructure-level crashes fall through to the paper dead-letter queue.
- Schema, semantic, or contradiction failure after one repair: withhold the answer.
- Suspected secret exposure: rotate the secret, invalidate internal callers, and review hashed rate/idempotency records and redacted traces.
