# Deployment and rollback runbook

## Before a canary

1. Confirm the provisioned D1 database, R2 bucket, KV namespace, processing queue, and dead-letter queue still match the checked-in resource IDs in `wrangler.jsonc`.
2. Set `GOOGLE_API_KEY` on `axon-document-vision`. Set `GOOGLE_API_KEY`, `SUPABASE_SERVICE_ROLE_KEY`, `TAVILY_API_KEY`, separate `AXON_INTERNAL_TOKEN` and `AXON_ADMIN_TOKEN` values, and `AXON_PSEUDONYM_KEY` on `axon-intelligence`. Never place these values in Wrangler config or GitHub logs.
3. Obtain explicit zero-data-retention attestations for Gemini and Workers AI document processing. Only then set `GEMINI_PRIVACY_MODE`, `WORKERS_AI_PRIVACY_MODE`, and `AXON_VISION_PRIVACY_MODE` to `zdr` in the audited deployment configs.
4. Set the current audited Gemini input/output USD-per-million-token rates; AXON will not report itself ready when cost telemetry is unconfigured.
5. Apply D1 migrations remotely, then `POST /v1/admin/capabilities/probe` with the admin bearer token and retain its complete JSON response as `capability-probe.json`. The endpoint persists the individual Gemini, Tavily, and document-vision results in D1 and emits the exact certification artifact; a false flag is a release blocker.
6. Assemble the private evidence bundle described in `release-evidence.md`, populate `certification/release.json` with its exact digests, and set `AXON_RELEASE_EVIDENCE_DIR` plus the stage being promoted in `AXON_RELEASE_TARGET_STAGE`. Never copy the example values without evidence. The preflight derives counts and quality metrics from the reviewed records rather than trusting the JSON summary.
7. Run `npm ci`, generated-binding checks, the repository typecheck/tests, both Worker checks, both dry-runs, `npm audit`, and `npm run release:preflight`.
8. Deploy `axon-document-vision` first and verify its private `/health` through a service binding. Deploy `axon-intelligence` only after that target exists. Neither Worker is part of the automatic `main` deployment matrix until certification is complete.

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
