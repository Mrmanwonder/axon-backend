# Specification implementation status

## Implemented and verified locally

- Worker-first tutor orchestration with immutable prompt/schema hashes, checked-in routing, strict privacy policies, provider abstraction, structured output, deterministic and model verification, one repair, controlled failure, and pedagogy rendering.
- Evidence and claim graphs with explicit information classes, provenance, contradiction checks, teacher-mark protection, current-fact retrieval requirements, prompt-injection fencing, and insufficient-evidence responses.
- Deterministic calculator, polynomial equivalence, linear solving/step validation, dimensional units/conversion, formula/molar-mass chemistry, and equation balancing.
- Immutable R2 paper ingestion, SHA-256 deduplication, quality policy, non-generative conditioning contract, layout/question/ink/mark/read stages, confidence escalation, trusted fields, student review, additive corrections, and active-learning prioritization.
- D1 provenance for deployments, routes, prompts, schemas, traces, evidence, claims, provider health, document stages, evaluation, shadow results, and capability probes.
- HMAC-pseudonymized student identifiers, bounded bodies, internal bearer authentication, D1 rate windows, correction idempotency, redacted trace evidence, and fail-closed public retrieval minimization.
- Canary/rollback policy, shadow-mode storage rules, minimum benchmark release gates, CI, migrations, and operational runbooks.
- Live Supabase routing now targets `gemini-3.5-flash-lite` for all six enabled stages, disables training on every route, records stage-specific thinking levels, and identifies the active paper-feedback contract as `paper_feedback.v2`.
- The intelligence Worker uses generated Wrangler bindings checked into source, the 2026-09-24 compatibility date, and current Cloudflare tooling; clean CI verifies that generated bindings remain in sync.

## Intentionally blocked from production release

The repository contains no fabricated certification. Cloudflare D1, R2, KV, and queue resources exist and their IDs are versioned, but the release remains blocked until production secrets are installed on `axon-intelligence`, zero-retention is contractually attested, the document-vision service is deployed, live Gemini/Tavily/vision probes pass, a reviewed scanner set reaches at least 100 papers and 1,500 questions, a hand-reviewed tutor set reaches at least 500 cases, a rollback drill is evidenced, and staged canary metrics pass. Physical scanner accuracy and provider behavior cannot be proven by local unit tests.
