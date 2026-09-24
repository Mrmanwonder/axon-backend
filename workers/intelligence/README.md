# AXON Intelligence System

Worker-first implementation of AXON v3: evidence before interpretation, verification before delivery, deterministic education tools where possible, privacy-preserving provider routing, and reproducible evaluation.

## Local setup

```sh
npm install
npm run types
npm run db:migrate:local
npm test
npm run dev
```

Create secrets with `wrangler secret put GOOGLE_API_KEY`, `wrangler secret put SUPABASE_SERVICE_ROLE_KEY`, `wrangler secret put TAVILY_API_KEY`, `wrangler secret put AXON_INTERNAL_TOKEN`, `wrangler secret put AXON_ADMIN_TOKEN`, and `wrangler secret put AXON_PSEUDONYM_KEY`. Document automation additionally requires `AXON_VISION_TOKEN`. Every `/v1/*` call must carry a Bearer credential; `/v1/admin/*` requires the separate admin token. Never place student data or secrets in configuration files.

## Worker endpoints

- `GET /health`
- `POST /v1/tutor`
- `POST /v1/papers/ingest`
- `GET /v1/papers/pages/:pageId`
- `POST /v1/papers/pages/:pageId/review`
- `POST /v1/papers/pages/:pageId/commit`
- `POST /v1/corrections`
- `POST /v1/insights/observations`
- `GET /v1/insights/patterns`
- `POST /v1/admin/capabilities/probe`
- `GET /v1/admin/provider-health`
- `GET /v1/admin/readiness`
- `GET /v1/admin/active-learning`
- `POST /v1/admin/active-learning/:id`

Paper uploads are immutable R2 artifacts and queued for bounded background processing. Tutor responses are rendered only from claims that pass schema, evidence, contradiction, retrieval, and tool-use checks.

Production release is deliberately fail-closed. See `docs/implementation-status.md`, `docs/vision-provider-contract.md`, and `docs/deployment-runbook.md`. `npm run release:preflight` must remain blocked until real resource configuration and immutable certification evidence are present.
