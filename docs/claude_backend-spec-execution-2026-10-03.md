# Backend spec execution audit — 3 October 2026

Evidence scope: repository `Mrmanwonder/axon-backend`, main `31a11cf87a731bc1087d87f2568c114d03196c06`; full Linear issues/relations and current discussion for AXO-125, AXO-126, AXO-131, AXO-138, AXO-41 and AXO-57. Root owns production, database and legal evidence. This report does not certify release.

## Delivered in this draft

AXO-138: `workers/content/src/index.ts` previously loaded the first crop/mask whenever available, then skipped the full-page branch. A multi-page region could therefore lose its continuation before model extraction. The worker now plans sources before loading images: crop for one distinct page; all available full-page sources for a continuation. Each page retains its own dimensions/frame. Four pure regression tests cover a two-page continuation with a crop, the continuation coordinate mapping, repeated same-page bands, absent crop and an empty span list. No prompt, model, schema, routing or historical answer change.

Limit: full-page input may contain neighboring questions; the existing target-label/region contract remains in force. Existing missing-image/dimension handling still needs a separate fidelity audit. This fix cannot make an unavailable page available.

WP0: AXO-126's superseded Flash-Lite-default instruction was surgically replaced in Linear by a reference to revised AXO-125 quality-first routing and its evaluation/provider/release gates. No route was changed.

## WP-F — Tutor / AXO-19, 35–40, 126

Authenticated gateway, server-hydrated student/paper context and tests, shared model client, structured reasoning, verifier plus at most one repair, strict privacy gate, public minimized Tavily retrieval, redacted audit persistence and tutor-only deployment plumbing already exist. Do not duplicate backend PR #149 or #143.

Open release conditions: privacy/provider eligibility decision and attestation (AXO-129); owner-managed secrets and rollout; verified nonempty D1/runtime binding state; approved feature-flag scope; human-reviewed curriculum corpus; grounded authenticated production UI/API smoke, trace with non-null cost and deletion parity; measured latency/timeout evidence. `GEMINI_PRIVACY_MODE=unverified` refuses private student chat through `STUDENT_CHAT_STRICT`; paid account alone does not attest retention.

Concrete remaining code/accounting gaps:
- `rate-limit.ts` implements a minute window, with Tutor 60/minute in `index.ts`. This is not a per-student daily message allowance or daily spend reservation. No cap ceiling or business policy was invented.
- `index.ts` estimates Tutor cost only when optional environment input/output prices and trace usage exist. Absent prices can produce no estimated cost; this differs from mandatory non-null-cost acceptance.
- The orchestrator aggregates successful initial/verifier/repair token usage in normal verified or withheld-result paths. Repair transport failure emits a trace without accrued initial/verifier tokens; verifier schema/transport exceptions report zero usage even if generation already returned billable output. These cases need ledger tests and careful accounting rather than a blanket claim all verifier calls are omitted.
- Runtime expected model and 8s/12s timeout constants remain in `runtime.v3.ts`; removing duplicated configuration and certifying latency follows AXO-125, not a new model choice here.

## WP-G — Quality routing / AXO-125, 41–44

One shared client/provider configuration and static-prefix ordering exist. Actual explain Worker still processes one region per queue message; per-question batching is explicitly not built and remains eval-gated in latest AXO-125 discussion. Confidence-driven rereading, cache benefit measurement, complete pricing and release comparison remain open.

Latest AXO-41 evidence reports one synthetic explain run `240bc976-427a-4730-ae36-aafe8eafbee0`: 24 synthetic cases, 22 scored per candidate; recorded `passed:false`, no route change. It is not reviewed multi-stage release evidence. 3.6 pricing, human cause labels and permission to send stored student cases to a judge remain owner decisions. No paid calls or eval was initiated.

Tutor draft `workers/intelligence/evals/drafts/axo40-tutor-golden-v0.1-draft.json` has 80 authored cases: 20 each Cambridge/CBSE/IBDP and 20 safety cases; `status:draft`, `reviewedBy:null`. Its note says original wording, no reproduced exam/scheme text. Repository tests keep it separate from certification cases. This is a readiness artifact, not legal permission or academic review. Real student papers require consent/provenance/retention and protected-content permission before ingestion or paid judging.

## WP-J — Guardian verification / AXO-56–59

Generic server callback exists in `workers/api/src/guardian_verification.ts`: HMAC-SHA256 over timestamp/body, freshness window, constant-time signature comparison; service-role RPC records the callback and owns state binding/replay/expiry. Eight synthetic Worker tests exist. These prove only the generic test protocol.

AXO-56 jurisdiction, provider, assurance claims, adulthood/relationship distinction, expiry and privacy/retention choices remain owner/counsel decisions. No provider is selected or configured. A future selected provider's JWT/JWKS/issuer/audience protocol may differ from generic HMAC and must be implemented/tested to its actual specification. AXO-58 UI states and AXO-59 provider sandbox/production proof/disclosures remain separately open. A test fixture naming a provider is not integration evidence.

## Extraction trace / AXO-138–141

Current schema/prompt distinguishes question, flat student answer, teacher marks and teacher remarks; answer block carries math/prose segments, working/final/crossed-out roles, pen annotations, bbox and confidence. Raw answer text stays alongside structured content.

Ordinary value boxes are mapped through model crop/mask/page frames to actual page pixels. `answer_block` is written directly from model data, so its segment bboxes remain in model-image coordinates. That needs a shared explicit page/coordinate contract and compatibility strategy with the renderer before correction; no coordinate rewrite was guessed. The current schema has no explicit table-cell representation. Review UI consumption and table rendering are owned by the root's academic/design work.

The screenshot cause remains unverified without original capture, conditioned image, region boundary, model output and persisted content for the exact examples. Source finding alone does not establish OCR accuracy. Preserve student-confirmed work and never replace incorrect work with a correct solution.

## Pending explanations / AXO-124

The explain Worker waits for `student_confirmed_at`; unconfirmed messages return with status pending. API `/review-complete` invokes `begin_explanations` and dispatches returned region IDs; retry endpoint queues failed eligible regions only. Pending confirmed rows require inspecting the deployed SQL function, run status, current statuses and queue delivery evidence before choosing a cause. Do not diagnose from this repo's legacy SQL snapshot alone or assume the run is terminal. Root database audit owns that investigation.

## Superseded work and verification

AXO-131 latest owner decision (2026-10-03 00:59 UTC) cancels it and closes PR #150 unmerged as superseded by AXO-110. Honored; no retry optimization reimplementation.

Local command execution failed because the environment is offline. No local typecheck/test/dry-run is claimed. Required exact-head GitHub Actions gates: generated types, typecheck, workspace tests, registry tests, cross-repo contract parity, prompt gate and dry-runs for every Worker including both intelligence profiles. No merge/deploy/migration is authorized by this report. Draft must remain reviewable until checks pass and owner release gates are met.
