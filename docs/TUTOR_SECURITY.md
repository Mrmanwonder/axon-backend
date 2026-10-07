# AXON Tutor Security, Privacy and Trust Boundary

**Status:** implementation contract for AXO-35 / AXO-106  
**Last reviewed:** 2026-09-27  
**Applies to:** `mastery-api /tutor`, `axon-intelligence /v1/tutor`, Tutor Gemini calls, Tavily retrieval, Tutor D1 audit/provenance, and any future student-facing UI that consumes this contract.

This document describes the implemented security boundary. It is not a statement that Tutor is released. The production deployment plan intentionally excludes `axon-intelligence` and `axon-document-vision` until AXO-13 certification, privacy attestations, live probes, rollback evidence and the remaining release gates pass. The public API must fail closed while the private Intelligence binding is unavailable.

## 1. Security objectives

Tutor must satisfy all of these invariants simultaneously:

1. **Exact active-student authority.** A signed-in guardian session may have multiple owned profiles, but a daily Tutor request is authorized only for the one active Student Mode capability bound to that exact auth session.
2. **No caller-authored authority.** Browser-supplied board, grade, topic, subject, paper references, evidence or retrieval context are never accepted as facts merely because they arrived in an authenticated request.
3. **Minimum evidence.** Only the smallest verified context needed for the task is assembled. Paper/teacher claims require paper/teacher evidence; current facts require retrieval; calculations require deterministic tool evidence.
4. **No public-web exfiltration.** Tavily receives public academic search context only. It never receives student answers, teacher remarks, names/emails, account IDs, auth material, signed URLs or arbitrary chat prose.
5. **No source-class promotion.** Web material cannot become an official marking scheme; generated prose cannot become an observed fact; caller-provided evidence cannot become verified evidence.
6. **Fail closed on uncertainty.** Missing evidence, unavailable retrieval, schema failure, non-compliant model routing or failed verification produces a controlled failure/withholding state, not an invented answer.
7. **Privacy-compliant model routing.** Student chat/document data may go only to a provider endpoint admitted by the configured privacy policy. No fallback may silently relax privacy.
8. **No raw private audit logging.** Operational telemetry stores provenance/verification metadata and public reference evidence, but not raw Tutor prompts, student/teacher/paper values, request-derived tool values or generated claim prose that may repeat private content.
9. **Bounded work.** Request size, model timeouts, repair attempts, retrieval source count, network response size and rate limits are bounded.
10. **Reproducibility.** Model, prompt, schema, config/deployment identity, tool use, verification status, latency and token metadata are recorded without storing private prompt text.

A change that weakens one of these invariants is a security change and requires review, tests and release evidence.

## 2. Current deployment state

The repository contains a complete private Intelligence worker and document-vision worker, but they are intentionally excluded from the production deploy matrix pending AXO-13 certification.

Current public API behavior:

- `POST /tutor` exists in `mastery-api`.
- It requires an authenticated Supabase user session.
- It verifies `student_scope_state()` before student/resource lookups.
- It verifies paper ownership when `paperId` is supplied.
- It hydrates curriculum identity from authenticated database records.
- It rejects caller-authored evidence as an authority source.
- If `INTELLIGENCE` or `AXON_INTERNAL_TOKEN` is unavailable, it returns 503 rather than falling back to a browser/model path.

A future production release must add the private service binding only as part of the certified deployment plan. The absence of the binding is a release gate, not an error to bypass.

**Internal stage (AXO-126, 2026-10-02).** `axon-intelligence` is now deployed from `workers/intelligence/wrangler.tutor.jsonc`, a Tutor-only profile with no document-vision binding and no paper queue, and `mastery-api` and `mastery-sweep` bind to it. This is not a release. Three things keep it internal:

- `/tutor` asks the database, as the signed-in guardian, whether `tutor_enabled` is on (`guardian_feature_flag`, default off). Anything other than an explicit `true` is refused, and an unreadable flag fails closed.
- `GEMINI_PRIVACY_MODE` is `paid_no_training` (owner decision, 7 Oct 2026; AXO-126). The key is on the Google AI Studio paid tier: Google does not train on the data and keeps it for a bounded period, which the Privacy Policy declares. It is not zero retention, and the capability probe's `zero_data_retention` check still fails for it; release readiness needs `no_training`. `zdr` remains a separate attestation the owner makes only with a written Google arrangement. Any other value fails closed with `NO_COMPLIANT_PROVIDER`.
- The remaining release gates (AXO-13 certification, probes, rollback evidence) are unchanged.

Deletion parity: deleting a paper, or erasing a student, queues `tutor_purge`; the sweep worker calls `POST /v1/admin/purge` on `axon-intelligence` (admin token), which deletes the paper's `ai_trace` rows and the claim and evidence rows under them. Tutor provenance is keyed by paper only, so a question asked with no paper carries no student link.

## 3. Architecture and trust boundaries

```text
Student browser
    |
    | HTTPS + Supabase bearer/session
    v
mastery-api (/tutor)
    |-- Supabase user client --> Student Mode + student/paper/curriculum records
    |
    | private Worker service binding + AXON_INTERNAL_TOKEN
    v
axon-intelligence (/v1/tutor)
    |-- D1: metadata/provenance audit
    |-- KV: bounded public retrieval cache
    |-- Gemini paid tier (no training, declared retention): private student reasoning
    |-- Tavily: public-only current/source retrieval
    |-- document-vision: private service binding for document pipeline only
    v
Structured verified Tutor response
```

### Boundary A — browser -> mastery-api

**Trusted only after validation:** authenticated session identity and the currently active Student Mode state resolved server-side.

**Untrusted input:** message text, requestId, depth preference, studentId, paperId and every optional client hint.

Controls:

- body size cap;
- exact Student Mode check before sibling/resource lookup;
- paper ownership check;
- caller evidence is not forwarded;
- board/grade/topic/retrievalContext are not accepted as authority;
- subject text may only select an exact stored subject identity;
- unrecognized subject remains unknown.

### Boundary B — mastery-api -> Supabase

The request uses the caller's authenticated Supabase client for student-scoped reads. RLS and Student Mode are independent server-side enforcement, not UI assumptions.

The API must not use service-role access to turn a client-supplied student/resource identifier into authority. Service-role access is reserved for operations that already have a verified scoped identity and need privileged pipeline behavior.

### Boundary C — mastery-api -> axon-intelligence

This is a private service boundary, not a public endpoint.

Allowed request fields are schema-validated. The API supplies:

- active student identifier;
- raw student message for the private reasoning provider;
- bounded request/depth metadata;
- authenticated server-derived curriculum context;
- authorized paper identifier when applicable;
- **server-hydrated paper evidence** (AXO-36, `workers/api/src/tutor_evidence.ts`): for an authorized paper, and optionally one question on it, the gateway loads the latest reviewable run's regions through the caller's own RLS-scoped client, with explicit `student_id`/`paper_id`/`run_id` joins. It forwards only label, question text, the student's answer, and the teacher's marks and remark, as `paper`/`teacher` evidence with `primary` authority. An `unsure` read is `unverified`; a student-confirmed one is `verified`; unreadable regions are omitted. At most 12 regions, each text field bounded. No names, image keys, boxes or signed URLs.

These are **Private schoolwork** (§4). They reach Gemini only through the `STUDENT_CHAT_STRICT` route, which admits `gemini-zdr` alone. With `GEMINI_PRIVACY_MODE=unverified`, the orchestrator returns a controlled failure and nothing is sent. Teacher marks are the fact being explained and are never re-graded.

The browser cannot provide verified Evidence objects or an internal `retrievalContext`. A body-supplied `evidence` field is dropped at the gateway (`test/tutor-evidence.test.ts`).

The internal token/service binding must never be exposed to the browser, logs, Tavily or model content.

### Boundary D — axon-intelligence -> Gemini

Private student chat may be sent only when the route satisfies `STUDENT_CHAT_STRICT`:

- admitted provider: `gemini-zdr`;
- retention: zero;
- training: disallowed;
- raw logging: disallowed;
- image input: prohibited for chat;
- fallback may not relax privacy.

If ZDR/privacy attestation is absent, routing fails closed with `NO_COMPLIANT_PROVIDER`.

The provider receives fenced evidence and a versioned prompt/schema. Model output is untrusted until schema and evidence verification pass.

### Boundary E — axon-intelligence -> Tavily

Tavily is an external **public-web** service. It is never a general-purpose Tutor context sink.

Search eligibility is intent-gated to current/source tasks. The outbound query is constructed from:

1. authenticated public curriculum identifiers (provider/programme/stage/subject/code); and
2. one closed, server-defined intent facet such as “official syllabus specification”.

Raw user prose is used only to select a constant facet; no substring of the message is copied into the query.

The final DLP gate rejects:

- email/contact-like values;
- UUID-like identifiers;
- answer/teacher/private markers;
- auth/token/key language;
- any URL in query material.

Search/extract URLs are validated before use:

- HTTPS only;
- no credentials;
- no localhost, `.local`, `.internal` or metadata hosts;
- no private/reserved IPv4;
- no literal IPv6;
- no signed/capability query parameters;
- trailing-dot hostnames are canonicalized before checks;
- extract may use only URLs accepted from the same validated search result set.

Every web payload is treated as untrusted reference content. Retrieval evidence remains a distinct evidence class and cannot populate official marking-scheme provenance.

### Boundary F — Tutor -> D1 audit/provenance

D1 is operational provenance, not a transcript store.

Allowed persistent data:

- trace ID and stage/capability;
- deployment/config/pipeline identity;
- provider/requested/served model;
- thinking level;
- prompt/schema IDs and hashes;
- tool names and retrieval-used flag;
- verification/repair status;
- latency, token counts and configured cost estimate;
- transport/schema/semantic success flags;
- artifact hashes;
- public/canonical retrieval and stable-knowledge evidence needed for reproducibility;
- redacted evidence nodes, claim type/risk/status and claim<->evidence graph edges.

Forbidden raw persistent data for Tutor audits:

- raw student message;
- student answer/paper/teacher/axon-db evidence values;
- raw deterministic tool values derived from student requests;
- generated claim prose that may repeat private content;
- auth/session tokens;
- signed URLs or secrets.

Private/request-derived evidence values are stored as redacted metadata only. Tutor claim text is stored as `[redacted:student-chat]`; evidence IDs and graph edges preserve audit structure.

## 4. Data classification and allowed destinations

| Class | Examples | Supabase/R2 | Gemini ZDR | Tavily | D1 Tutor audit | General logs |
| --- | --- | --- | --- | --- | --- | --- |
| Public academic | provider/programme/stage labels, public subject code, official public URL/content | allowed | allowed | allowed when needed | allowed | metadata only |
| Account/identity | guardian/student IDs, name, email | system of record only | minimize; only if genuinely required by future feature | **never** | hashes/redacted only | **never raw** |
| Private schoolwork | chat, answers, paper text/crops, teacher remarks/marks | system of record/private object store | allowed only under strict privacy route | **never** | redacted only | **never raw** |
| Request-derived private | calculator/tool output derived from student request, generated claim that may quote schoolwork | only if product record explicitly requires it | allowed inside request lifecycle | **never** | redacted only | **never raw** |
| Public retrieval | Tavily result URL/title/content | optional verified evidence store | allowed as evidence | source | allowed for reproducibility | metadata only |
| Secrets/capabilities | JWT, service token, API keys, signed URLs | secret/binding only | **never** | **never** | **never** | **never** |
| Operational metadata | hashes, model IDs, prompt IDs, timing, status | allowed | n/a | n/a | allowed | allowed |

If a value belongs to more than one class, apply the most restrictive class.

## 5. Context authority rules

### Curriculum identity

Provider/programme/stage/grade are read from the active student's normalized profile. Client hints cannot override them.

### Subject identity

Order of authority:

1. verified paper-bound subject identity/snapshot when an authorized `paperId` is present;
2. exact match against the active student's stored subject rows;
3. unknown.

Free text must not create a new subject identity.

### Paper/question evidence

A paper ID is only an authorization reference, not the evidence itself. Paper-feedback/work-check/mistake-diagnosis tasks must receive server-loaded paper/student/teacher evidence before the model call. If required evidence is absent, the orchestrator returns `insufficient_evidence` without inventing a marking reason.

### Official scheme evidence

Official marking-scheme claims are a separate authority class. They may enter Tutor only after AXO-12 resolves exact assessment + exact question against an authorized ready stored source. Tavily is not a substitute for official scheme evidence, and Tier-2 paper explanation keeps web search disabled.

### Prior model text

Old assistant/model prose is never promoted to verified evidence by repetition. Only source-backed/canonical/tool-verified records can satisfy an evidence-required claim.

## 6. Model-output trust doctrine

A successful provider response is not yet a successful Tutor answer.

Required sequence:

1. provider transport success;
2. structured schema validation;
3. deterministic evidence/claim verification;
4. contradiction and source-class checks;
5. risk-dependent model verifier where configured;
6. at most one bounded repair attempt;
7. revalidation;
8. render only supported/partially-supported structured reasoning.

Any terminal failure returns a controlled/withheld response.

Important checks include:

- stable claims must be backed by canonical stable knowledge;
- calculation claims require verified deterministic tool evidence;
- retrieved claims require verified retrieval/official-source evidence;
- paper claims may not invent teacher intent;
- source IDs must actually exist in the evidence graph;
- hint mode may not reveal a full solution when the requested behavior is a hint.

## 7. Prompt-injection and untrusted-content handling

Untrusted text includes:

- student messages;
- paper/teacher text;
- public web pages;
- model-generated intermediate content.

Controls:

- evidence is fenced before Gemini;
- external web content is labeled untrusted reference data;
- model/tool instructions found inside evidence are not authority;
- browser-authored Evidence objects are downgraded/rejected as verified authority;
- output cannot create new evidence IDs and pass verification merely by citing them;
- high-risk outputs receive independent verification;
- no external tool may widen its own data access from model-authored arguments.

## 8. Threat register

| Threat | Example | Required control / expected result |
| --- | --- | --- |
| Cross-sibling access | Student A session asks Tutor about Student B | exact live Student Mode check before resource/model work; 403 |
| Revoked/stale scope | old tab replays a prior student ID | scope state fails closed; no resource/model work |
| Caller context forgery | fake board/grade/topic/subject | server profile hydration; unknown subject stays unknown |
| Caller evidence forgery | browser posts “official” Evidence object | public gateway does not forward caller evidence; inbound internal evidence is downgraded when source is privileged |
| Public-web exfiltration | private answer embedded in “current syllabus” question | Tavily query uses server public context + constant facet only |
| Name/contact leakage | ordinary name/email in arbitrary prose | raw message never serialized into Tavily request |
| Signed/private URL exfiltration | R2/Supabase signed URL in chat/result | query DLP + public URL validator; extract refuses it |
| SSRF/private target | localhost/private IP/metadata URL in result | rejected before extract |
| Prompt injection from web | page says “ignore system prompt” | web is untrusted evidence; verifier/source rules still apply |
| Source promotion | blog treated as official marking scheme | evidence classes + authority scoring + exact AXO-12 scheme path |
| Hallucinated marking reason | model guesses why marks were lost | paper/teacher evidence required; otherwise insufficient evidence |
| Calculation bypass | model does arithmetic without tool evidence | calculation claim rejected |
| Privacy fallback | ZDR route unhealthy | no relaxed fallback; controlled failure |
| Raw telemetry leak | claim echoes student answer | claim/tool/private evidence redacted before D1 persistence |
| Secret leak in logs | JWT/internal token in error | log path/error class only; never request headers/body |
| Cost/loop amplification | repeated schema repair/tool loop | bounded body, timeout, retrieval count, one repair, rate limit |
| Stale/current fact | current rule answered from memory | current/source intent requires retrieval or controlled failure |
| Deployment mismatch | public API points to uncertified Intelligence | service binding withheld until release gate; API fails 503 |
| Shadow-model privacy drift | candidate model receives private data without attestation | shadow execution permitted only under the same admitted privacy mode |

## 9. Logging, retention and deletion

### Application/Worker logs

Do not log raw request bodies, auth headers, signed URLs, paper text, student answers or teacher remarks.

Structured logs may contain event name, route/path, trace ID, coarse error class and non-private operational metadata.

### D1 Tutor provenance

D1 audit rows are designed for reproducibility without transcript retention. Private values are redacted before persistence. Public retrieval/canonical evidence may be retained as provenance.

The database currently retains core trace/evidence graph rows until an explicit operational retention/delete policy is applied. Because those rows must contain no raw private Tutor content, retention duration must not be used as a substitute for redaction.

Provider observations are operational health data and are periodically pruned by the scheduled maintenance path.

### Provider retention

Production Tutor release requires documented zero-data-retention/privacy attestation for Gemini and the private vision path. A configuration string alone is not sufficient release evidence.

### Legal/public disclosure

AXO-27 owns reconciliation of product disclosures and legal policy with this actual architecture. Security implementation must not claim a data use/retention behavior broader or narrower than the final public disclosure.

## 10. Availability, abuse and cost controls

- request body is bounded;
- Tutor is authenticated and student-scoped;
- per-route rate limiting is enforced inside Intelligence;
- model calls have deadline budgets;
- repair is capped at one attempt;
- Tavily response bytes and source counts are bounded;
- provider circuit-breaker state prevents uncontrolled retries;
- cost is estimated only when explicit audited rate inputs exist;
- no silent fallback may change model/provider privacy policy.

A temporary provider/retrieval failure produces a typed/controlled response, not an unbounded retry loop.

## 11. Release gates

Do not expose production Tutor just because unit tests pass.

Release requires, at minimum:

1. AXO-13 private reviewed benchmark evidence and model promotion decision;
2. Gemini ZDR/privacy attestation and configured release secrets;
3. document-vision privacy attestation if that service is part of the released path;
4. Tavily privacy/tool tests (AXO-37) green;
5. this no-raw-audit regression (AXO-106) green;
6. AXO-36 verified context work needed for the released Tutor surface;
7. AXO-38 pedagogy/uncertainty/provenance acceptance;
8. AXO-39 UI/lifecycle/accessibility;
9. AXO-40 cross-curriculum safety/adversarial evaluation;
10. live capability probes;
11. rollback and staged-canary evidence;
12. legal/public disclosure reconciliation where required.

The deploy matrix must add `axon-intelligence` / `axon-document-vision` deliberately. Do not bypass the gate with a one-off manual deploy.

## 12. Verification evidence map

| Invariant | Primary automated evidence |
| --- | --- |
| Student Mode exact scope | API Tutor Student Mode abuse suite |
| Caller context cannot override normalized profile | AXO-37 gateway-context regression |
| Raw chat never becomes Tavily query | AXO-37 Tutor adversarial retrieval test |
| Unsafe URLs never reach Extract | AXO-37 Tavily URL serialization tests |
| Unattested provider fails closed | Tutor/reliability tests |
| Invalid evidence cannot become verified | claim/evidence verification tests |
| One bounded repair | Tutor orchestrator tests |
| Raw Tutor text not persisted | AXO-106 D1 operational regression |
| Prompt/model/schema provenance | operational/provenance tests + D1 readiness |
| Worker packages remain deployable | CI Wrangler dry-run matrix |

## 13. Security review checklist for future changes

Before merging a Tutor change, answer all of these:

- Does it add a new source of student/private data?
- Does it change who may select a student/resource?
- Does it let the browser assert a new “verified” field?
- Does it add a new external network destination?
- Could any new value enter Tavily?
- Could any new value be persisted/logged raw?
- Does it weaken ZDR/privacy routing or introduce a fallback?
- Does it change evidence authority/source classification?
- Does it add a model/tool loop or retry?
- Does it affect signed URLs/secrets/capabilities?
- Does it change prompt/schema/model identity without versioning?
- Are cross-student, prompt-injection, privacy and controlled-failure tests updated?
- Does the change require AXO-27 disclosure updates?
- Does it alter any release gate?

If any answer is yes, the PR must state the new data flow and include corresponding regression evidence.

## 14. Related tracked work

- **AXO-35:** Tutor foundation and this threat model.
- **AXO-36:** verified curriculum/paper/context hydration.
- **AXO-37:** Tavily eligibility, DLP and URL/source boundary.
- **AXO-106:** no raw student-derived Tutor audit persistence.
- **AXO-13:** Gemini 3.5 Flash-Lite certification/release gate.
- **AXO-12:** authentic marking-scheme RAG authority.
- **AXO-27:** legal/public disclosure reconciliation.
