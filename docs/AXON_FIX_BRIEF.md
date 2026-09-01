# Axon — Implementation Brief

**For:** Claude Code
**Written:** 2026-08-31
**Author:** an investigation pass with live access to Cloudflare, Supabase, and both repos. Every claim below is either marked **verified** (read the live source, live config, or queried production this pass) or **carried** (established in an earlier audit, not re-checked here) or **unverified** (named so it does not get treated as fact).

---

## 0. Read this first

### What this document is

A complete issue ledger and an ordered set of work packages. Sections 1–3 are context. Section 4 is every known defect. Sections 5–9 are the work, in dependency order. Section 10 is what *not* to do.

### Ground rules

1. **Verify before you fix.** Several findings in older project docs are now stale — the pipeline was substantially repaired on 2026-08-31 and previous audits describe a system that no longer exists. Section 4 marks what is dead. If a claim here is marked *carried* or *unverified*, re-check it against live source before acting.
2. **Do not "fix" deliberate behaviour.** At least one thing that looks like a bug (`needs_review` set on every question) is an intentional product decision with a code comment explaining it. It is flagged. Changing it needs a product call, not a patch.
3. **The pipeline currently works end to end.** As of 2026-08-31 10:24 UTC a real 2-page paper ran triage → structure → content → adjudicate and produced marks for all 7 questions. Do not restart from scratch. Do not act on the older recommendation to delete the Cloudflare pipeline — see §4.F2.
4. **Ship in the order given.** The order is forced by dependencies, not preference. Cropping before the scanner produces illegible crops. Scanner work before the repo means another round of hand-patched bundles.
5. **Every work package ends with a live verification step.** Do not mark one done on a successful deploy. Deploys succeeding is not the same as the pipeline working — that exact mistake was made earlier today.

---

## 1. System map

### Repositories

| Repo | Contents | State |
|---|---|---|
| `github.com/Mrmanwonder/Axon-Site` | Frontend: Vite + React + TypeScript, plus a vanilla-JS scan engine in `src/scan/`. Also the never-deployed `supabase/functions/` pipeline. | Live, active. Last commit 2026-08-31. **232 files.** |
| `github.com/Mrmanwonder/axon-backend` | **NOT the workers.** A Flutter app (`lib/`, 324 Dart files) plus a Python/Firebase backend (`functions/`), IGCSE syllabus data, admissions/planner services. 105 commits, 82 branches, last touched 2026-06-19. Abandoned. | To be repurposed — see §5. |
| *(none)* | The `mastery-*` Cloudflare Workers exist **only as deployed bundles.** There is no source repository. | The problem §5 solves. |

### Cloudflare (account `957f007b821f8092608ed9dd1844e873`)

Nine workers: `mastery-triage`, `mastery-structure`, `mastery-content`, `mastery-reconcile`, `mastery-adjudicate`, `mastery-explain`, `mastery-sweep` (cron `*/15 * * * *`), `mastery-api`, `axon-site` (static asset shim, real frontend is in Workers Assets).

Six queues plus six DLQs. **All consumers are now `batch_size: 1`, `max_wait_time_ms: 1000`** — changed 2026-08-31, do not raise them, see §4.D3.

R2 buckets: `axon-originals`, `axon-derived`.
Zone `axonstudy.online` on the **Free** plan. Cloudflare Images is **not** subscribed (`allowed: 0`). Image Resizing is unavailable. This constrains §7.

### Supabase (project `dlgcqieyevoebefhcggi`, ap-northeast-2, PG 17.6)

Core tables: `paper`, `paper_page`, `extraction_run`, `question_region`, `teacher_mark`, `region_explanation`, `student_attempt`, `mark_loss_event`, `model_call`, `model_route`, `page_unreadable`.

Views: `attempt_analytics`, `mark_loss_analytics`, `student_analytics_readiness`, `consent_current`, `review_queue`, `paper_progress`.

Key RPCs: `run_advance`, `run_heartbeat`, `advance_after_structure`, `advance_after_content`, `advance_after_explain`, `begin_explanations`, `commit_extraction_run`, `private.sweep_stuck_runs`.

### The two data models — **critical to understand before touching anything**

There are two parallel representations of the same thing, and the frontend and the pipeline each use a different one:

| Concern | Pipeline writes | Frontend reads |
|---|---|---|
| A question on a paper | `question_region` | `student_attempt` |
| Why marks were lost | `region_explanation` | `mark_loss_event` |

The **only** bridge is `commit_extraction_run(p_run_id)`, which copies `question_region` → `student_attempt` and `region_explanation` → `mark_loss_event`. Until a run commits, **the frontend cannot see any pipeline output at all.**

Live counts (verified 2026-08-31): `question_region` 48, `student_attempt` 7, `region_explanation` **0**, `mark_loss_event` **0**, `paper` 35, `extraction_run` 60 (47 failed, 9 rejected, 3 needs_review, 1 committed).

### Model

All five model-calling stages use `gemini-3.1-flash-lite` via an OpenAI-compatible endpoint with `GOOGLE_API_KEY`. **Paid tier.** No fallbacks configured. `model_route.allow_training = true` (a deliberate prior decision by the user — see the notes column, do not silently change it).

---

## 2. How to verify things

- **Worker source:** `GET /accounts/{acct}/workers/scripts/{name}/content/v2` returns the deployed bundle as multipart. `r.result` is the string. Slice between the first `\r\n\r\n` and the last `\r\n--{boundary}--`. The bundles are ~750–800KB; **never** pull one into context whole — grep server-side and return small windows. The `// src/index.ts` marker separates bundled dependencies from the worker's own code, and that tail is only 4–8KB.
- **Deploying a worker without source:** `POST .../versions` with multipart (`metadata` + `index.js`), then `POST .../deployments` with `{strategy:"percentage", versions:[{version_id, percentage:100}]}`. The metadata **must** include `keep_bindings: ["secret_text","r2_bucket","queue"]` or every secret is wiped and cannot be restored (they are not readable). Cloudflare parse-checks the module at upload, so a syntax error fails the version POST rather than breaking production.
- **Worker logs:** `POST /accounts/{acct}/workers/observability/telemetry/query` with `filters: [{key:"$metadata.service", operation:"eq", value:"mastery-content", type:"string"}]`. This is the only way to see why a worker silently did nothing — it was decisive today.
- **After §5 ships, none of the above applies.** Use `wrangler deploy`.

---

## 3. What was fixed on 2026-08-31 — do not re-break

Recorded so these are not undone or re-diagnosed. All are live.

1. **Triage status guard.** `if (run.status !== "queued") return skipped` combined with advancing to `triaging` *before* the model call meant any retryable error stranded the run permanently. Now accepts `["queued","triaging"]`.
2. **Structure page-status stranding.** `structure_status` lives on `paper_page` (per paper, not per run), so a second run on the same paper skipped every page, created no regions, and hung forever. Triage now resets the paper's pages to `pending` before fan-out, and structure's "already done" path calls `advance_after_structure` instead of dead-ending.
3. **Queue batch size.** Content's consumer was `batch_size: 30`. A queue batch is one Worker invocation, and one invocation has a hard subrequest cap. Seven questions × ~12 subrequests killed the invocation with *"Too many subrequests by single Worker invocation"* — **before** the model call, which is why the stage looked like it hung doing nothing. All consumers are now `batch_size: 1`. **Do not raise these.** Papers with ≤3 questions used to squeak under the cap, which is why this looked intermittent.
4. **In-process transient retry.** `callModel` now retries once inline on 408/409/425/429/500/502/503/504 or a network error. Timeouts are deliberately excluded (they have already burned 90s). Verified live absorbing a 503.
5. **Handler deadline.** `HANDLE_TIMEOUT_MS` 100s → 200s to accommodate the inline retry.

---

## 4. The issue ledger

Severity: **P0** blocks the product working at all · **P1** materially breaks the experience · **P2** real but survivable · **P3** hygiene.

---

### A. Post-scan flow — why the app "does nothing"

This is the single most important section. The pipeline reads papers correctly and the student sees none of it.

#### A1 · [P0, verified] Explanations are kicked off before review and always fail, silently

`src/scan/ui.js` ~line 382:

```js
await openReview(result.runId);

explainPaper({
  runId: result.runId,
  regions: result.regions,
  onQuestion: () => scheduleReviewRefresh(),
}).catch(() => { /* a failed explanation leaves the marks intact and visible */ });
```

and `src/scan/pipeline.js` `explainPaper()` begins:

```js
await reviewComplete({ run_id: runId });
```

`mastery-api`'s `/review-complete` refuses with **409** while any region has `needs_review = true` and `student_confirmed_at IS NULL`:

```js
if ((count ?? 0) > 0) {
  return failure(`${count} question(s) still need your eyes.`, 409, { outstanding: count });
}
```

At the moment `explainPaper` fires, the student has confirmed nothing. So it **always** 409s, on every scan, and `.catch(() => {})` swallows it without a toast, a log, or a retry.

**Evidence:** `region_explanation` has **0 rows** across the app's entire history, against 48 question regions and 35 papers.

#### A2 · [P0, verified] Nothing ever calls `explainPaper` again

`save()` in `src/scan/ui.js` goes straight to `commitRun(S.runId)`. There is no second invocation of `explainPaper` after the student finishes confirming. So even a perfectly reviewed paper commits with zero explanations.

#### A3 · [P0, verified] Therefore commit always produces marks with no "why"

`commit_extraction_run` copies explanations into `mark_loss_event` with `INSERT … SELECT … FROM region_explanation`. With `region_explanation` empty, that select returns nothing, every time.

**Evidence:** `student_attempt` = 7, `mark_loss_event` = **0**.

The product promise is "know exactly what you're doing wrong, where, why, and how to improve." The *why* has never once been delivered. **A1 + A2 + A3 is the whole reason.**

#### A4 · [P1, verified] No question has ever reached `confident` — so review is 100% manual

0 of 48 regions are `confident`. 23 `unsure`, 25 `unreadable`. `assess()` in `mastery-reconcile`:

```js
const signals = {
  recognition: input.recognition === "high" || input.recognition === "medium",
  structural:  input.numberingSound,
  arithmetic:  input.paperReconciled,      // ← PAPER-level
  plausibility: plausible(input.awarded, input.available)
};
...
const allPass = Object.values(signals).every(Boolean);
if (allPass && !input.layerFallback) return { tier: "confident", signals };   // ← PAPER-level
return { tier: "unsure", signals };
```

Two of the four gates are **paper-scoped and apply to every question on the paper**:

- `arithmetic: input.paperReconciled` — if the paper's totals don't add up *anywhere*, every question becomes `unsure`, including ones read perfectly. `paper.reconciled` is `true` on **0 of 35** papers.
- `layerFallback` — set when marking isn't red or the student wrote in red. **11 of 80** pages carry `student_wrote_red`. One such page vetoes confidence for the entire paper.

The frontend has a bulk-accept path for cleanly-read questions (`cleanUnconfirmed` in `src/scan/review.js:109`, filtered to `tier === 'confident'`). Because nothing is ever confident, **that list is always empty**, and the student must tap through every question individually. On a 40-question paper that is 40 taps before anything happens.

> **⚠️ DO NOT "fix" this by making `needs_review` conditional.** `needs_review: true` is hardcoded in reconcile *deliberately*. `src/scan/review.js:97` explains why: *"review is mandatory in v1 and that is the whole point. Counting only the doubtful ones put 'Save to Library' on a button the server then refused, every time a paper had a cleanly-read question on it."* The defect is that **nothing is ever `confident`**, which breaks bulk-accept. Fix the tier computation, not the review requirement. Whether review stays mandatory is a product decision — §11.

#### A5 · [P0, verified] The post-scan screens do not exist

`src/ui/pages/PaperOverview.tsx` (16 lines), `PaperReview.tsx` (15), `QuestionDetail.tsx` (16) are all `RouteStub` placeholders. Their own comments say so — QuestionDetail's reads *"Milestone 1 in CLAUDE.md … and still unbuilt."*

`Library.tsx` links every row to `paths.paper(id)` → `/library/:paperId` → `PaperOverview` → **a stub**. So even the one committed paper cannot be opened.

The scan-time review sheet **is** built (`src/ui/scan/ReviewSheet.tsx`), so review works *during* a scan. There is no way back into it afterwards.

#### A6 · [P1, verified] The Library cannot show work in progress

`listPapers()` selects `paper_page(count), student_attempt(count)`. A paper with no committed attempts renders "Not read yet" forever. A paper that is mid-pipeline, or that failed, is indistinguishable from one never scanned. There is no status, no thumbnail (`Thumb()` renders decorative placeholder lines), and no retry.

---

### B. Scanner

The scan engine is **far more complete than it looks from the database.** `src/scan/` is 4,288 lines across 20 modules including `camera.js`, `capture.js`, `edges.js`, `quad.js`, `geometry.js`, `conditioning.js`, `layers.js`, `colour.js`, `crops.js`, `quality.js`. Edge detection, quad fitting and homography **exist**. The problem is that their output is not reaching storage.

#### B1 · [P1, verified] `source_kind` is hardcoded, so capture provenance is lost

`src/scan/pipeline.js:178` sets `source_kind: 'upload'` for every page regardless of how it was captured. All 80 pages in production say `upload`. `conditioning_meta.capture_path` and `.live_gate` are **null on every page**. It is currently impossible to tell from data whether the camera path is being used at all.

#### B2 · [P1, verified] `warped: false` on every page

The homography code exists but `conditioning_meta.warped` is `false` on all 80 pages. Either the warp is never applied, or it is applied and the flag is never set. **Determine which before changing anything** — these are different bugs.

#### B3 · [P0 for cropping, verified] The original is thrown away

`paper_page.original_key` is **null on all 80 pages**. Only the conditioned, downscaled derivative is kept. Nothing can be re-derived: a bad crop, a wrong box, or a conditioning regression is unrecoverable without asking the student to rescan. **Cropping is not safe to build until this changes.**

#### B4 · [P1, verified] `thumb_key` is declared and never written

Null on all 80 pages. This is why triage ships full-resolution pages to answer "is there marking on this" — see C3.

#### B5 · [P1, verified] Output resolution is inconsistent and sometimes far too low

`conditioning_meta.source_size` shows long edges of **1000px** on some pages and **2400px** on others. A question cropped out of a 1000px page is illegible. Cropping requires a guaranteed floor.

#### B6 · [P1, carried — re-verify] Quality-gate thresholds are miscalibrated

From `src/scan/contract.js` (per the 2026-08-26 audit; confirm against current source):

```
BLUR_WARN 0.22   BLUR_FAIL 0.10
GLARE_WARN 0.005 GLARE_FAIL 0.035     ← 7× apart, undocumented
RESOLUTION_WARN 1800  RESOLUTION_FAIL 1000
```

`RESOLUTION_FAIL` is reportedly defined but never wired into `scorePage()`'s verdict. Live verdict distribution: **14 fail, 6 warn, 4 ok, 56 null.** Most pages are never scored at all. Pages with `glare: 0.94` and `clipping: 0.91` were accepted as `warn` and pushed into the pipeline.

#### B7 · [P2, carried — re-verify] Live gate and final scoring use different scales

The live gate scores a ~240px proxy using the same absolute constants as the final 2400px image. Sharpness and glare metrics are scale-dependent; comparing them across a 10× scale difference is not meaningful.

#### B8 · [P3, carried — re-verify] Dead scan code

`skewDegrees()` / `SKEW_WARN_DEG` implemented, exported, never called. `anisotropy()` computed on every page and, by its own comment, "decided nothing on its own."

#### B9 · [P1, verified] Triage sends up to six full-resolution pages

`PAGES_TO_LOOK_AT = 6` in `mastery-triage`. See C3 for the measured cost.

---

### C. Cropping and payload size

#### C1 · [P0 for speed, verified] `crop_key` is read by two stages and written by nothing

`mastery-content` and `mastery-adjudicate` both branch on `region.crop_key`. **0 of 48** regions have one. Content therefore always takes the fallback path and sends the **entire page** for every question. A 7-question paper sends 7 full pages to answer 7 narrow questions.

#### C2 · [unverified] `src/scan/crops.js` exists (73 lines)

Client-side. Probably generates crops for the review UI from local blobs, not for the pipeline. **Read it before designing §7** — it may already contain reusable box→crop geometry.

#### C3 · [P1, verified] Latency is dominated by image count, superlinearly

Measured across all successful calls in the last 2 days:

| Stage | Images | Calls | Avg | Avg input tokens |
|---|---|---|---|---|
| adjudicate | 1 | 5 | 3.7s | 1,538 |
| triage | 1 | 9 | 9.4s | 1,364 |
| structure | 1 | 9 | 10.2s | 1,499 |
| content | 1 | 41 | 11.4s | 1,690 |
| structure | **2** | 6 | **24.0s** | 2,599 |
| triage | **2** | 6 | **27.2s** | 2,466 |

Doubling images roughly **triples** latency while tokens rise only ~1.8×. This is per-image transfer and decode, not billed compute. Pages average 257KB and peak at 1MB, inlined as base64 (+33%), so two pages is up to a **2.8MB POST body per call**.

**Paying for a better model cannot fix this.** Smaller and fewer images can.

#### C4 · [P2, verified] The `detail` hint is accepted and discarded

`imageRef(env, bucket, key, detail)` takes `"low"` / `"high"` and returns it, but `callModel` builds content as:

```js
content.push({ type: "image_url", image_url: { url: image.url } });
```

`detail` never reaches the request. Triage explicitly asks for `"low"` — with a code comment explaining that a thumbnail settles the question — and silently sends full resolution. It was stripped during the Gemini compatibility work; re-adding it is **not** safe without testing, since that is likely why it was removed.

---

### D. Pipeline — remaining defects

#### D1 · [P1, verified] Reconcile will hit the subrequest ceiling on a long paper

`mastery-reconcile` updates every question region in a single invocation (`Promise.all` over all regions). At the current maximum of 7 questions per run this is fine. A paper with ~35+ questions hits the **same** subrequest limit that took down the content stage. Real exam papers routinely have more than 35 parts.

#### D2 · [P2, verified] `mastery-reconcile` has no `SELF_QUEUE` binding

Its bindings are `queue:ADJUDICATE_QUEUE` plus secrets — no `SELF_QUEUE`. The manual re-enqueue retry path therefore always falls through to native `message.retry()`. Not fatal (the fallback works) but inconsistent with the other five workers and undocumented.

#### D3 · [P2, verified] The sweep only half-recovers a stranded paper

`private.sweep_stuck_runs()` resets `paper_page.structure_status` only where it equals `'running'`. Pages left at `'done'` by a failed run stay `'done'`. Before the §3.2 fix this poisoned the paper permanently. The triage-side reset now covers it, but the sweep is still not self-sufficient — worth aligning.

#### D4 · [P1, verified] No source of truth for worker code

Three different indentation variants of the same shared `callModel` block exist across the six workers, from repeated hand-editing of deployed bundles. Three of the four bugs fixed on 2026-08-31 are the kind a type-check or review catches instantly. §5 exists to end this.

---

### E. Security, compliance, hygiene

#### E1 · [P0 legal, carried] Guardian verification is a stub in production

`src/config.js`: `export const VERIFICATION_ADAPTER = 'stub';`

`src/verification.js`'s own comment on that adapter: *"It proves nothing about a real person and must never be enabled in production."* It is the shipped, active adapter. It waits 600ms and returns success unconditionally.

DPDP Rule 10 requires verifying a parent's identity, adulthood, and relationship to the child before processing a minor's data. **This is live legal exposure, not technical debt.** It is a product/legal decision (§11), not something to patch — but it must not ship to real families.

#### E2 · [P1, verified today] `public.Subscribers` is a foreign table exposed over the REST API

Foreign tables **do not respect RLS**. Anything in it is readable by anyone who can reach the API. Confirmed still present 2026-08-31.

#### E3 · [P1, verified today] `delete_my_account()` is `SECURITY DEFINER` and callable by any authenticated user

Via `/rest/v1/rpc/delete_my_account`. Almost certainly intentional — erasure has to bypass RLS. **Confirm with an actual test, not a reading, that it can only ever act on the caller's own account.** A `SECURITY DEFINER` erasure function that can be pointed at another ID is a one-line deletion of another family's data.

#### E4 · [P2, verified today] Leaked-password protection is disabled

Supabase Auth's HaveIBeenPwned check is off. One setting.

#### E5 · [P3, verified today] Five tables have RLS enabled with zero policies

`eval_result`, `eval_run`, `model_call`, `model_route`, `r2_deletion`. This means "nobody gets anything," which is presumably intended for service-role-only tables — but that is an assumption, not a documented one. Add an explicit comment per table.

#### E6 · [P2, carried] ~15 foreign keys without covering indexes

Across `extraction_run`, `paper_page`, `question_region`, `teacher_mark`, `region_explanation`, `student_attempt`, `upload`, `consent_event`, `mark_loss_event`, `page_unreadable`. Cheap to add and correctness-adjacent. **Re-run `get_advisors(type:"performance")` for the current list rather than trusting this one.**

#### E7 · [P3, verified] Production data is contaminated with test data

35 papers, 60 extraction runs (47 failed), 80 pages — nearly all of it from debugging. There is no `is_seed` marker. Any accuracy measurement taken today is meaningless.

#### E8 · [P2, carried] No CI

The 2026-08-26 audit found no `.github/` directory in Axon-Site. `npm test` runs against example fixtures only. Verify and fix as part of §5.

---

### F. Stale findings — do NOT act on these

#### F1 · Fixed: model IDs pointed at nonexistent models

Older docs state `model_route` points at `google/gemma-4-31b-it:free` and that 10/10 triage calls failed with `no_compliant_provider`. **Superseded.** All stages now run paid `gemini-3.1-flash-lite` and the pipeline completes.

#### F2 · Superseded: "delete the Cloudflare pipeline, deploy the Supabase one"

`app-audit-2026-08-26.md` §1 recommends keeping `supabase/functions/` and deleting the `mastery-*` workers, on the grounds that the Cloudflare pipeline was broken and the Supabase one better designed. **The premise no longer holds.** As of 2026-08-31 the Cloudflare pipeline runs a real paper end to end. The Supabase pipeline has never been deployed and has never processed a submission. Deleting a working system in favour of an untested one is the wrong trade. **Treat `supabase/functions/` as a reference implementation to mine for ideas, not as the target.** This remains a product decision (§11) but the engineering recommendation has reversed.

#### F3 · Superseded: "the provider is degraded"

`pipeline-repair-2026-08-31.md` claims the model provider was badly degraded and recommends a paid fallback. That was an overclaim from one 503. The full log shows, on the paid Gemini key, exactly **six** failures ever: four HTTP 400s in one 26-minute window this morning *before* the request-body compatibility fixes landed (none since), one 503, one timeout. Every other error in `model_call` belongs to a previous provider — Groq's 413s, OpenCode Zen's 429/401, OpenRouter's `no_compliant_provider`. **The endpoint is fine. Do not add a fallback. The slowness is payload size (C3).**

---

## 5. WP1 — Repository (do this first)

**Why first:** everything after this touches worker code, and there is currently nowhere to put it.

### 5.1 Preserve what is there

`axon-backend` holds a Flutter app and a Python backend. The user has approved resetting it, **on condition the old work is preserved first.**

```bash
git clone https://github.com/Mrmanwonder/axon-backend.git
cd axon-backend
git tag -a legacy-flutter -m "Flutter app + Python/Firebase backend, archived 2026-08-31"
git push origin legacy-flutter
git ls-remote --heads origin > /tmp/legacy-branches.txt   # keep the branch list
```

**Verify the tag is on the remote before deleting anything.** `git ls-remote --tags origin | grep legacy-flutter` must return a hash.

### 5.2 Structure

```
axon-backend/
  package.json                 # workspaces
  tsconfig.base.json
  shared/
    src/
      model.ts                 # callModel, classify, ModelError, getRoute, logCall
      worker.ts                # consumeQueue, withDeadline, isRetryable, failRun
      r2.ts                    # imageRef, signAssetUrl, base64url
      supabase.ts              # serviceClient
      prompts/
        triage.v1.ts  structure.v1.ts  content.v1.ts
        adjudicate.v1.ts  explain.v1.ts  untrusted.ts
      attribution.ts  reconcile.ts  confidence.ts  schemas.ts
  workers/
    triage/ structure/ content/ reconcile/ adjudicate/ explain/ sweep/ api/
      src/index.ts
      wrangler.toml
  db/migrations/
  .github/workflows/deploy.yml
```

**The point is that `shared/` is genuinely shared.** Today there are three divergent copies of the same `callModel` block across the workers. One import, one copy, forever.

### 5.3 Seeding it from the live bundles

There is no source. Reconstruct it:

1. For each worker, fetch `content/v2` and extract the tail from `// src/index.ts` — that is the worker's own code, 4–8KB, readable and clean.
2. The bundled dependencies above that marker are esbuild output from `shared/` — reconstruct `shared/` from **one** worker's copy (use `mastery-triage`, which has the current `callModel` with the inline retry), not by merging the variants.
3. Convert JS back to TypeScript with real types. Do not skip this — the type-checker is the whole point.
4. `wrangler.toml` per worker: bindings, queue producers/consumers, `compatibility_date = "2026-08-01"`, `compatibility_flags = ["nodejs_compat"]`, and the consumer settings from §1 (**`max_batch_size = 1`**).
5. **Secrets are not in the repo.** They already exist on the deployed workers. `wrangler deploy` preserves existing secrets — but verify per worker after the first deploy with `GET .../settings` that `secret_text:GOOGLE_API_KEY`, `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`, `ASSET_SIGNING_SECRET`, `MASTERY_ASSET_URL` are all still bound.

### 5.4 Acceptance

- [ ] `legacy-flutter` tag on the remote, verified by `git ls-remote --tags`.
- [ ] `npm run typecheck` passes across all workspaces.
- [ ] Each worker builds with `wrangler deploy --dry-run`.
- [ ] A no-op deploy of `mastery-triage` from the repo produces a bundle whose `// src/index.ts` tail is **semantically identical** to what is live. Diff it before deploying for real.
- [ ] Bindings verified intact on all six after the first real deploy.
- [ ] CI runs typecheck + tests on PR.
- [ ] `docs/` carries this brief and a README explaining that these workers were once hand-edited in production and must never be again.

### 5.5 While you are here

Add a `db/migrations/` snapshot of the current schema and the RPC definitions (`run_advance`, `advance_after_*`, `begin_explanations`, `commit_extraction_run`, `private.sweep_stuck_runs`). They exist only in the live database.

---

## 6. WP2 — Post-scan flow (highest value, smallest change)

**Why now:** this is a handful of lines and it is the difference between a pipeline that produces nothing a student can see and one that delivers the product.

### 6.1 Fix the explain ordering (A1, A2, A3) — **the critical fix**

The current call graph is wrong: `explainPaper()` both *starts* explanations and *waits* for them, and it is invoked before review, where its first statement is guaranteed to fail.

**Split it in two:**

```js
// pipeline.js — NEW: start explanations. Call this only after review is complete.
export async function startExplanations(runId) {
  return reviewComplete({ run_id: runId });   // throws 409 if anything is unconfirmed
}

// pipeline.js — CHANGED: watch only. No longer calls reviewComplete.
export async function watchExplanations({ runId, regions, onQuestion }) { /* existing poll loop */ }
```

In `src/scan/ui.js`:

- **Remove** the `explainPaper({...}).catch(() => {})` block after `openReview()`. It cannot succeed and it hides its own failure.
- In `save()`, after the outstanding-count guard passes and **before** `commitRun`:

```js
async function save() {
  if (!S.runId) return;
  if (S.review?.outstanding) {
    toast(`${S.review.outstanding} question(s) still need a look. They are at the top.`);
    return;
  }
  try {
    await startExplanations(S.runId);
    watchExplanations({ runId: S.runId, regions, onQuestion: () => scheduleReviewRefresh() })
      .catch((e) => { console.error('explanations', e); toast('Explanations are still coming.', 'warn'); });
    // ... then commit — see 6.2 for the ordering constraint
  } catch (error) { toast(error.message || 'That could not be saved.', 'warn'); }
}
```

**Never swallow an error from this path again.** Log it and surface it. A silent `.catch(() => {})` is what hid this for the app's entire history.

### 6.2 Resolve the commit/explanation ordering conflict

`commit_extraction_run` copies `region_explanation` → `mark_loss_event` **at commit time**. Explanations are generated asynchronously and can take minutes. So committing immediately after starting them guarantees zero `mark_loss_event` rows — the current bug, just moved.

Pick one, deliberately:

**Option A (recommended):** commit only once every region's `explain_status` is settled (`done` / `skipped` / `failed`). `run_advance` already moves the run to `ready` when nothing is pending — commit on `ready`. The paper is in the Library with marks and explanations together. Costs a wait after "Save".

**Option B:** commit immediately, and have `mastery-explain` write directly into `mark_loss_event` for already-committed runs (it can find the attempt via `question_region.committed_attempt_id`, which commit already sets). The paper appears instantly and explanations fill in. More moving parts, better UX.

**Do not leave it as-is.** Whichever is chosen, add a regression test asserting `mark_loss_event` is non-empty for a committed run that had marks lost.

### 6.3 Fix the confidence tiers (A4)

In `mastery-reconcile`'s `assess()`, demote the two paper-level vetoes:

- **`arithmetic`** — a paper that does not reconcile has a discrepancy in a *specific place*. `mastery-adjudicate` already exists to identify suspect regions. Apply the arithmetic penalty only to regions adjudicate flags; leave the rest able to reach `confident`. Do not punish 7 questions for one bad total.
- **`layerFallback`** — scope it to the pages it actually affects (via `question_region.page_spans`), and let it lower `recognition` rather than veto the paper.

**Acceptance:** on a clean paper, a majority of questions reach `confident`, so `cleanUnconfirmed` is non-empty and the bulk-accept button appears. Keep `needs_review = true` — see the warning in A4.

### 6.4 Build the post-scan screens (A5)

- `QuestionDetail.tsx` — the marks, the teacher's remark, the crop, and the explanation (`body`, `do_this_next`, `cause`, `concepts` from `mark_loss_event`). `CLAUDE.md` calls this milestone 1; it sets the design language, so give it a real pass.
- `PaperOverview.tsx` — the question list with marks, total, reconciliation state, and a link into each question.
- `PaperReview.tsx` — reachable re-entry into review after the scan session ends. `src/ui/scan/ReviewSheet.tsx` already exists; reuse it rather than building a second review UI.

### 6.5 Make the Library honest (A6)

- Show every paper from upload onward with a live status: **Scanning → Reading → Needs your eyes → Ready → Saved**, derived from `extraction_run.status`. `paper_progress` and `review_queue` views already exist for this.
- A failed paper stays visible, says it failed, and offers a retry. It must never silently vanish.
- Replace `student_attempt(count)` as the sole signal — a paper mid-pipeline should not read "Not read yet".
- Use `thumb_key` for the row thumbnail once §8 writes it.

### 6.6 Acceptance for WP2

- [ ] Scan a real paper end to end. `region_explanation` has rows. `mark_loss_event` has rows.
- [ ] A majority of questions on a clean paper are `confident`; bulk-accept appears.
- [ ] Tapping a Library row opens a real paper screen; tapping a question opens a real question screen with an explanation.
- [ ] A paper appears in the Library within seconds of upload, with a status.
- [ ] Kill the network mid-run: the paper shows as failed with a retry, not as missing.

---

## 7. WP3 — Scanner

**Prerequisite:** WP1. Read `src/scan/*.js` in full first — the engine is more complete than the database suggests, and this is mostly wiring, not building.

### 7.1 First, diagnose — do not assume

Answer these with evidence before writing code:

1. Is the camera path ever taken? Add real telemetry to `conditioning_meta.capture_path` (B1) — `'camera'` / `'gallery'` / `'pdf'` — and stop hardcoding `source_kind`.
2. Is the homography applied and the flag not set, or not applied at all (B2)? Different bugs, different fixes.
3. Why is `long_edge` 1000 on some pages and 2400 on others (B5)?
4. Why do 56 of 80 pages have no `quality_verdict` (B6)?

### 7.2 Then wire what exists

- **`source_kind` + `capture_path`** — record the truth.
- **`warped`** — make the flag mean something, and make the warp actually run.
- **`original_key`** — store the full-resolution, warped, unconditioned image in `axon-originals`. **Blocking prerequisite for WP4.** Respect the retention design in `STORAGE_R2.md`.
- **`thumb_key`** — write a 512px thumbnail. Cheap, and it unblocks the largest single latency win (§7.5).
- **Resolution floor** — guarantee ≥2400px long edge on the conditioned page, or refuse the capture and say why.
- **`live_gate`** — populate it. The field exists and is null everywhere.

### 7.3 The live gate

Auto-capture only when all hold for ~5 consecutive frames: quad locked and ≥60% of frame; corners stable; sharpness above threshold measured on the page interior; glare below threshold; clipping below threshold; projected long edge after warp ≥2400px. Manual shutter always available.

**The principle: every rejection the pipeline currently makes two minutes late, the camera should make instantly.**

Fix B7 as part of this — either compute live metrics at a scale comparable to the final image, or calibrate two separate threshold sets and document why they differ.

### 7.4 Recalibrate the quality gate (B6)

`GLARE_WARN 0.005` vs `GLARE_FAIL 0.035` is a 7× gap nobody can justify from the source. Derive thresholds from real captures, wire `RESOLUTION_FAIL` into `scorePage()`, and **document each threshold with the evidence behind it.** 14 of 80 pages currently fail; verify that is real and not a false-reject, since a false reject on a good scan is the worst possible failure for this product.

### 7.5 Fix triage's payload (B9, C3)

Point triage at `thumb_key`. Classification — "is this a marked exam paper" — does not need full resolution; the code comment already says so. **Expected: triage from ~27–43s to a few seconds.** This is the single cheapest large speed win in the whole system and it depends only on 7.2's thumbnail.

Consider reducing `PAGES_TO_LOOK_AT` from 6, but note it feeds `marked_page_count` — do not change it without checking that consumer.

### 7.6 What "better than Adobe Scan" means here

Do not try to out-engineer Adobe at general document scanning. Adobe's cleanup pushes toward crisp bilevel text, which **destroys the red/blue ink separation this entire app depends on**. Win on the job Adobe is not doing:

1. **Preserve the marking.** Illumination flattening, never binarization. Keep the CIELAB ink separation in `colour.js` / `layers.js`.
2. **Reject at the camera, not after the upload.** The live gate can say "this is not a marked exam page" while the student is still holding the phone.
3. **Guarantee legibility of the smallest meaningful mark** — a circled 3 in a margin — not of body text.
4. **Never discard the original**, so mistakes are recoverable server-side.
5. **Measure and store everything**, so a refusal can always explain itself. That is the app's stated promise.

### 7.7 Acceptance

- [ ] `source_kind` / `capture_path` reflect reality; `live_gate` populated.
- [ ] `warped: true` on camera captures, and the warp is visibly correct.
- [ ] `original_key` and `thumb_key` non-null on every new page.
- [ ] Every new conditioned page ≥2400px long edge.
- [ ] Every new page has a `quality_verdict`.
- [ ] Triage latency drops to single-digit seconds on a 2-page paper. Measure from `model_call.latency_ms`.
- [ ] Ten real captures in varied lighting: no false rejects on good pages, correct rejects on bad ones.

---

## 8. WP4 — Cropping

**Prerequisites:** WP3 (§7.2's `original_key` and the resolution floor). Cropping a 1000px page is worthless — do not start early.

### 8.1 Constraint

Cloudflare Images is **not** subscribed and the zone is on the Free plan, so URL-based transforms and `cf.image` are unavailable. **Decision taken: build it in WASM inside the Worker.** Do not add a paid image service.

### 8.2 Design

A crop step between structure and content:

1. Structure already returns per-question boxes and writes `question_region.page_spans`.
2. After structure finishes a page, decode that page **once** — WASM (`@jsquash/webp` or `photon`). Prefer the original from `axon-originals` over the conditioned derivative.
3. Cut all N question crops with generous padding (a mark in the margin must survive the crop — pad wider than the box, especially horizontally).
4. Encode each as WebP, write to `axon-derived`, set `question_region.crop_key`. Do the same for the ink mask → `cropmask_key`.
5. Content and adjudicate already prefer `crop_key` when present. **No change needed in those workers** — they take the good path automatically once the column is populated.

**Watch the subrequest budget.** One decode plus N R2 writes plus N DB updates in one invocation. With `batch_size: 1` you have room, but a 40-question page could still approach the cap — batch the DB updates into a single call and consider chunking R2 writes. This is exactly the failure that took down the content stage; do not repeat it.

### 8.3 Read `src/scan/crops.js` first (C2)

73 lines, client-side. It may already contain the box→crop geometry, coordinate normalisation, and padding logic worth reusing. Do not reimplement blindly.

### 8.4 Leave `detail` alone (C4)

The `detail` hint is stripped from the request body, probably deliberately during the Gemini compatibility work. Re-adding it is a separate, tested experiment. **Crops make it irrelevant** — do not bundle the two.

### 8.5 Acceptance

- [ ] Every region on a new run has a non-null `crop_key`.
- [ ] Crops are visually correct: the question, its number, its marks, and the teacher's margin annotation all inside the frame. **Inspect at least 20 by eye.** A crop that clips the mark is worse than no crop.
- [ ] Content-stage latency drops measurably (`model_call.latency_ms`, 1-image calls).
- [ ] Content-stage accuracy does not regress — compare extracted marks against the same papers pre-crop.
- [ ] No subrequest-limit errors in the observability logs for the crop step.

---

## 9. WP5 — Remaining pipeline defects

### 9.1 Reconcile's subrequest ceiling (D1)

Replace the per-region `Promise.all` of individual updates with **one** call. Add a Postgres function taking a JSON array:

```sql
create or replace function public.apply_region_confidence(p_rows jsonb)
returns integer language plpgsql security definer
set search_path to 'public', 'pg_temp' as $$
declare v_count integer;
begin
  update public.question_region r
     set confidence_tier   = (e->>'tier')::public.confidence_tier,
         confidence_signals = coalesce(r.confidence_signals,'{}'::jsonb) || (e->'signals'),
         needs_review      = (e->>'needs_review')::boolean,
         updated_at        = now()
    from jsonb_array_elements(p_rows) e
   where r.id = (e->>'id')::uuid;
  get diagnostics v_count = row_count;
  return v_count;
end; $$;
```

One subrequest regardless of question count. **Test with a synthetic 60-question run** — do not wait for a real long paper to find the ceiling.

### 9.2 Add `SELF_QUEUE` to reconcile (D2)

For consistency with the other five workers. Declare it in `wrangler.toml` as part of WP1.

### 9.3 Make the sweep self-sufficient (D3)

`private.sweep_stuck_runs()` should also reset `paper_page.structure_status` from `'done'` to `'pending'` for pages of a paper whose run it just failed — otherwise recovery depends entirely on the triage-side reset.

### 9.4 Security and hygiene

In rough priority:

1. **`public.Subscribers`** (E2) — remove from the exposed schema or lock it down. Foreign tables ignore RLS.
2. **`delete_my_account()`** (E3) — write a test proving it can only act on the caller. This is a data-loss risk, not a lint.
3. **Leaked-password protection** (E4) — enable it.
4. **FK indexes** (E6) — re-run `get_advisors(type:"performance")` and add the missing ones. Leave the unused-index list alone; they are unused because nothing has run through the pipeline, not because they are wrong.
5. **RLS-no-policy tables** (E5) — add an explicit comment on each stating service-role-only is intended.
6. **Test data** (E7) — decide: `is_seed` flag or a clean reset. Do it before measuring accuracy on anything.
7. **CI** (E8) — typecheck, tests, and `wrangler deploy --dry-run` on every PR.

---

## 10. Do not do these

- **Do not raise any queue `batch_size` above 1** without recalculating the subrequest budget per message. This caused the "stuck at finding answers and marking" bug.
- **Do not deploy a worker without `keep_bindings`.** Secrets are unreadable and unrecoverable.
- **Do not make `needs_review` conditional** to make review shorter. It is deliberate (A4). Fix the confidence tier instead.
- **Do not delete the Cloudflare pipeline** in favour of `supabase/functions/`. That recommendation is stale (F2).
- **Do not add a model fallback or change providers.** The endpoint is fine (F3). The latency is payload size.
- **Do not re-add the `detail` hint** as part of the cropping work (C4).
- **Do not swallow errors.** `.catch(() => {})` on the explanation path hid a total product failure for the app's entire history. Every catch logs and surfaces.
- **Do not reset `axon-backend` before the `legacy-flutter` tag is confirmed on the remote.**
- **Do not enable `VERIFICATION_ADAPTER` beyond `'stub'` in production** without the legal decision in §11.
- **Do not measure accuracy against the current database.** It is contaminated with debugging data (E7).

---

## 11. Product decisions — for the user, not the implementer

These block or redirect real work. Do not guess.

1. **CBSE or Cambridge?** `package.json` says CBSE. `src/curriculum.js` and the syllabus-code system are Cambridge-only. The one real test paper is CBSE. `axon-backend` carries IGCSE syllabus data. The canonical question bank, past-paper matching, and syllabus matching all depend on the answer. Asked in two prior audits, still unanswered.
2. **Is review mandatory for every question in v1?** Currently yes, deliberately (A4). Once confidence tiers work, bulk-accept makes this bearable — but "confirm all 40 questions before you see anything" may still be the wrong product. §6.3 assumes mandatory review stays.
3. **Guardian verification (E1).** Is DigiLocker in scope now, or is a hand-verified soft-launch cohort acceptable? The stub cannot ship to real families either way.
4. **Commit before or after explanations (§6.2)?** Instant Library entry with explanations filling in, versus a wait and a complete paper.
5. **Does the undeployed `supabase/functions/` pipeline stay in the repo?** It is a well-designed reference implementation and it is also a permanent source of confusion about which pipeline is real. Archive it, or delete it and keep the tag.

---

## 12. Suggested order

| Order | Package | Size | Unblocks |
|---|---|---|---|
| 1 | §5 Repo | days | everything else |
| 2 | §6 Post-scan flow | days | the product actually delivering its promise |
| 3 | §9.1 Reconcile ceiling | hours | long papers |
| 4 | §7 Scanner | weeks | cropping, capture quality, triage speed |
| 5 | §8 Cropping | days | content speed and accuracy |
| 6 | §9.4 Security & hygiene | days | launch readiness |

§6 is worth doing before §7 even though the scanner is the louder complaint: it is far smaller, and until it lands, no scanner improvement changes anything a student can see.
