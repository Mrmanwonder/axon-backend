# Security & hygiene pass — 2026-09-01

Findings and actions against `AXON_FIX_BRIEF.md` §4.E / §9.4, re-verified
live against the project (`dlgcqieyevoebefhcggi`) rather than trusted from
the brief, per its own rule 1 ("re-check against live source before
acting"). One item (E1) is explicitly a product/legal decision and not
touched here — see §11 of the brief.

## E2 — `public."Subscribers"` foreign table exposure — **fixed, was live**

Confirmed the worst finding in the whole brief: `anon` (fully
unauthenticated — the role PostgREST uses for a request with no bearer
token) held `SELECT`, `INSERT`, `UPDATE` and `DELETE` on this table, as did
`authenticated`. It is a Stripe-customers FDW table (`id`, `email`, `name`,
`description`, `created`, `attrs`) — real customer emails and names,
readable, writable and deletable by anyone who could reach the REST API,
logged in or not. Foreign tables cannot have RLS, so the only fix is the
grants themselves.

**Action:** `revoke all on table public."Subscribers" from anon;` and the
same for `authenticated`. `service_role` keeps its access. Verified with
`has_function_privilege`-equivalent role-grant checks before and after, and
a `SET LOCAL ROLE anon; SELECT ...` probe that now raises
`insufficient_privilege`.

## E3 — `delete_my_account()` scoping — **verified with an actual test**

The brief asked for a test, not a reading. `db/migrations/tests/delete_my_account.test.sql`
creates two synthetic guardian/student pairs inside a transaction, simulates
guardian A's own session (`SET LOCAL request.jwt.claim.sub`, the same GUC
`auth.uid()` reads from a real bearer token), calls the RPC, and asserts:
guardian A's student is erased, guardian A's `auth.users` row is gone,
guardian B's student/guardian rows are completely untouched. Then rolls
back — nothing here persists. **Passed** on 2026-09-01. The function is
correctly scoped: it only ever resolves the caller's own guardian via
`auth.uid()`, never a caller-supplied ID.

## E4 — Leaked-password protection — **not done; no tool to do it**

Confirmed still disabled via `get_advisors(type:"security")`
(`auth_leaked_password_protection`, WARN). This is a Supabase Auth
dashboard/Management-API setting (Authentication → Policies → Password
Security in the dashboard), not something reachable from SQL or any tool
available in this pass. Flagging rather than fabricating a fix: toggle it
at <https://supabase.com/dashboard/project/dlgcqieyevoebefhcggi/auth/providers>
or via the Management API's `PATCH /v1/projects/{ref}/config/auth`.

## E5 — RLS-enabled-no-policy tables — **documented, behavior unchanged**

`eval_result`, `eval_run`, `model_call`, `model_route`, `r2_deletion` —
confirmed live, same five as the brief. RLS with zero policies means
"nobody but `service_role`," which was already the case and is correct for
these (internal eval-harness output, raw model-call logs, routing config,
the R2 GC queue — none have a per-guardian/student ownership concept RLS
could even express). Added an explicit `COMMENT ON TABLE` to each stating
that intent, so the next person to see "RLS enabled, no policies" in an
advisor report doesn't have to re-derive whether it's a bug.

## E6 — Missing FK indexes — **already fixed, nothing to do**

The brief listed ~15 FKs across nine tables. Re-ran
`get_advisors(type:"performance")` per its own explicit instruction rather
than trusting that list, and it returned exactly one `unindexed_foreign_keys`
result — `stripe._managed_webhooks.fk_managed_webhooks_account`, in the
Stripe FDW's own managed schema, not application code and not something to
touch. The `phase3_3_missing_fk_indexes` migration (2026-08-26, already
live before this pass) evidently closed the real ones. Confirms the brief's
own warning that this list goes stale.

## E7 — Test-data contamination — **not touched, needs a human call**

35 papers / 60 extraction runs / 80 pages, almost all debugging output, no
`is_seed` marker — confirmed still the case. Not acted on in this pass:
telling which of the current rows are "real" test-account activity worth
keeping versus disposable debug output isn't something derivable from the
data alone, and marking the wrong ones would be worse than leaving it
undecided. Needs a person who knows which papers are real to either add an
`is_seed` flag and backfill it, or do a clean reset — either way, per the
brief, before trusting any accuracy measurement taken from this project.

## E8 — CI — done in #82 (WP1)

`.github/workflows/deploy.yml`: typecheck + test + per-worker dry-run on
every PR.
