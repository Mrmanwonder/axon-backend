# db/migrations

This is a **reference snapshot**, not a replayable migration history. Per
`AXON_FIX_BRIEF.md` §5.5: "Add a `db/migrations/` snapshot of the current
schema and the RPC definitions... They exist only in the live database."

## What's here

- **`functions.sql`** — the RPC and trigger-function definitions the
  pipeline and frontend depend on (`run_advance`, `run_heartbeat`,
  `advance_after_structure`, `advance_after_content`, `advance_after_explain`,
  `begin_explanations`, `commit_extraction_run`, `submit_paper`,
  `private.sweep_stuck_runs`, `private.run_lock`, `claim_deletions`,
  `finish_deletion`, `delete_my_account`), pulled verbatim via
  `pg_get_functiondef()` against the live project (`dlgcqieyevoebefhcggi`)
  on 2026-09-01. Read-only reference — reapplying these is a
  `CREATE OR REPLACE FUNCTION`, safe to run against the same schema, but
  this directory is not wired into any migration runner.
- **`2026*.sql`** — the migrations written *from this repository*, applied
  live and kept here in full. These are replayable, unlike the reference
  material above: every statement in them was applied exactly as written.
  The WP3/WP4 set (`20260901120000` through `20260901120300`) adds the
  `cropping` run status, `paper_page.crop_status`, `advance_after_crop`,
  `apply_region_crops`, the crop clauses in the sweep, and the grant
  correction that followed from reading the advisor rather than trusting
  the revoke.
- **`MANIFEST.md`** — the ordered list of the 26 migrations Supabase has
  actually recorded for this project (version + name only; Supabase's
  migration history table does not retain the SQL body, so the statements
  themselves aren't reproducible from here).

## What's deliberately not here

A full `schema.sql` (every `CREATE TABLE`, constraint, index, enum, view,
and RLS policy) is not included in this pass. Reconstructing exact `CHECK`
constraints, defaults, and enum values from introspection JSON is exactly
the kind of detail that's easy to get subtly wrong without a way to verify
it — unlike the worker code in `../../workers/`, there's no `tsc` or
`wrangler --dry-run` to catch a mistake here. The reliable way to get one:

```bash
supabase db dump --db-url "$SUPABASE_DB_URL" -f db/migrations/schema.sql
```

run against the live project by someone who can (safely, since it's a pure
read) hold the credentials — not attempted here.

## Do not hand-edit the live database to match this snapshot

This documents what's live, in the direction live → repo. Schema changes
from here forward should go through `supabase migration new` /
`apply_migration` and land as new files in this directory, same as any
other migration-managed project.
