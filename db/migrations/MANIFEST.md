# Migration history (as recorded by Supabase, 2026-09-01)

Project `dlgcqieyevoebefhcggi`. Version, then name. Newest last.

| Version | Name |
|---|---|
| 20260810130016 | enable_rls_with_owner_policies |
| 20260810173605 | drop_legacy_planner_schema |
| 20260810173906 | identity_and_consent |
| 20260810174051 | academic_model |
| 20260810174147 | rls_and_analytics |
| 20260810174248 | storage_papers |
| 20260810174749 | harden_helpers_into_private_schema |
| 20260810180011 | preferences_and_erasure |
| 20260810180229 | paper_pages_ingestion |
| 20260810234336 | fix_erasure_student_tombstone |
| 20260811164138 | student_avatar_seed |
| 20260823192635 | extraction_pipeline |
| 20260823192657 | pipeline_health |
| 20260823192703 | expand_board_enum |
| 20260823192713 | runtime_status_values |
| 20260823192833 | r2_and_runtime |
| 20260823192950 | pipeline_runtime |
| 20260824063536 | 20260811120000_board_caie |
| 20260824063543 | 20260811120100_caie_defaults_and_syllabus |
| 20260824063718 | 20260811120200_canonical_question_cambridge_schema |
| 20260825043237 | cloudflare_queue_fanout |
| 20260825045847 | submit_paper_accept_existing_draft |
| 20260826073407 | phase3_2_security_hardening |
| 20260826073419 | phase3_3_missing_fk_indexes |
| 20260826074533 | public_schema_default_grants |
| 20260827075741 | model_call_error_detail |
| 20260901045632 | apply_region_confidence_rpc |
| 20260901045730 | sweep_resets_done_pages_too |
| 20260901045753 | sweep_resets_done_pages_too_fix |
| 20260901050006 | revoke_subscribers_public_access |
| 20260901050315 | document_service_role_only_tables |
| 20260901050352 | restrict_apply_region_confidence_to_service_role |
| 20260901050418 | restrict_apply_region_confidence_to_service_role_v2 |

The last four are §9.4 security fixes (E2, E5) from this same branch — see
`docs/SECURITY.md`. `restrict_apply_region_confidence_to_service_role` (no
suffix) turned out to be a no-op — it revoked from `anon`/`authenticated`
directly, which doesn't touch the `PUBLIC`-pseudo-role grant Postgres
creates by default and that those roles inherit through; `_v2` is the real
fix (`REVOKE ... FROM PUBLIC`). Both are kept in this history rather than
silently replaced, since it's exactly the mistake `docs/SECURITY.md`
documents finding.

The three before those are also from this branch (AXON_FIX_BRIEF.md §9.1 and §9.3):
`apply_region_confidence_rpc` adds the single-statement confidence update
`workers/reconcile/src/index.ts` now calls instead of one UPDATE per region;
the two `sweep_resets_done_pages_too*` migrations replace
`private.sweep_stuck_runs()` so it also resets a page stuck at
`structure_status = 'done'` under a run it just failed, not only `'running'`
(the `_fix` migration corrects the first attempt, which had also changed the
existing `'running'` -> `'failed'` behavior instead of leaving it alone —
`functions.sql` reflects only the corrected, final version).

Note the three entries whose *names* carry an earlier date
(`20260811120000_board_caie` etc., applied 20260824) — they were written
earlier and applied later. Not investigated further here; flagging it so a
future migration-tooling change doesn't get confused by the mismatch.


## Added 2026-09-01 (WP3 / WP4)

Four migrations, applied live in this order:

| version | name | what it does |
| --- | --- | --- |
| `20260901120000` | `crop_stage_enum_and_status` | `cropping` in `extraction_status`, `paper_page.crop_status` |
| `20260901120100` | `crop_stage_functions` | `advance_after_structure` hands off to cropping; `advance_after_crop`; `apply_region_crops` |
| `20260901120200` | `sweep_covers_crop` | the sweep closes out a crop left running under a run it fails |
| `20260901120300` | `crop_functions_revoke_direct_grants` | the grants correction below |

The last one is worth reading before writing any new `SECURITY DEFINER`
function against this project, because it is the *second* time the same
question has been got wrong from the opposite direction.

On 2026-08-31, `apply_region_confidence` was fixed by revoking from
`PUBLIC` after revoking from `anon` and `authenticated` turned out to be a
no-op — Postgres grants `EXECUTE` to `PUBLIC` on creation and both roles
inherit through it.

On 2026-09-01, revoking from `PUBLIC` alone turned out to be insufficient:
this project also runs `ALTER DEFAULT PRIVILEGES` on the `public` schema
(Axon-Site's `20260826074500_public_schema_default_grants.sql`), which
grants `EXECUTE` to `anon` and `authenticated` **directly** on every new
function. A new function therefore carries both kinds of grant at once.

**Revoke from all three, and then read `pg_proc.proacl` to check.** A
correct-looking `REVOKE` proves nothing; the ACL is the only thing that
answers the question. Both crop functions now read
`{postgres=X/postgres,service_role=X/postgres}`.

The `page_source` enum gained `'camera'` and `'pdf'` in the same session
(`20260901090000`, applied from the Axon-Site repo, where the rest of the
`page_source` history lives).
