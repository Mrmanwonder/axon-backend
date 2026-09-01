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

The last three are from this branch (AXON_FIX_BRIEF.md §9.1 and §9.3):
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
