# Submission timeout, 24 September 2026

Production inspection found two runs stranded in `structure`, with all 15
pages still `pending` and no structure model calls. Triage completed in seconds.

The structure worker selected `paper_page.margin_band`. This column does not
exist, and is not persisted by the upload contract. PostgreSQL reproduced
42703. The worker dropped the error, interpreted null data as a deleted page,
and returned success, acknowledging the queue message without doing the work.

The corrected select was validated against production with `LIMIT 0`. It uses
only persisted columns and a checked maybe-single read. A real missing row stays
distinct from an error. Schema drift is a configuration failure, never a claim
that the student's page is unreadable. Transient reads retry, and permanent
failure writes must succeed before acknowledgement. Terminal page retries also
check the advance RPC instead of discarding its errors.

The recovery Worker called a nonexistent public RPC, while the real
`private.sweep_stuck_runs()` had no database cron schedule. The operational SQL
in `db/operations/enable-stuck-run-recovery.sql` schedules the existing private
function directly. It retains stored pages and turns abandoned processing into
a visible retryable failure. It does not requeue or reprocess old submissions.

## Verification and release

- Six regression tests exercise the actual Supabase client against a simulated
  PostgREST schema/error response and the shared queue harness.
- Run `npm run typecheck` and `node --import tsx --test shared/src/__tests__/*.test.ts`.
- Bundle the structure and sweep Workers with `wrangler deploy --dry-run`.
- Deploy through the repository's existing CI after review and merge.
- The two old messages were acknowledged by the buggy worker. They cannot be
  recovered merely by deployment. After the recovery job marks their runs
  failed, resubmit the saved draft to create a new run, preserving uploaded pages.
- Verify a new one-page and multipage run reaches question review. Passing unit
  tests and a successful deploy do not constitute that production proof.
