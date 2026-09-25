-- Run as the database owner. No API grants or exposed wrappers are needed:
-- the maintenance function stays in private and pg_cron invokes it directly.
-- Re-running this uses the same named job rather than adding duplicates.
select cron.schedule(
  'axon-stuck-run-recovery',
  '* * * * *',
  'select private.sweep_stuck_runs();'
);
-- Rollback of the schedule only:
-- select cron.unschedule('axon-stuck-run-recovery');
