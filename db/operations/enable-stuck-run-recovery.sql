-- Run as the database owner. No API grants or exposed wrappers are needed:
-- the maintenance function stays in private and pg_cron invokes it directly.
-- Re-running this uses the same named job rather than adding duplicates.
-- Every 5 minutes (AXO-274, 13 Oct 2026): the sweep only fails runs whose
-- heartbeat is older than 10 minutes, so checking every minute bought at most
-- four minutes and cost a pg_cron worker each time. Axon-Site migration
-- 20261013220000_axo_274_cron_cadence.sql altered the live job to match.
select cron.schedule(
  'axon-stuck-run-recovery',
  '*/5 * * * *',
  'select private.sweep_stuck_runs();'
);
-- Rollback of the schedule only:
-- select cron.unschedule('axon-stuck-run-recovery');
