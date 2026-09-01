-- WP4 (AXON_FIX_BRIEF.md §8): a crop step between structure and content.
--
-- Content and adjudicate already prefer `question_region.crop_key` when it is
-- set and fall back to the whole page when it is not. Nothing has ever set it.
-- This is the schema half of the stage that will: a place for the run to be
-- while it happens, a per-page status so a partial failure is visible rather
-- than a stall, and one batched write so a forty-question page does not spend
-- forty round trips setting forty keys.

-- ── the run has somewhere to be ────────────────────────────────────────────
-- Sorted between 'structure' (2) and 'content' (3). The run genuinely is doing
-- something else during this, and reporting it as still "finding the questions"
-- would be telling the student something untrue about their own paper.
alter type public.extraction_status add value if not exists 'cropping' after 'structure';

-- ── per-page progress ──────────────────────────────────────────────────────
-- Mirrors `structure_status`, deliberately: the advance gate counts pages that
-- are still pending or running, so a page that fails is *counted as finished*
-- and the run moves on. A crop failure must never hold up a paper — it costs
-- the crops, and content silently takes the full-page path it takes today.
--
--   pending | running | done | failed | skipped
--
-- 'skipped' is its own value and not a synonym for 'done': a page with no
-- regions to cut, or one whose page image never arrived, has nothing to report
-- and should not read as a successful crop in the data.
alter table public.paper_page
  add column if not exists crop_status text not null default 'pending';

comment on column public.paper_page.crop_status is
  'pending | running | done | failed | skipped. A crop failure never blocks the run: content falls back to the full page.';
