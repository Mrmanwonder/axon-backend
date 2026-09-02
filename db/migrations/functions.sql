-- Reference snapshot of RPC/trigger functions the pipeline and frontend
-- depend on, pulled verbatim via pg_get_functiondef() against the live
-- project (dlgcqieyevoebefhcggi) on 2026-09-01, and kept in step with it as
-- this repo's own migrations land. See README.md — this is documentation,
-- not a migration this repo's tooling runs.

-- ============================================================
-- private.run_lock — per-run advisory lock used by run_advance and the
-- advance_after_* functions so two concurrent workers finishing at once
-- can't race each other's status transition.
-- ============================================================
CREATE OR REPLACE FUNCTION private.run_lock(p_run_id uuid)
 RETURNS void
 LANGUAGE sql
 SET search_path TO 'public', 'pg_temp'
AS $function$
  select pg_advisory_xact_lock(hashtextextended(p_run_id::text, 0));
$function$;

-- ============================================================
-- public.run_advance — the single place extraction_run.status changes.
-- Terminal states (committed/failed/rejected) refuse to be overwritten, so
-- a late worker finishing after a sweep already failed the run can't
-- resurrect it into a state nothing will advance further.
-- ============================================================
CREATE OR REPLACE FUNCTION public.run_advance(p_run_id uuid, p_to extraction_status, p_reason text DEFAULT NULL::text)
 RETURNS extraction_status
 LANGUAGE plpgsql
 SECURITY DEFINER
 SET search_path TO 'public', 'pg_temp'
AS $function$
declare v_from public.extraction_status;
begin
  perform private.run_lock(p_run_id);
  select status into v_from from public.extraction_run where id = p_run_id for update;
  if v_from is null then
    raise exception 'no such extraction run' using errcode = 'P0002';
  end if;

  -- Terminal is terminal. A late worker finishing after a sweep already failed
  -- the run must not resurrect it into a state nothing will ever advance.
  if v_from in ('committed', 'failed', 'rejected') then
    return v_from;
  end if;

  update public.extraction_run
     set status        = p_to,
         status_reason = coalesce(p_reason, case when p_to in ('failed','rejected') then status_reason end),
         heartbeat_at  = now(),
         finished_at   = case when p_to in ('committed','failed','rejected') then now() else finished_at end
   where id = p_run_id;

  return p_to;
end; $function$;

-- ============================================================
-- public.run_heartbeat — called at the start of every queue handler
-- invocation (see shared/worker.ts's consumeQueue). Feeds
-- private.sweep_stuck_runs' staleness check.
-- ============================================================
CREATE OR REPLACE FUNCTION public.run_heartbeat(p_run_id uuid)
 RETURNS void
 LANGUAGE sql
 SECURITY DEFINER
 SET search_path TO 'public', 'pg_temp'
AS $function$
  update public.extraction_run set heartbeat_at = now() where id = p_run_id;
$function$;

-- ============================================================
-- public.advance_after_structure — called after every page's structure
-- pass; only actually advances the run once every page on the paper is
-- past 'pending'/'running'.
--
-- Rewritten 2026-09-01 for the crop stage (AXON_FIX_BRIEF.md §8): it used
-- to advance straight to 'content' and return the region ids to fan out to
-- CONTENT_QUEUE. It now advances to 'cropping' and returns the *page* ids
-- for CROP_QUEUE; advance_after_crop is what returns the region ids once
-- the crops have settled. `enqueue_reconcile` keeps its old meaning
-- exactly — a paper with no regions at all has nothing to crop and nothing
-- to read, and goes straight to reconciliation rather than sitting in a
-- stage with no work in it.
-- ============================================================
CREATE OR REPLACE FUNCTION public.advance_after_structure(p_run_id uuid)
 RETURNS jsonb
 LANGUAGE plpgsql
 SECURITY DEFINER
 SET search_path TO 'public', 'pg_temp'
AS $function$
declare
  v_paper   uuid;
  v_pending integer;
  v_regions integer;
  v_pages   uuid[];
begin
  perform private.run_lock(p_run_id);
  select paper_id into v_paper from public.extraction_run where id = p_run_id;
  if v_paper is null then return jsonb_build_object('advanced', false); end if;

  select count(*) into v_pending from public.paper_page
   where paper_id = v_paper and structure_status in ('pending', 'running');
  if v_pending > 0 then return jsonb_build_object('advanced', false); end if;

  if (select status from public.extraction_run where id = p_run_id) <> 'structure' then
    return jsonb_build_object('advanced', false);
  end if;

  select count(*) into v_regions
    from public.question_region
   where run_id = p_run_id and extract_status = 'pending';

  if v_regions = 0 then
    perform public.run_advance(p_run_id, 'content');
    return jsonb_build_object('advanced', true, 'enqueue_crop', '[]'::jsonb, 'enqueue_reconcile', true);
  end if;

  perform public.run_advance(p_run_id, 'cropping');

  select array_agg(distinct pp.id) into v_pages
    from public.paper_page pp
    join public.question_region qr on qr.paper_id = pp.paper_id
    join lateral jsonb_array_elements(qr.page_spans) span on true
   where qr.run_id = p_run_id
     and pp.paper_id = v_paper
     and pp.r2_key is not null
     and (span ->> 'page')::int = pp.page_number;

  return jsonb_build_object(
    'advanced', true,
    'enqueue_crop', coalesce(to_jsonb(v_pages), '[]'::jsonb),
    'enqueue_reconcile', false);
end; $function$;

-- ============================================================
-- public.advance_after_crop — the gate between cropping and content.
--
-- Counts only pages still 'pending' or 'running', so a page whose crop
-- FAILED is counted as finished and the run moves on. That one omission is
-- the whole "content waits for crops but is never blocked by a crop
-- failure" requirement (§8.2): a page that could not be cropped leaves its
-- regions' crop_key null, and content falls back to the full page exactly
-- as it did before this stage existed.
-- ============================================================
CREATE OR REPLACE FUNCTION public.advance_after_crop(p_run_id uuid)
 RETURNS jsonb
 LANGUAGE plpgsql
 SECURITY DEFINER
 SET search_path TO 'public', 'pg_temp'
AS $function$
declare
  v_paper   uuid;
  v_pending integer;
  v_regions uuid[];
begin
  perform private.run_lock(p_run_id);
  select paper_id into v_paper from public.extraction_run where id = p_run_id;
  if v_paper is null then return jsonb_build_object('advanced', false); end if;

  select count(*) into v_pending from public.paper_page
   where paper_id = v_paper and crop_status in ('pending', 'running');
  if v_pending > 0 then return jsonb_build_object('advanced', false); end if;

  if (select status from public.extraction_run where id = p_run_id) <> 'cropping' then
    return jsonb_build_object('advanced', false);
  end if;

  perform public.run_advance(p_run_id, 'content');

  select array_agg(id order by order_index) into v_regions
    from public.question_region
   where run_id = p_run_id and extract_status = 'pending';

  return jsonb_build_object(
    'advanced', true,
    'enqueue_content', coalesce(to_jsonb(v_regions), '[]'::jsonb),
    'enqueue_reconcile', coalesce(array_length(v_regions, 1), 0) = 0);
end; $function$;

-- ============================================================
-- public.apply_region_crops — one statement for a page's worth of crop
-- keys. §8.2 names the subrequest budget as the constraint and it is the
-- failure that took the content stage down once; a forty-question page is
-- one call rather than forty. Scoped by run_id as well as region id, and
-- writes only the two key columns — nothing here can touch a mark, a
-- confidence tier, or a student's correction.
--
-- EXECUTE is revoked from PUBLIC *and* from anon and authenticated by
-- name. See MANIFEST.md: this project's default privileges grant directly
-- to those two roles as well as through PUBLIC, so revoking either alone
-- leaves the other standing.
-- ============================================================
CREATE OR REPLACE FUNCTION public.apply_region_crops(p_run_id uuid, p_rows jsonb)
 RETURNS integer
 LANGUAGE plpgsql
 SECURITY DEFINER
 SET search_path TO 'public', 'pg_temp'
AS $function$
declare v_count integer;
begin
  if jsonb_typeof(p_rows) <> 'array' then
    raise exception 'apply_region_crops expects an array' using errcode = '22023';
  end if;

  update public.question_region qr
     set crop_key     = row_in.crop_key,
         cropmask_key = row_in.cropmask_key,
         updated_at   = now()
    from (
      select (value ->> 'id')::uuid       as id,
             value ->> 'crop_key'         as crop_key,
             value ->> 'cropmask_key'     as cropmask_key
        from jsonb_array_elements(p_rows)
    ) row_in
   where qr.id = row_in.id
     and qr.run_id = p_run_id;

  get diagnostics v_count = row_count;
  return v_count;
end; $function$;

-- ============================================================
-- public.advance_after_content — mirrors advance_after_structure for the
-- content pass; advances to 'attribution' once every region on the run is
-- past 'pending'/'running'.
-- ============================================================
CREATE OR REPLACE FUNCTION public.advance_after_content(p_run_id uuid)
 RETURNS jsonb
 LANGUAGE plpgsql
 SECURITY DEFINER
 SET search_path TO 'public', 'pg_temp'
AS $function$
declare v_pending integer;
begin
  perform private.run_lock(p_run_id);
  select count(*) into v_pending from public.question_region
   where run_id = p_run_id and extract_status in ('pending', 'running');
  if v_pending > 0 then return jsonb_build_object('advanced', false); end if;

  if (select status from public.extraction_run where id = p_run_id) <> 'content' then
    return jsonb_build_object('advanced', false);
  end if;

  perform public.run_advance(p_run_id, 'attribution');
  return jsonb_build_object('advanced', true, 'enqueue_reconcile', true);
end; $function$;

-- ============================================================
-- public.advance_after_explain — 'failed' counts as finished here
-- deliberately: a question that couldn't be explained is one visible gap
-- on one card, not a reason to strand the other nineteen good explanations
-- behind it.
-- ============================================================
CREATE OR REPLACE FUNCTION public.advance_after_explain(p_run_id uuid)
 RETURNS boolean
 LANGUAGE plpgsql
 SECURITY DEFINER
 SET search_path TO 'public', 'pg_temp'
AS $function$
declare v_pending integer;
begin
  perform private.run_lock(p_run_id);
  -- 'failed' counts as finished. A question we could not explain is a visible
  -- gap on one card; nineteen good explanations waiting on it would be a
  -- stalled paper, which is the worse failure and the invisible one.
  select count(*) into v_pending from public.question_region
   where run_id = p_run_id and explain_status in ('pending', 'queued', 'running');
  if v_pending > 0 then return false; end if;

  if (select status from public.extraction_run where id = p_run_id) <> 'explaining' then
    return false;
  end if;
  perform public.run_advance(p_run_id, 'ready');
  return true;
end; $function$;

-- ============================================================
-- public.begin_explanations — the ONLY place explanations start (called
-- from mastery-api's /review-complete, see workers/api/src/index.ts). Gates
-- on every review-required region being confirmed; raises 42501 otherwise,
-- which is what /review-complete's 409 response is built on.
-- ============================================================
CREATE OR REPLACE FUNCTION public.begin_explanations(p_run_id uuid)
 RETURNS jsonb
 LANGUAGE plpgsql
 SECURITY DEFINER
 SET search_path TO 'public', 'pg_temp'
AS $function$
declare v_pending integer; v_regions uuid[];
begin
  perform private.run_lock(p_run_id);

  select count(*) into v_pending from public.question_region
   where run_id = p_run_id and needs_review and student_confirmed_at is null;
  if v_pending > 0 then
    raise exception '% question(s) still need review before explanations can start', v_pending
      using errcode = '42501';
  end if;

  if (select status from public.extraction_run where id = p_run_id) not in ('needs_review', 'explaining') then
    return jsonb_build_object('queued', 0, 'region_ids', '[]'::jsonb);
  end if;
  perform public.run_advance(p_run_id, 'explaining');

  update public.question_region r
     set explain_status = 'skipped'
   where r.run_id = p_run_id
     and r.explain_status = 'pending'
     and (r.confidence_tier = 'unreadable'
          or r.marks_awarded is null or r.marks_available is null
          or r.marks_awarded >= r.marks_available);

  select array_agg(id) into v_regions from (
    select r.id from public.question_region r
     where r.run_id = p_run_id and r.explain_status = 'pending'
     order by (r.marks_available - r.marks_awarded) desc, r.order_index
  ) ordered;

  update public.question_region
     set explain_status = 'queued'
   where id = any(coalesce(v_regions, '{}'::uuid[]));

  if not exists (select 1 from public.question_region
                  where run_id = p_run_id and explain_status in ('pending', 'queued', 'running')) then
    perform public.run_advance(p_run_id, 'ready');
  end if;

  return jsonb_build_object('queued', coalesce(array_length(v_regions, 1), 0),
                             'region_ids', coalesce(to_jsonb(v_regions), '[]'::jsonb));
end; $function$;

-- ============================================================
-- public.commit_extraction_run — the only bridge between the pipeline's
-- tables (question_region, region_explanation) and the frontend's
-- (student_attempt, mark_loss_event). See AXON_FIX_BRIEF.md §1's "two data
-- models" section — until this runs, the frontend cannot see any pipeline
-- output at all.
-- ============================================================
CREATE OR REPLACE FUNCTION public.commit_extraction_run(p_run_id uuid)
 RETURNS jsonb
 LANGUAGE plpgsql
 SET search_path TO 'public', 'pg_temp'
AS $function$
declare
  v_run       public.extraction_run;
  v_paper     public.paper;
  v_pending   integer;
  v_region    public.question_region;
  v_attempt   uuid;
  v_committed integer := 0;
begin
  select * into v_run from public.extraction_run where id = p_run_id;
  if v_run.id is null then
    raise exception 'no such extraction run' using errcode = 'P0002';
  end if;
  if v_run.committed_at is not null then
    raise exception 'this run is already committed' using errcode = '23505';
  end if;

  select * into v_paper from public.paper where id = v_run.paper_id;

  -- Review is a required step. A region that still needs the student's eyes has
  -- not had them, and committing it would put an unconfirmed reading into the
  -- record where it starts shaping insights.
  select count(*) into v_pending
  from public.question_region r
  where r.run_id = p_run_id
    and r.needs_review
    and r.student_confirmed_at is null;

  if v_pending > 0 then
    raise exception '% question(s) still need review before this paper can be saved', v_pending
      using errcode = '42501';
  end if;

  for v_region in
    select * from public.question_region
    where run_id = p_run_id
    order by order_index
  loop
    -- Hard rule 4: a region nobody could read is not quietly dropped, and it is
    -- not turned into an attempt with a guessed mark either. It stays visible as
    -- a region and contributes nothing.
    if v_region.confidence_tier = 'unreadable' then
      continue;
    end if;
    -- No mark on the page means no fact to record. The question is real and the
    -- region keeps it; there is simply nothing to aggregate.
    if v_region.marks_awarded is null or v_region.marks_available is null then
      continue;
    end if;

    insert into public.student_attempt (
      student_id, paper_id, paper_tier, canonical_question_id,
      question_label, question_text,
      student_answer, marks_awarded, max_marks, marks_source, teacher_remark,
      extraction_confidence,
      student_confirmed_at
    ) values (
      v_region.student_id, v_region.paper_id, v_paper.tier,
      -- Only a Tier 2 paper may carry one. On a Tier 1 fallback this is null
      -- whatever the region holds, so a stale match cannot survive the
      -- downgrade and trip the constraint instead of being dropped.
      case when v_paper.tier = 'tier_2' then v_region.canonical_question_id end,
      coalesce(v_region.question_label, 'Q' || (v_region.order_index + 1)),
      v_region.question_text, v_region.student_answer,
      v_region.marks_awarded, v_region.marks_available,
      -- Hard rule 1. The number was read off the teacher's pen; the model has no
      -- standing to be its source and there is no enum value that would let it be.
      'teacher_pen',
      v_region.teacher_remark,
      -- The spec's three tiers collapse onto the database's three-value enum
      -- here. `confident` is `likely` — read cleanly, not yet confirmed by the
      -- person who sat the exam — and everything else is `unsure`, which the
      -- analytics views exclude until it is confirmed.
      case when v_region.confidence_tier = 'confident' then 'likely'::public.confidence
           else 'unsure'::public.confidence end,
      v_region.student_confirmed_at
    )
    returning id into v_attempt;

    update public.question_region
       set committed_attempt_id = v_attempt, updated_at = now()
     where id = v_region.id;

    -- Stage 8's output becomes a loss event now that there is an attempt for it
    -- to hang off. Only where the model actually had something to say: a
    -- question it could not explain leaves no row, which is the honest outcome
    -- and keeps the empty slot empty rather than filling it with a shrug.
    insert into public.mark_loss_event (
      attempt_id, student_id, cause, marks_lost, ai_explanation, do_this_next, confidence
    )
    select v_attempt, e.student_id, e.cause, e.marks_lost, e.body, e.do_this_next,
           case when v_region.confidence_tier = 'confident' then 'likely'::public.confidence
                else 'unsure'::public.confidence end
      from public.region_explanation e
     where e.region_id = v_region.id
       and e.cause is not null
       and e.marks_lost is not null
       and e.marks_lost <= v_region.marks_available - v_region.marks_awarded;

    v_committed := v_committed + 1;
  end loop;

  update public.extraction_run
     set status = 'committed', committed_at = now(), finished_at = coalesce(finished_at, now())
   where id = p_run_id;

  return jsonb_build_object(
    'run_id', p_run_id,
    'attempts_committed', v_committed,
    'reconciled', v_run.reconciled,
    'reconcile_delta', v_run.reconcile_delta
  );
end;
$function$;

-- ============================================================
-- public.submit_paper — called from mastery-api's /paper-submit. Idempotent
-- on p_idempotency_key; upserts pages by (paper_id, page_number); reuses an
-- existing non-terminal extraction_run for the paper rather than always
-- creating a new one.
-- ============================================================
CREATE OR REPLACE FUNCTION public.submit_paper(p_student_id uuid, p_type paper_type, p_tier paper_tier, p_date_taken date, p_subject text, p_pages jsonb, p_idempotency_key uuid, p_reported_total numeric DEFAULT NULL::numeric, p_stated_maximum numeric DEFAULT NULL::numeric, p_pipeline_version text DEFAULT '1.0.0'::text, p_paper_id uuid DEFAULT NULL::uuid)
 RETURNS jsonb
 LANGUAGE plpgsql
 SET search_path TO 'public', 'pg_temp'
AS $function$
declare
  v_paper   public.paper;
  v_run_id  uuid;
  v_page    jsonb;
  v_created boolean := false;
begin
  if jsonb_typeof(p_pages) <> 'array' or jsonb_array_length(p_pages) = 0 then
    raise exception 'a paper needs at least one page' using errcode = '22023';
  end if;

  if p_paper_id is not null then
    select * into v_paper from public.paper where id = p_paper_id and student_id = p_student_id;
  end if;

  if v_paper.id is null and p_paper_id is not null then
    raise exception 'that paper does not exist or is not yours' using errcode = '42501';
  end if;

  if v_paper.id is not null then
    update public.paper set
      type = p_type, tier = p_tier, date_taken = coalesce(p_date_taken, date_taken),
      subject = p_subject,
      reported_total = coalesce(p_reported_total, reported_total),
      stated_maximum = coalesce(p_stated_maximum, stated_maximum)
    where id = v_paper.id
    returning * into v_paper;
  else
    if p_idempotency_key is null then
      raise exception 'an idempotency key is required' using errcode = '22004';
    end if;

    select * into v_paper from public.paper where idempotency_key = p_idempotency_key;

    if v_paper.id is null then
      insert into public.paper (student_id, type, tier, date_taken, subject,
                                reported_total, stated_maximum, idempotency_key)
      values (p_student_id, p_type, p_tier, coalesce(p_date_taken, current_date), p_subject,
              p_reported_total, p_stated_maximum, p_idempotency_key)
      returning * into v_paper;
      v_created := true;
    end if;
  end if;

  for v_page in select * from jsonb_array_elements(p_pages) loop
    insert into public.paper_page (
      paper_id, student_id, page_number, source_kind, status,
      r2_bucket, r2_key, mask_key, original_key, thumb_key,
      bytes, sha256, etag, preprocess_version,
      quality_verdict, quality_signals, conditioning_meta, layer_fallback,
      teacher_marks, teacher_mark_count)
    values (
      v_paper.id, p_student_id,
      (v_page ->> 'page_number')::smallint,
      coalesce((v_page ->> 'source_kind')::public.page_source, 'upload'),
      'stored',
      coalesce(v_page ->> 'r2_bucket', 'derived'),
      v_page ->> 'r2_key',
      v_page ->> 'mask_key',
      v_page ->> 'original_key',
      v_page ->> 'thumb_key',
      (v_page ->> 'bytes')::integer,
      v_page ->> 'sha256',
      v_page ->> 'etag',
      coalesce(v_page ->> 'preprocess_version', 'v2'),
      v_page ->> 'quality_verdict',
      coalesce(v_page -> 'quality_signals', '{}'::jsonb),
      coalesce(v_page -> 'conditioning_meta', '{}'::jsonb),
      v_page ->> 'layer_fallback',
      coalesce(v_page -> 'teacher_marks', '[]'::jsonb),
      coalesce(jsonb_array_length(v_page -> 'teacher_marks'), 0))
    on conflict (paper_id, page_number) do update set
      r2_bucket = excluded.r2_bucket, r2_key = excluded.r2_key, mask_key = excluded.mask_key,
      original_key = excluded.original_key, thumb_key = excluded.thumb_key,
      bytes = excluded.bytes, sha256 = excluded.sha256, etag = excluded.etag,
      quality_verdict = excluded.quality_verdict, quality_signals = excluded.quality_signals,
      conditioning_meta = excluded.conditioning_meta, layer_fallback = excluded.layer_fallback,
      teacher_marks = excluded.teacher_marks, teacher_mark_count = excluded.teacher_mark_count;
  end loop;

  select id into v_run_id from public.extraction_run
   where paper_id = v_paper.id and status not in ('failed', 'rejected')
   order by started_at desc limit 1;

  if v_run_id is null then
    insert into public.extraction_run (paper_id, student_id, pipeline_version,
                                       preprocess_version, status, heartbeat_at)
    values (v_paper.id, p_student_id, p_pipeline_version,
            coalesce(p_pages -> 0 ->> 'preprocess_version', 'v2'), 'queued', now())
    returning id into v_run_id;
  end if;

  return jsonb_build_object(
    'paper_id', v_paper.id, 'run_id', v_run_id, 'created', v_created,
    'pages', (select count(*) from public.paper_page where paper_id = v_paper.id));
end; $function$;

-- ============================================================
-- private.sweep_stuck_runs — mastery-sweep's cron (*/15 * * * *) calls this
-- first. Fails a run stale past p_stale (default 10 minutes) with no
-- heartbeat, and closes out any question_region/paper_page left 'running'
-- under it. Updated on this branch per AXON_FIX_BRIEF.md §4.D3/§9.3: also
-- resets a page stuck at structure_status = 'done' back to 'pending' — a
-- completed-then-stalled page used to only get reset by a *new* run's
-- triage-side pass (§3.2), which never fired if no new run was ever
-- started. The sweep no longer depends on that.
--
-- Updated again 2026-09-01 for the crop stage: crop_status gets the same
-- two clauses. A crop left 'running' under a swept run is failed (no
-- worker is coming back for it), and one that finished is reset so a fresh
-- run re-cuts it — the page image may itself be why the run stalled, and
-- reusing last run's crops would carry that forward.
-- ============================================================
CREATE OR REPLACE FUNCTION private.sweep_stuck_runs(p_stale interval DEFAULT '00:10:00'::interval)
 RETURNS integer
 LANGUAGE plpgsql
 SECURITY DEFINER
 SET search_path TO 'public', 'pg_temp'
AS $function$
declare
  v_swept integer;
  v_run_ids uuid[];
begin
  select array_agg(id) into v_run_ids
    from public.extraction_run
   where status not in ('queued', 'needs_review', 'ready', 'committed', 'failed', 'rejected')
     and coalesce(heartbeat_at, started_at) < now() - p_stale;

  if v_run_ids is null then
    return 0;
  end if;

  update public.extraction_run
     set status        = 'failed',
         status_reason = 'This paper stopped partway through. Your pages are kept — you can try again.',
         finished_at   = now()
   where id = any(v_run_ids);
  get diagnostics v_swept = row_count;

  -- Close out anything left mid-flight under a run we just failed, so a
  -- swept paper never leaves individual questions/pages frozen at
  -- "running"/"pending" forever (the orphaned-region bug found 2026-08-31).
  update public.question_region
     set extract_status = case when extract_status = 'running' then 'failed' else extract_status end,
         explain_status = case when explain_status = 'running' then 'failed' else explain_status end,
         confidence_tier = case when extract_status = 'running' then 'unreadable' else confidence_tier end,
         needs_review = needs_review or extract_status = 'running',
         confidence_signals = case when extract_status = 'running'
           then coalesce(confidence_signals, '{}'::jsonb) || '{"unreadable_reason":"We could not finish checking this question. It has been flagged for review."}'::jsonb
           else confidence_signals end,
         updated_at = now()
   where run_id = any(v_run_ids)
     and (extract_status = 'running' or explain_status = 'running');

  update public.paper_page
     set structure_status = 'failed'
   where paper_id in (select paper_id from public.extraction_run where id = any(v_run_ids))
     and structure_status = 'running';

  -- AXON_FIX_BRIEF.md §4.D3 / §9.3.
  update public.paper_page
     set structure_status = 'pending'
   where paper_id in (select paper_id from public.extraction_run where id = any(v_run_ids))
     and structure_status = 'done';

  update public.paper_page
     set crop_status = 'failed'
   where paper_id in (select paper_id from public.extraction_run where id = any(v_run_ids))
     and crop_status = 'running';

  update public.paper_page
     set crop_status = 'pending'
   where paper_id in (select paper_id from public.extraction_run where id = any(v_run_ids))
     and crop_status in ('done', 'skipped');

  return v_swept;
end; $function$;

-- ============================================================
-- public.apply_region_confidence — mastery-reconcile's confidence-tier
-- update, added on this branch per AXON_FIX_BRIEF.md §9.1. One statement
-- for every region on a run instead of one UPDATE per region inside a
-- Promise.all — the same shape of subrequest-ceiling bug that took down
-- mastery-content before batch_size was capped at 1 (§3.3), which is fine
-- at today's <=7-question papers and would not be at ~35+. Verified against
-- a synthetic 60-row batch before this shipped.
--
-- SECURITY DEFINER, service_role only — EXECUTE is explicitly revoked from
-- PUBLIC (Postgres grants it there by default on CREATE FUNCTION, which
-- anon/authenticated then inherit; revoking from anon/authenticated
-- directly does NOT remove that, a mistake this function's own first
-- migration made and get_advisors(type:"security") caught within minutes —
-- see the two REVOKE statements below, applied as a follow-up on this same
-- branch). Worker-only: called solely from
-- workers/reconcile/src/index.ts's service-role client, never by a
-- browser/authenticated session.
-- ============================================================
CREATE OR REPLACE FUNCTION public.apply_region_confidence(p_rows jsonb)
 RETURNS integer
 LANGUAGE plpgsql
 SECURITY DEFINER
 SET search_path TO 'public', 'pg_temp'
AS $$
declare v_count integer;
begin
  update public.question_region r
     set confidence_tier    = (e->>'tier')::public.confidence_tier,
         confidence_signals = coalesce(r.confidence_signals, '{}'::jsonb) || (e->'signals'),
         needs_review       = (e->>'needs_review')::boolean,
         updated_at         = now()
    from jsonb_array_elements(p_rows) e
   where r.id = (e->>'id')::uuid;
  get diagnostics v_count = row_count;
  return v_count;
end;
$$;

revoke execute on function public.apply_region_confidence(jsonb) from public;
grant execute on function public.apply_region_confidence(jsonb) to service_role;

-- ============================================================
-- public.claim_deletions / public.finish_deletion — the R2 garbage
-- collector mastery-sweep drives every tick. claim_deletions locks a batch
-- of pending r2_deletion rows (FOR UPDATE SKIP LOCKED, so concurrent sweep
-- runs can't double-claim); finish_deletion marks one done, or records an
-- error and leaves it for the next tick to retry.
-- ============================================================
CREATE OR REPLACE FUNCTION public.claim_deletions(p_limit integer DEFAULT 5)
 RETURNS TABLE(id bigint, bucket text, prefix text, key text, attempts integer)
 LANGUAGE plpgsql
 SECURITY DEFINER
 SET search_path TO 'public', 'pg_temp'
AS $function$
begin
  return query
  update public.r2_deletion d
     set attempts = d.attempts + 1
   where d.id in (
     select c.id from public.r2_deletion c
      where c.done_at is null
        and c.attempts < 20
      order by c.created_at
      for update skip locked
      limit p_limit)
  returning d.id, d.bucket, d.prefix, d.key, d.attempts;
end; $function$;

CREATE OR REPLACE FUNCTION public.finish_deletion(p_id bigint, p_error text DEFAULT NULL::text)
 RETURNS void
 LANGUAGE sql
 SECURITY DEFINER
 SET search_path TO 'public', 'pg_temp'
AS $function$
  update public.r2_deletion
     set done_at = case when p_error is null then now() else null end,
         last_error = p_error
   where id = p_id;
$function$;

-- ============================================================
-- public.delete_my_account — DPDP erasure path, SECURITY DEFINER and
-- callable by any authenticated user. Scoped entirely through
-- auth.uid() -> guardian lookup; there is no caller-supplied ID anywhere in
-- this function, which is what AXON_FIX_BRIEF.md §4.E3 asks to be
-- confirmed with an actual test rather than a reading. Erasure keeps a
-- tombstone row (guardian/student rows are scrubbed and marked deleted_at,
-- not removed) so the ledger of "someone existed here" survives; papers and
-- their content are hard-deleted via the cascade on public.paper.
-- ============================================================
CREATE OR REPLACE FUNCTION public.delete_my_account()
 RETURNS jsonb
 LANGUAGE plpgsql
 SECURITY DEFINER
 SET search_path TO 'public', 'pg_temp'
AS $function$
declare
  v_auth     uuid := (select auth.uid());
  v_guardian uuid;
  v_students int;
begin
  if v_auth is null then raise exception 'not authenticated' using errcode='42501'; end if;

  select g.id into v_guardian from public.guardian g where g.auth_user_id = v_auth;
  if v_guardian is null then raise exception 'no account for this session' using errcode='42501'; end if;

  -- Papers first. This cascades paper_page, student_attempt, mark_loss_event,
  -- page_unreadable and attempt_concept — all of the actual content.
  delete from public.paper
   where student_id in (select id from public.student where guardian_id = v_guardian);

  delete from public.student_subject
   where student_id in (select id from public.student where guardian_id = v_guardian);

  -- Strip the student's personal data, keep the row for the ledger.
  update public.student
     set first_name = '[erased]',
         deleted_at = now(),
         updated_at = now()
   where guardian_id = v_guardian
     and deleted_at is null;
  get diagnostics v_students = row_count;

  delete from public.app_preference where guardian_id = v_guardian;

  update public.guardian
     set name = '[erased]', contact = '[erased]',
         verified_at = null, verification_method = null, verification_ref = null,
         auth_user_id = null, deleted_at = now(), updated_at = now()
   where id = v_guardian;

  -- Releasing the auth row is what ends access. Last, because auth.uid() is
  -- needed above.
  delete from auth.users where id = v_auth;

  return jsonb_build_object(
    'erased', true,
    'students_erased', v_students,
    'guardian_retained_as_tombstone', v_guardian);
end;
$function$;
