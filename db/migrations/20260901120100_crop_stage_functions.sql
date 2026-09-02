-- WP4, part two: the gate that puts cropping between structure and content, and
-- the one batched write the crop worker uses.
--
-- Applied as a separate migration from the enum addition on purpose. Postgres
-- will not let a transaction use an enum value it added itself, and
-- `advance_after_structure` below advances runs to 'cropping'.

-- ── structure now hands off to cropping, not to content ────────────────────
--
-- The shape is otherwise unchanged from the version this replaces, including
-- the "already done" re-entry path (§3.2) that the caller depends on: it still
-- returns advanced:false while any page is mid-structure, and still refuses to
-- advance a run that is not in 'structure'.
--
-- What changes is what it enqueues. It used to return the region ids for the
-- content queue; it now returns the page ids for the crop queue, and
-- `advance_after_crop` returns the region ids once the crops have settled.
-- `enqueue_reconcile` keeps its meaning exactly: a run that produced no regions
-- at all has nothing to crop and nothing to read, and goes straight to
-- reconciliation rather than sitting in a stage with no work in it.
create or replace function public.advance_after_structure(p_run_id uuid)
returns jsonb
language plpgsql
security definer
set search_path to 'public', 'pg_temp'
as $$
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

  -- Only pages a region actually sits on. A cover sheet with no questions on it
  -- has nothing to cut and should not be sent a message that will decode a
  -- 2400px page to discover that.
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
end; $$;

-- ── cropping hands off to content ──────────────────────────────────────────
--
-- Counts only pages still pending or running, so a page whose crop failed is
-- counted as finished and the run moves on. That is the whole "content waits
-- for crops but is never blocked by a crop failure" requirement, and it is
-- one word of SQL: 'failed' is not in the list.
create or replace function public.advance_after_crop(p_run_id uuid)
returns jsonb
language plpgsql
security definer
set search_path to 'public', 'pg_temp'
as $$
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
end; $$;

-- ── one write for a page's worth of crops ──────────────────────────────────
--
-- §8.2 is explicit that the subrequest budget is what took the content stage
-- down once already. A forty-question page is forty keys; this makes it one
-- call. Rows are {id, crop_key, cropmask_key}.
--
-- Scoped by run_id as well as region id so a malformed batch cannot reach
-- across runs, and it only ever writes the two key columns — nothing here can
-- touch a mark, a confidence tier, or a student's correction.
create or replace function public.apply_region_crops(p_run_id uuid, p_rows jsonb)
returns integer
language plpgsql
security definer
set search_path to 'public', 'pg_temp'
as $$
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
end; $$;

-- Postgres grants EXECUTE to PUBLIC on creation, and anon/authenticated inherit
-- through it — revoking from those two roles by name is a no-op. This is the
-- correction that had to be made for `apply_region_confidence` on 2026-08-31
-- and it is written the right way round from the start here. These are
-- SECURITY DEFINER functions the pipeline calls with the service key; nothing
-- holding a student's session has any business calling them.
revoke execute on function public.advance_after_structure(uuid) from public;
revoke execute on function public.advance_after_crop(uuid) from public;
revoke execute on function public.apply_region_crops(uuid, jsonb) from public;
grant execute on function public.advance_after_structure(uuid) to service_role;
grant execute on function public.advance_after_crop(uuid) to service_role;
grant execute on function public.apply_region_crops(uuid, jsonb) to service_role;
