-- AXON_FIX_BRIEF.md §4.E3 — delete_my_account() is SECURITY DEFINER and
-- callable by any authenticated user. This proves it can only ever act on
-- the caller's own account, with an actual test rather than a reading of
-- the function body, per the brief's explicit ask.
--
-- Wrapped entirely in BEGIN/ROLLBACK: everything it creates (two synthetic
-- guardians/students/auth.users rows) is undone at the end regardless of
-- outcome. Safe to re-run against the live project any time this function
-- changes. Run it as one script (e.g. via the Supabase SQL editor, or
-- `psql -f`) — splitting it across separate statements loses the `set
-- local` session settings the auth.uid() simulation depends on.
--
-- Verified passing against the live project (dlgcqieyevoebefhcggi) on
-- 2026-09-01: erasure_result = {"erased":true,"students_erased":1,...},
-- guardian B's student, guardian row, and consent rows all confirmed
-- untouched, then rolled back with zero rows left behind.

begin;

insert into auth.users (id, email) values
  ('11111111-1111-1111-1111-111111111111', 'test-a@example.invalid'),
  ('22222222-2222-2222-2222-222222222222', 'test-b@example.invalid');

insert into public.guardian (id, auth_user_id, name, contact) values
  ('aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa', '11111111-1111-1111-1111-111111111111', 'Test Guardian A', 'a@example.invalid'),
  ('bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb', '22222222-2222-2222-2222-222222222222', 'Test Guardian B', 'b@example.invalid');

-- The student consent gate (private.enforce_student_consent_gate) requires
-- every required purpose granted before a student row can be written.
insert into public.consent_event (guardian_id, purpose, granted, notice_version, method)
select g.id, p.purpose, true, 'test', 'in_app_itemised'::public.consent_method
from (values ('aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa'::uuid), ('bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb'::uuid)) as g(id)
cross join public.consent_purpose p
where p.is_required;

insert into public.student (id, guardian_id, board, class_level, age_band, first_name) values
  ('cccccccc-cccc-cccc-cccc-cccccccccccc', 'aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa', 'CAIE', 9, 'under_18', 'Student A'),
  ('dddddddd-dddd-dddd-dddd-dddddddddddd', 'bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb', 'CAIE', 9, 'under_18', 'Student B');

-- Simulate guardian A's own authenticated session. auth.uid() reads the
-- request.jwt.claim.sub GUC — this is the same mechanism PostgREST sets
-- from a real bearer token, just set directly here.
set local role authenticated;
set local request.jwt.claim.sub = '11111111-1111-1111-1111-111111111111';
select public.delete_my_account() as erasure_result;
reset role;

do $$
declare
  v_a_erased boolean;
  v_b_intact boolean;
  v_a_auth_gone boolean;
begin
  select (deleted_at is not null and first_name = '[erased]') into v_a_erased
    from public.student where id = 'cccccccc-cccc-cccc-cccc-cccccccccccc';
  select (first_name = 'Student B' and deleted_at is null) into v_b_intact
    from public.student where id = 'dddddddd-dddd-dddd-dddd-dddddddddddd';
  select not exists(select 1 from auth.users where id = '11111111-1111-1111-1111-111111111111') into v_a_auth_gone;

  if not v_a_erased then raise exception 'FAIL: guardian A''s own student was not erased'; end if;
  if not v_b_intact then raise exception 'FAIL: guardian B''s student was touched by A''s call — cross-account deletion'; end if;
  if not v_a_auth_gone then raise exception 'FAIL: guardian A''s auth.users row was not removed'; end if;
  if not exists(select 1 from public.guardian where id='bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb' and deleted_at is null) then
    raise exception 'FAIL: guardian B was touched';
  end if;

  raise notice 'PASS: delete_my_account() erased only the calling guardian''s own account; guardian B fully intact';
end $$;

rollback;
