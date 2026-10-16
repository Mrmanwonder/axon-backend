-- AXO-223: read-only measured-run provenance; returns no case/input/output text.
select r.id, r.kind, r.golden_set_version, r.stages, r.passed, r.started_at, r.finished_at,
       r.thresholds_ref, count(e.id) as result_count,
       count(distinct e.case_id) as distinct_case_count,
       count(*) filter (where e.status = 'done') as done_count,
       count(*) filter (where e.schema_valid is true) as schema_valid_count,
       array_agg(distinct e.model) as models,
       encode(extensions.digest(string_agg(jsonb_build_object(
         'case_id', e.case_id, 'candidate_key', e.candidate_key, 'model', e.model,
         'prompt_version', e.prompt_version, 'status', e.status,
         'schema_valid', e.schema_valid, 'output', e.output)::text, E'\n'
         order by e.case_id, e.candidate_key), 'sha256'), 'hex') as result_sha256
from public.eval_run r join public.eval_case_result e on e.eval_run_id = r.id
where r.id in ('7ca49ba7-6f11-4d11-b212-416d7d1987d1', 'c7a66b34-3c94-4609-9fe4-a8917fb8b2c5')
group by r.id order by r.id;

select stage, golden_set_version, count(*) as cases,
       count(*) filter (where needs_human_label = false and human_labels is not null
         and labelled_by is not null and labelled_at is not null) as labelled_cases,
       min(labelled_at) as first_labelled_at, max(labelled_at) as last_labelled_at
from public.eval_case
where golden_set_version in ('topic-tag-synthetic-v1', 'scheme-check-synthetic-v1')
group by stage, golden_set_version order by stage;
