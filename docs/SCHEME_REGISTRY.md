# Official marking-scheme registry

Axon never fetches an awarding-body marking scheme during a student marking
request. Official material is ingested offline once, verified, hashed, stored
with provenance, and then retrieved only through an exact assessment identity.

## Rights boundary

The database-side `scheme_source_policy` is authoritative.

- **CBSE**: first-party public marking schemes may be ingested when the active
  policy permits reproduction and the source host is `cbseacademic.nic.in`.
- **Cambridge**: metadata-only unless Axon has an applicable explicit licence.
  Cambridge states that it does not grant permission to reproduce mark schemes.
- **IB**: metadata-only unless Axon has applicable written/licensed permission
  for the protected assessment material.

The admin CLI refuses ingestion when the stored source policy does not allow it.

## CBSE admin workflow

Prerequisites:

- Node 22
- `pdftotext` from Poppler
- for database writes only: `SUPABASE_URL` and
  `SUPABASE_SERVICE_ROLE_KEY`

Discover first-party Class XII pairs without touching Supabase:

```sh
npm run schemes:cbse -- discover --class=12
```

Filter to one subject:

```sh
npm run schemes:cbse -- discover --class=12 --subject=Physics
```

Verify the live official PDFs through the exact parser without Supabase credentials:

```sh
npm run schemes:cbse -- verify --class=12 --subject=Physics
```

This is also run by the dedicated weekly/PR source-smoke workflow. It performs
no database writes.

Dry-run ingestion. This downloads the official SQP/MS, verifies identity,
question labels and mark totals, and prints the result without writing:

```sh
SUPABASE_URL=... SUPABASE_SERVICE_ROLE_KEY=... \
  npm run schemes:cbse -- ingest --class=12 --subject=Physics
```

Write only after the dry run is clean:

```sh
SUPABASE_URL=... SUPABASE_SERVICE_ROLE_KEY=... \
  npm run schemes:cbse -- ingest --class=12 --subject=Physics --write
```

The write path can enrich an otherwise uniquely matched CBSE subject offering
with the official three-digit subject code. It refuses conflicting or ambiguous
matches.

Revoke a document if rights or source validity changes:

```sh
SUPABASE_URL=... SUPABASE_SERVICE_ROLE_KEY=... \
  npm run schemes:cbse -- revoke \
  --document=<scheme-document-uuid> \
  --reason="Official source withdrawn" --write
```

## Fail-closed rules

A document is not runtime-eligible unless all of these are true:

1. exact assessment identity exists;
2. exact or deterministic parent question label exists;
3. source policy is active and reproduction-permitted;
4. document access class is `public_official` or `licensed_official`;
5. document is not revoked;
6. source hostname matches the stored policy;
7. extraction status is ready/complete/extracted;
8. the canonical question carries scheme source + version + document ID.

Parser ambiguity, mismatched SQP/MS identity, mismatched mark totals, or less than
75% verified top-level question coverage causes ingestion to fail rather than
inventing missing scheme content.


## Production operating model

The registry has three distinct roles. One person may hold more than one role in
a small team, but the evidence gates remain the same.

- **Source reviewer** confirms first-party identity, access class, source policy,
  assessment metadata, parser coverage and any rights/licensing limits.
- **Production operator** runs the protected ingestion/revoke workflow only
  after review evidence is attached to the owning Linear issue.
- **Incident owner** can revoke an exact document immediately and coordinates a
  provider-wide disable through a reviewed Supabase migration if policy or
  licensing changes invalidate multiple documents.

The owning work items are AXO-12 (marking-scheme registry), AXO-34 (production
rollout) and AXO-26 (ongoing curriculum/source drift operations).

### One-time GitHub environment setup

Create a GitHub Actions environment named `production-scheme-registry`.

Configure it with:

1. required reviewers;
2. deployment branch restricted to `main`;
3. environment secret `SUPABASE_URL`;
4. environment secret `SUPABASE_SERVICE_ROLE_KEY`.

Do not add the service-role key as a repository-wide secret and do not expose it
to pull-request workflows. The manual admin workflow is the only GitHub Actions
path that consumes it.

The workflow is:

`.github/workflows/scheme-registry-admin.yml`

It is `workflow_dispatch` only. `verify` is read-only and does not use
Supabase credentials. `ingest` and `revoke` require environment approval and
the literal confirmation value `PRODUCTION`.

## Review -> approve -> ingest runbook

### 1. Discover and verify without database credentials

Run locally or use the weekly/read-only smoke workflow:

```sh
npm run schemes:cbse -- discover --class=12 --subject=Physics
npm run schemes:cbse -- verify --class=12 --subject=Physics
```

Record in the Linear task:

- first-party question-paper URL;
- first-party marking-scheme URL;
- provider/programme/subject code;
- exam year/session/route;
- parser version;
- SQP and scheme SHA-256 values;
- verified question count and parser coverage;
- active `scheme_source_policy` ID/access class;
- reviewer name/date and any source caveats.

Reject the candidate if identity is ambiguous, the source policy is inactive,
reproduction is not permitted, the source host is unexpected, top-level mark
totals disagree, or verified coverage is below the parser threshold.

### 2. Run the read-only Actions verification

Open **Actions -> Scheme registry production admin -> Run workflow** and choose:

- action: `verify`
- class: the reviewed class
- subject: the reviewed subject

Attach the generated verification log artifact (or its run ID) to the Linear
issue. The log is evidence of the source state that was reviewed; it is not
permission to skip the write path's second verification.

### 3. Approve production mutation

A source reviewer checks the read-only output against the Linear manifest. The
production operator then starts the same workflow with:

- action: `ingest`
- reviewed class/subject
- confirmation: `PRODUCTION`

A required reviewer on the `production-scheme-registry` environment approves
the job. The job first runs an ingestion dry-run and then runs the real write.
The write path downloads and validates the official sources again, so a source
change between review and mutation cannot bypass parser/hash/provenance checks.

### 4. Verify the write before enabling a canary

After ingestion, query production with a privileged read path:

```sql
select
  sd.id,
  sd.assessment_identity_id,
  sd.source_url,
  sd.source_version,
  sd.sha256,
  sd.parser_version,
  sd.copyright_access_class,
  sd.extraction_status,
  sd.retrieved_at,
  sd.supersedes_id,
  sd.superseded_by_id,
  sd.revoked_at,
  count(cq.id) as canonical_questions
from public.scheme_document sd
left join public.canonical_question cq
  on cq.scheme_document_id = sd.id
where sd.id = '<reviewed-document-id>'
group by sd.id;
```

Required result:

- one exact document;
- `extraction_status = 'ready'`;
- immutable hash/parser/source version populated;
- access class is `public_official` or `licensed_official`;
- not revoked and not superseded;
- canonical question count matches the reviewed ingestion output.

Then verify every canonical row points back to the same assessment/document:

```sql
select count(*) as mismatched_rows
from public.canonical_question cq
join public.scheme_document sd on sd.id = cq.scheme_document_id
where cq.scheme_document_id = '<reviewed-document-id>'
  and (
    cq.assessment_identity_id is distinct from sd.assessment_identity_id
    or cq.scheme_source is distinct from sd.source_url
    or cq.scheme_version is distinct from sd.source_version
  );
```

`mismatched_rows` must be zero.

## Production canary for Tier 2

Use a paper whose exact assessment identity is already in the reviewed corpus.
Do not use a semantically similar paper.

Trace one run through:

1. `paper.assessment_identity_id` — must equal the reviewed assessment;
2. `question_region.canonical_question_id` — must point inside that assessment;
3. `region_explanation.tier` — must be `tier_2` only when verified evidence
   was returned;
4. `region_explanation.assessment_identity_id`;
5. `region_explanation.scheme_document_id`;
6. `region_explanation.canonical_question_id`;
7. `region_explanation.scheme_retrieval_mode`;
8. `region_explanation.scheme_source` + `scheme_version`.

The three immutable IDs must join to the same stored document/assessment and the
source/version text must match that document. If any field is absent or
inconsistent, treat the run as Tier-1-only and do not claim official grounding.

Web/Tavily output is never acceptable evidence for this check.

## Drift and re-ingestion

The read-only source smoke job detects source/parser drift. When official bytes
change:

1. re-run `verify`;
2. review the new hash and parsed output;
3. ingest through the protected workflow;
4. keep the old `scheme_document` immutable;
5. allow the ingestion lifecycle to link `supersedes_id` /
   `superseded_by_id`;
6. rerun the production canary before relying on the new version.

Never update a ready scheme document's hash/source version in place.

## Revocation and emergency disable

For one bad/withdrawn document, use the protected workflow:

- action: `revoke`
- document ID: exact `scheme_document.id`
- reason: concrete operational/legal reason
- confirmation: `PRODUCTION`

Verify `revoked_at` and `revocation_reason` are populated. New retrieval must
return no evidence from that document; existing historical explanation rows keep
their immutable evidence IDs for audit.

If an entire provider/source class becomes invalid, stop ingestion immediately
and ship a reviewed Supabase migration that sets the affected
`scheme_source_policy.active = false` (or
`reproduction_permitted = false` as appropriate). Do not edit production
policy ad hoc in a dashboard. Re-run retrieval regression tests and revoke
affected ready documents as required by the incident review.

## Operational evidence checklist

A production operation is complete only when the owning Linear issue contains:

- source verification run ID;
- exact source URLs and hashes;
- assessment identity;
- scheme document ID/version;
- parser version and verified coverage;
- approval/reviewer record;
- production mutation run ID;
- post-write SQL verification;
- canary run ID when the change is intended for live Tier 2;
- revoke/supersession evidence when replacing an existing source.

Secrets, downloaded copyrighted source files, and service-role values are never
attached to Linear or GitHub artifacts.
