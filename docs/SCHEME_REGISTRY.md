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


## Production review and approval runbook

Treat source discovery, source review, approval, ingestion, and runtime eligibility as separate gates.

### Roles and evidence

- **Discoverer/operator** runs `discover` and `verify`; this step must not need database-write credentials.
- **Reviewer/approver** confirms the first-party hostname, active source policy, subject/class/session identity, parser version, SHA-256 hashes, parsed question counts, coverage, and any parser diagnostics before a write is authorized.
- **Production operator** performs the write only from the exact reviewed source bytes or a cryptographically bound reviewed bundle. A production credential must never be printed, committed, or placed in an artifact.
- The production record must preserve the source URLs, source version, exact scheme SHA-256, parser version, verified coverage/counts, and the verification run or review reference in `scheme_document.metadata`.

A reviewed bundle is valid only while all of those immutable identifiers still match the target `scheme_document`. If the source bytes or policy changed after review, stop and review the new version instead of updating the old row in place.

### Approval sequence

1. Run `discover` against the current first-party index.
2. Run `verify` against the exact SQP/MS pair.
3. Review identity, hashes, parser version, coverage, and question count.
4. Confirm the active `scheme_source_policy` still permits ingestion/reproduction for that exact source class.
5. Ingest into a `pending` scheme document and canonical questions.
6. Assert every canonical question points to the same assessment identity, scheme document, source URL, and source version.
7. Only after the corpus is complete, move the document to `ready`.
8. Re-run the same source once to prove idempotence. An unchanged hash must not create another version or downgrade a ready document.

Never make a partially ingested document runtime-eligible.

### Reparse and source changes

- Same bytes + same hash: reuse the immutable document version; reparsing may refresh derived rows only after review.
- Changed bytes/hash: create a new `scheme_document` version and link the previous active version through `superseded_by_id`.
- Never reactivate a superseded or revoked hash through ingestion.
- Parser-version changes require a review of the new parsed output even when the source hash is unchanged.
- Retain historical rows required to explain already-generated provenance.

### Emergency disable / rights withdrawal

If source validity, rights, or parser integrity becomes uncertain, fail closed immediately:

1. disable the relevant `scheme_source_policy` or set reproduction permission false to stop new runtime use at the policy gate;
2. revoke the affected document with `schemes:cbse -- revoke ... --write` and record the reason;
3. confirm Tier 2 no longer returns evidence from that document;
4. investigate/review before any replacement document becomes `ready`.

Do not delete the historical document or canonical rows merely to disable runtime retrieval; revocation preserves auditability.

### Release evidence

For each production corpus promotion, retain:

- verification workflow/run reference;
- source URLs and source-policy version;
- SQP and MS SHA-256 hashes;
- parser version;
- parsed block counts, expected-question count, verified coverage, and canonical-question count;
- final production document ID and assessment identity ID;
- an idempotence check;
- a runtime proof showing Tier 2 persisted the exact assessment/document/canonical-question provenance, or an explicit note that no matching production paper exists yet.
