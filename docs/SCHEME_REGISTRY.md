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
