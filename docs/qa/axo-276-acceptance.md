# AXO-276 backend remediation

The consolidated acceptance ledger is in
[Axon-Site docs/qa/axo-276-acceptance.md](https://github.com/Mrmanwonder/Axon-Site/blob/fix/axo-276-audit-remediation/docs/qa/axo-276-acceptance.md).
The master [AXO-276](https://linear.app/axon-26/issue/AXO-276) remains In Progress:
source/automated fixes do not certify production, device, account-isolation or provider billing.

Baseline: `773893edf4def6133862f60e78f83c0da06b844c`. Existing baseline CI/deployment
workflow [38425313391](https://github.com/Mrmanwonder/axon-backend/actions/runs/38425313391)
was successful; currently running versions were not independently observed.

B-01: metadata JSON is read with a streamed 256KiB cap (Tutor 100,000 bytes),
10-second timeout, media-type check, UTF-8 decoding and predictable 413/415/408.
Malformed JSON retains existing null/400 semantics. No response includes request data.
Images continue to upload directly to R2. Signed provider callbacks, intelligence and
document-vision retain their separate contracts and require separate acceptance.

B-02: every queue delivery has an AbortSignal and a Supabase client whose fetches
use it. Handler budget now covers the initial heartbeat. Timeout aborts before
replacement enqueue/native retry. Triage/structure/content/adjudicate/explain pass
that signal to paid model requests. Aborted work cannot start an inline model retry,
return a successful model result, or issue new database requests. Durable terminal
recording has a fresh controller/client and retains the existing acknowledgement rule.

This cannot roll back an already committed database operation or guarantee provider
charge cancellation. R2 and Queue bindings do not accept this signal. Existing work
leases, fenced checkpoint saves, durable outbox and idempotency remain required;
physical worker-crash/redelivery and live cost acceptance are still NOT TESTED.

Rollback: revert this PR; no migrations, scanner contracts, model routes or prompts
change. Follow the consolidated ledger for B-03/B-04/B-05 and production gates.
