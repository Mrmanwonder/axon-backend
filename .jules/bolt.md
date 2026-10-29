## 2024-05-15 - [Initial setup]
**Learning:** Initial journal creation.
**Action:** Always document learnings.
## 2024-05-15 - [API Worker N+1 DB Select Optimization]
**Learning:** Sequential await loops on database lookups (like `await admin.from('upload').select(...)` in `workers/api/src/index.ts`) create an N+1 performance bottleneck.
**Action:** Always batch DB requests or use concurrency-bounded parallelization like `mapLimit` when resolving multiple database operations.
## 2024-05-15 - [API Worker N+1 DB Select Optimization II]
**Learning:** We need to fix the N+1 latency bottleneck in the API worker by running `headObject` and database checks concurrently with `mapLimit`.
**Action:** Replace for-loop with bounded concurrency.
## 2024-05-15 - [Test command fixes]
**Learning:** Some test scripts output 'not found' or similar issues. The test command requires full workspace setup.
**Action:** Use specific test command `npm run typecheck && npm run test && npm run dry-run && npm run check:contract-parity` as per guidelines.
