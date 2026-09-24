## 2024-05-18 - Cloudflare Queue `sendBatch` Hard Limits
**Learning:** Cloudflare Workers' `Queue.sendBatch()` API throws runtime errors if given an array containing more than 100 messages. This limit causes silent or catastrophic failures during large fan-out operations (e.g. processing 100+ pages of a document).
**Action:** Always chunk arrays passed to `sendBatch` into batches of 100 or less, and use `Promise.all` to dispatch the resulting chunks concurrently.
