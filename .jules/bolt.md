## 2023-10-24 - Cloudflare Queues batching
**Learning:** Cloudflare Workers have a 100-message limit per `sendBatch` call. Array of messages must be chunked to lengths of 100 or fewer before dispatching to avoid runtime limits.
**Action:** When migrating from `queue.send()` to `queue.sendBatch()`, implement chunking logic (e.g. `slice(i, i + 100)`) and dispatch with `Promise.all()` to ensure limits are respected while maintaining performance.
