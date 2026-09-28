## 2026-09-28 - [Optimize Queue Dispatch]
**Learning:** In Cloudflare Workers architecture, `Queue.sendBatch()` has a strict limit of 100 messages per batch. Passing arrays mapped directly to `.sendBatch()` throws errors on larger payloads or can silently drop messages due to size.
**Action:** Use a loop to chunk the array into sizes of 100 or less and issue `sendBatch()` using `Promise.all()` for concurrent dispatches to avoid N+1 bottleneck.
