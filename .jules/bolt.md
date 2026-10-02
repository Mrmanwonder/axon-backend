## 2026-09-28 - [Optimize Queue Dispatch]
**Learning:** In Cloudflare Workers architecture, `Queue.sendBatch()` has a strict limit of 100 messages per batch. Passing arrays mapped directly to `.sendBatch()` throws errors on larger payloads or can silently drop messages due to size.
**Action:** Use a loop to chunk the array into sizes of 100 or less and issue `sendBatch()` using `Promise.all()` for concurrent dispatches to avoid N+1 bottleneck.
## 2024-05-24 - [N+1 Bottleneck in Workers API]
**Learning:** Found an N+1 bottleneck in `workers/api/src/index.ts` where `headObject` was awaited sequentially inside a `for...of` loop on retries. I resolved it using `mapLimit` and the `IO_CONCURRENCY` constant already present in the file.
**Action:** Always check for sequential network calls inside loops in Cloudflare Workers pipelines. Use bounded concurrency utilities like `mapLimit` instead of bare `Promise.all()` when making many external requests.
