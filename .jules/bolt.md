## 2026-09-28 - [Optimize Queue Dispatch]
**Learning:** In Cloudflare Workers architecture, `Queue.sendBatch()` has a strict limit of 100 messages per batch. Passing arrays mapped directly to `.sendBatch()` throws errors on larger payloads or can silently drop messages due to size.
**Action:** Use a loop to chunk the array into sizes of 100 or less and issue `sendBatch()` using `Promise.all()` for concurrent dispatches to avoid N+1 bottleneck.

## 2024-05-18 - R2 Bulk Deletion over Promise.all
**Learning:** Cloudflare Workers subrequest limits and network overhead make concurrent individual R2 deletions via `Promise.all(keys.map(key => bucket.delete(key)))` an anti-pattern. R2 natively supports array bulk deletion `bucket.delete(keys)` for up to 1000 items, dramatically reducing connection latency.
**Action:** When performing multi-key removals from R2 buckets in Workers, always prefer `bucket.delete([...keys])`. Ensure the array size is capped (e.g., via pagination limit) below R2's 1000-key limit.
