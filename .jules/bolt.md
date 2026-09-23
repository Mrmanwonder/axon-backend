
## 2024-05-18 - [Optimize Cloudflare Worker Sequential Awaits]
**Learning:** Sequential awaits inside array iteration loops (like `for...of`) are a severe N+1 anti-pattern in serverless code (like Cloudflare Workers), creating massive cumulative latency when talking to R2 or the database. This architecture must maximize concurrent operations.
**Action:** When mapping over items that require independent network/DB operations (e.g. `headObject` or inserting database rows), use `Promise.all` to parallelize them. If early returns are required, split the sequence into a synchronous validation loop first, followed by a concurrent `Promise.all` mapping phase for the valid items.
