## 2024-09-15 - Batched Queue Dispatches in Cloudflare Workers
**Learning:** Sequential queue dispatch loops (`await queue.send()`) block execution on network I/O per iteration, causing severe N+1 latency bottlenecks, especially when scaling (e.g., when a paper requires hundreds of explanations).
**Action:** Always replace sequential queue `send` loops with `sendBatch`, chunk arrays to respect Cloudflare`s 100-message limit per batch, and execute chunks concurrently using `Promise.all`.
