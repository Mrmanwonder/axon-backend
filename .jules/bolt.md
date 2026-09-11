## 2024-06-25 - Cloudflare Workers Queue Batching
**Learning:** In Cloudflare Workers architecture, avoid sequential network operations (e.g., Queue `send()` calls) inside loops, which cause N+1 latency bottlenecks. `Queue.sendBatch()` has a strict limit of 100 messages per batch.
**Action:** Always refactor sequential queue dispatches into arrays chunked to lengths of 100 or fewer and use `Promise.all()` with `sendBatch()` for optimized concurrent execution.
