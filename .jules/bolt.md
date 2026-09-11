## 2024-09-05 - Cloudflare Workers N+1 Latency Optimization
**Learning:** In Cloudflare Workers architecture, sequential network operations like `R2.headObject`, database queries, or `Queue.send()` calls inside loops create severe N+1 latency bottlenecks, compounding execution time unnecessarily.
**Action:** Always replace sequential async operations in loops with concurrent executions. Use `Promise.all()` for network/database calls and `Queue.sendBatch()` (chunked to a maximum of 100 messages) for queue dispatches.
