## 2024-05-15 - Batched Queue Dispatches
**Learning:** Sequential `Queue.send()` calls in Cloudflare Workers within loops cause significant N+1 network latency. `Queue.sendBatch()` provides a solution, but it enforces a strict 100-message limit per batch.
**Action:** Always chunk message arrays to lengths of 100 or fewer and dispatch them concurrently using `Promise.all` and `sendBatch` instead of using a `for...of` loop with `send`. When typing the promises for `sendBatch` in TypeScript, do not type the array explicitly (let it infer) to prevent TS2345 errors.
