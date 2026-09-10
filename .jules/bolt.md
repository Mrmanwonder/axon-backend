## 2024-03-24 - [Batch Queue Dispatches]
**Learning:** Sequential `Queue.send()` calls inside a loop introduce severe N+1 latency bottlenecks in Cloudflare Workers. `Queue.sendBatch()` is the performant alternative, but Cloudflare strictly limits batches to 100 messages.
**Action:** Always use `Promise.all()` with `Queue.sendBatch()` chunked into arrays of 100 or fewer for fan-out messaging.
