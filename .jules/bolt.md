## 2024-09-04 - Queue Batching in Cloudflare Workers
**Learning:** Cloudflare Workers impose a strict 100-message limit per `Queue.sendBatch()` call. Attempting to send an arbitrarily sized array of messages concurrently will fail if it exceeds this hard limit.
**Action:** Always chunk arrays into sizes of 100 or less before calling `sendBatch()` to dispatch messages to a queue.
