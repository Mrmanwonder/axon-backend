## 2026-09-07 - Cloudflare Workers Queue.sendBatch limit
**Learning:** In Cloudflare Workers, `Queue.sendBatch()` has a strict limit of 100 messages per batch. Arrays of messages must be chunked into lengths of 100 or fewer before dispatching to avoid runtime limits. This limit was hit when we tried to dispatch multiple messages in a batch.
**Action:** When using `Queue.sendBatch()` in Cloudflare Workers, always ensure that the array of messages is chunked into lengths of 100 or fewer before dispatching.
