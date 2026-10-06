## 2024-05-18 - Cloudflare R2 Bulk Delete
**Learning:** Using `Promise.all()` to concurrently fire hundreds of individual R2 `delete()` operations consumes excessive subrequests, causes high memory spikes, and risks exceeding Cloudflare Worker subrequest limits.
**Action:** Use R2's bulk deletion method `bucket.delete(arrayOfKeys)` when deleting multiple objects. It accepts up to 1000 keys per request, significantly reducing latency and network overhead.
