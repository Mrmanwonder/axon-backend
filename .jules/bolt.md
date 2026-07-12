## 2025-02-18 - Prevent cache poisoning from Firestore errors
**Learning:** When implementing in-memory caching for batched Firestore queries, it's crucial to differentiate between actual "missing" documents (which we want to negative-cache) and missing results due to transient errors/timeouts. Caching `None` when an error occurs poisons the cache, making valid IDs unresolvable for the entire TTL duration.
**Action:** Track failed query batches explicitly. Avoid storing cache entries for items that were part of a batch request that threw an exception.
