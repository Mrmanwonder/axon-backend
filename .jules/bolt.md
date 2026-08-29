## 2026-08-29 - Optimize build_syllabus_context
**Learning:** Found N+1 query issue in `build_syllabus_context` (functions/app.py) where Firestore is queried inside a loop for each learning objective ID. This can cause latency overhead and OOM errors at scale. Also, Firestore in-memory mock `_FakeCollectionRef.where` needs to be updated to support the `in` operator when refactoring to batched queries.
**Action:** Use an in-memory thread-safe TTL cache with batched 'in' queries chunked by limits of 30, falling back to cache miss lookups safely.
