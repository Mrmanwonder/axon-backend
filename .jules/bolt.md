## 2024-05-24 - Optimize Firestore Syllabus Query
**Learning:** Fetching items sequentially in a loop using individual Firestore `.get()` queries is an anti-pattern that can cause severe N+1 query problems and degraded performance, especially in serverless environments.
**Action:** Use batched chunked queries via the Firestore `in` operator (up to the limits of 30) paired with an in-memory TTL cache. Explicitly cache successful 'not found' queries to avoid retry loops, and reconstruct final output matching requested order since batching alters native result order.
