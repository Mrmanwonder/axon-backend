## 2024-05-18 - Fix N+1 queries by caching Firestore 'not found' queries
**Learning:** In the serverless Python backend, repeatedly querying Firestore for missing keys using the 'in' operator causes bottlenecks if you don't explicitly cache 'not found' values (e.g. as empty dicts). Failed lookups otherwise bypass the cache and hit the database every single time.
**Action:** When implementing in-memory caches, explicitly populate cache misses with empty objects during batch lookups, and ensure locks do not block network I/O.
