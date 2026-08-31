## 2024-05-18 - [Optimize Syllabus Context Loading]
**Learning:** In the serverless Python backend, fetching entire static collections into memory is an anti-pattern. Also, performing N+1 queries using `.get()` in a loop causes high latency.
**Action:** Replace sequential `.get()` queries with an in-memory TTL cache and chunked `in` queries (batch size up to 30) for specific missing keys, which fixes the N+1 problem and handles caching efficiently.
