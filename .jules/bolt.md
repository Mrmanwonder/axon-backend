## 2024-05-24 - N+1 Queries in Firestore
**Learning:** Sequential `.get()` queries on Firestore inside a loop cause massive latency. Batching with `in` (up to 30 items) combined with an in-memory thread-safe cache (`threading.Lock()`) drastically improves performance.
**Action:** When seeing loops with Firestore reads, always look to batch them and cache the results to eliminate N+1 queries.
