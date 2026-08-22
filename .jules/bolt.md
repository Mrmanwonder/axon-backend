## 2025-03-10 - Fetching Multiple Specific Firestore Documents Without N+1
**Learning:** In this backend, iterating over IDs to fetch individual documents from Firestore creates severe N+1 latency.
**Action:** When mapping IDs to Firestore documents, use batched `in` queries (chunked to the Firestore limit of 30) combined with an in-memory TTL cache (e.g., using a `threading.Lock`). Crucially, cache empty strings for "not found" results to avoid repeated query attempts for missing IDs, and reconstruct the final list in its original order since batched results return unordered.
