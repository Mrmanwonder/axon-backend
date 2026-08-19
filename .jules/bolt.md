## 2024-05-24 - N+1 Firestore queries in Python Backend
**Learning:** Sequential `.get()` calls to Firestore inside loops in the serverless backend (like `build_syllabus_context` in `functions/app.py`) cause severe cold-start latency due to multiple network round-trips.
**Action:** Replace `for id in ids: ref.where('id', '==', id).get()` with batched queries using the Firestore `in` operator (e.g. `ref.where('id', 'in', chunk_of_30_ids).get()`), and reconstruct the final output by iterating over the original list of requested IDs to preserve deterministic ordering.
