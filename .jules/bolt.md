## 2025-05-19 - Fast API N+1 Query in `build_syllabus_context`
**Learning:** In `functions/app.py` `build_syllabus_context`, we fetch `syllabus_maps` objectives via `.get()` calls for each ID, leading to an N+1 query issue. Fetching entire collections via `syllabus_maps_collection().get()` also causes OOM errors. We should batch these queries.
**Action:** Use an `in` operator query over a chunked list of IDs (max 30 chunks for Firestore) instead of looping `.get()`. We should also incorporate an in-memory TTL cache to minimize repeating queries for the same objectives in batched operations. Make sure to preserve order since batch queries do not return results in the order requested.

## 2025-05-19 - Fast API Error Handling in Caching
**Learning:** When implementing batched queries with caching in Python backend, errors querying Firestore should not be silently swallowed by logging and ignoring them. Doing so could result in cache poisoning where failed IDs get cached as `None`, leading to persistent silent failures.
**Action:** Let exceptions bubble up natively, or raise an HTTPException with a 500 status code so the client knows it failed. Ensure we only cache chunks that were successfully retrieved.
