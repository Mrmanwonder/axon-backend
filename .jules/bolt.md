## 2024-05-14 - Optimize N+1 Queries in Firestore with Memory Cache
**Learning:** In the Python backend, `build_syllabus_context` issues an N+1 query pattern by iterating `learning_objective_ids` and calling `.get()` for each ID on the Firestore `syllabus_maps_collection`. This blocks the thread synchronously for each item.
**Action:** Replace the N+1 pattern with an `in` operator (chunked by 30 since Firestore limits `in` queries to 30 items) and wrap it in a thread-safe `SyllabusCache` with TTL and bounded size. Ensure failed/empty lookups are cached as empty strings to prevent repeated missing key lookups poisoning the cache strategy.

## 2024-05-14 - Thread Safety and Import Awareness
**Learning:** Adding new modules like `threading` or `time` to a script requires explicitly importing them at the top level. Missing these imports will lead to runtime crashes (`NameError`). Furthermore, handling transient errors (e.g., from Firestore queries) correctly is crucial when caching data, so as not to cache failed queries and degrade functionality for the TTL period.
**Action:** Always manually verify that all standard library modules required by new classes or functions are imported at the top of the file. Explicitly avoid caching failed network requests to prevent cache poisoning.
