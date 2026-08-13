## 2024-05-18 - Avoid N+1 Firestore queries by using caching
**Learning:** In the Python backend, `build_syllabus_context` executes a Firestore query (`.get()`) for every ID in `learning_objective_ids`. This N+1 query issue can cause significant latency if the list of IDs is long.
**Action:** Implement a thread-safe, size-bounded TTL cache with chunked `in` queries (to fetch missing entries in batches up to 30) instead of executing `.get()` in a loop. Do not fetch entire static collections into memory.

## 2024-05-18 - Caching Eviction Logic Pitfall
**Learning:** When implementing an in-memory TTL cache, evicting only expired items when the cache hits a size limit can cause severe lock contention and memory leaks. If the working set of unexpired items exceeds the limit, every subsequent call will acquire the lock and loop over the entire cache without deleting anything.
**Action:** Always include a hard fallback (e.g., `cache.clear()`, or removing oldest elements) if deleting expired items doesn't bring the size back under the limit.

## 2024-05-18 - Text File Corruption
**Learning:** Ensure files like `requirements.txt` are clean UTF-8 plain text. If git detects binary corruption (e.g. from bad Windows PowerShell output), just resetting or committing the file again isn't enough - it must be forcibly removed with `git rm -f` and recreated cleanly before staging to resolve the binary diff issue.
**Action:** Use `git rm -f` and recreate files from scratch when encountering null bytes or unexpected `Binary files differ` messages in git for text files.
