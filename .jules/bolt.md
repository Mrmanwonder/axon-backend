## 2024-08-11 - Database Fetching Exceptions
**Learning:** Swallowing database exceptions during network calls changes the application logic. The original behavior allowed database errors to bubble up naturally. Swallowing them results in dropping valid records silently.
**Action:** When refactoring database logic to use batched fetching, keep the original error propagation behavior and allow exceptions to naturally propagate unless explicitly handled.
