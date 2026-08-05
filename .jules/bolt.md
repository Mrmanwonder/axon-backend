## 2024-08-05 - Firestore IN limits and Cache poisoning

**Learning:** Firestore restricts `in` operator queries to a maximum of 30 items. Combining this limitation with a TTL cache is great for performance, but if database queries fail, you must ensure exceptions propagate natively and do NOT aggressively cache empty results, otherwise you introduce poison caching and drop errors silently.

**Action:** Always structure batch queries using chunks (e.g. `chunk_size = 30`), and let network or permissions errors raise naturally out of the try block instead of swallowing them with `pass`.
