## 2024-05-18 - Missing Standard Library Imports Breaking Caches
**Learning:** Adding standard modules like `time` or `threading` mid-file for a cache without placing the imports correctly at the top level causes `NameError`.
**Action:** Always verify all required imports for newly introduced primitives are correctly placed at the top-level of the file to comply with static analysis and avoid runtime crashes.
