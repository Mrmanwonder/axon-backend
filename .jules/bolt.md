## 2026-06-20 - DB stream field-selection optimization for study pulse analysis
**Learning:** We can reduce db payload load by selecting only needed fields, removing querying of unused collections completely (`mastery` table), and avoid extra redundant memory copy allocations by using python generator expressions rather than `[..]` list syntax.
**Action:** When querying Firebase via Firestore `collection.stream()`, inspect if we are only using a handful of fields and implement `.select(["field1", "field2"])` before `.stream()`. Also check if collections are completely unused!
