## 2024-05-18 - Cloudflare Queue `sendBatch` Type Inference

**Learning:** When using `Queue.sendBatch()` in a Cloudflare Workers TypeScript project, explicitly typing the array of promises (e.g. `const promises: Promise<void>[] = []`) causes TS2345 type check errors because `sendBatch` actually returns `Promise<QueueSendBatchResponse>`.
**Action:** Do not explicitly type the promises array when accumulating `sendBatch` calls. Instead, let TypeScript infer the type (e.g. `const promises = []`), and use `await Promise.all(promises);` to await completion without type clashes.
