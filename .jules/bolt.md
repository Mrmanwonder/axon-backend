## 2024-06-20 - Concurrent HTTP fetching with `asyncio` and `httpx`

**Learning:** When fetching multiple resources sequentially over the network (like PDF files), blocking I/O calls (e.g., `requests.get`) limit the performance linearly based on the number of resources. We can significantly improve the performance by parallelizing the requests.

**Action:** Replace sequential `requests.get` with concurrent asynchronous fetching using `httpx.AsyncClient` combined with `asyncio.gather`. Be careful to use `asyncio.Semaphore` when fetching a large list to prevent overwhelming the application memory (OOM) or hitting rate limits. Furthermore, when adapting a synchronous function to use `asyncio.run()`, properly handle exceptions (e.g., `RuntimeError` if an event loop is already running) to prevent silent application errors.
