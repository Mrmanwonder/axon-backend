/**
 * Dispatch items to a Cloudflare Queue in batches of up to 100, honoring
 * the Queue API's batch size limit while bounding concurrency to prevent
 * overwhelming downstream workers or memory limits.
 *
 * It throws on the first error encountered, immediately halting dispatch of
 * any subsequent batches.
 *
 * @param queue The Cloudflare Queue binding.
 * @param items The array of items to send.
 * @param mapper A function that maps an item to a { body: ... } structure.
 * @param concurrencyLimit Maximum number of concurrent sendBatch requests (default: 5).
 */
export async function chunkedSendBatch<T, U>(
  queue: { sendBatch: (messages: U[]) => Promise<any> },
  items: T[],
  mapper: (item: T) => U,
  concurrencyLimit: number = 5
): Promise<void> {
  const maxInFlight = concurrencyLimit;
  let inFlight = 0;
  let error: unknown = null;

  const promises = new Set<Promise<void>>();

  for (let i = 0; i < items.length; i += 100) {
    if (error) break;
    const chunk = items.slice(i, i + 100);
    const mappedChunk = chunk.map(mapper);

    while (inFlight >= maxInFlight) {
      // Wait for at least one promise to resolve before dispatching the next batch
      await Promise.race(promises);
      if (error) break;
    }

    if (error) break;

    inFlight++;

    // We create a promise that catches its own errors so Promise.race doesn't
    // unhandled-reject. We store the error to break the loop.
    const p = queue.sendBatch(mappedChunk)
      .then(() => {})
      .catch((err: any) => {
        error = err;
      }).finally(() => {
        inFlight--;
        promises.delete(p);
      });

    promises.add(p);
  }

  // Wait for all remaining dispatches to finish (or fail)
  await Promise.all(promises);
  if (error) {
    throw error;
  }
}
