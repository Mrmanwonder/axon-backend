/**
 * An in-isolate mutex per key: callers with the same key run one at a time,
 * in arrival order; callers with different keys do not wait for each other.
 *
 * It only coordinates work inside ONE isolate, so it is never a correctness
 * guarantee on its own: the database still has the final word (unique
 * indexes, the paper lock in pipeline_write). What it removes is pointless
 * contention. When a whole paper's pages arrive in one queue batch and their
 * model calls finish within the same moment, letting them all read the run's
 * regions at once guarantees they all plan the same order_index and all but
 * one lose the insert (AXO-211). Taking turns for the short read-plan-write
 * step costs a few hundred milliseconds per paper and keeps the database
 * retry for the rarer case of two invocations racing.
 */
export class KeyedMutex {
  private tails = new Map<string, Promise<void>>();

  async run<T>(key: string, fn: () => Promise<T>): Promise<T> {
    const previous = this.tails.get(key) ?? Promise.resolve();
    let release!: () => void;
    const mine = new Promise<void>((resolve) => { release = resolve; });
    const tail = previous.then(() => mine);
    this.tails.set(key, tail);
    try {
      await previous;
      return await fn();
    } finally {
      release();
      if (this.tails.get(key) === tail) this.tails.delete(key);
    }
  }

  /** Keys with a holder or waiters (tests and diagnostics). */
  get size(): number {
    return this.tails.size;
  }
}
