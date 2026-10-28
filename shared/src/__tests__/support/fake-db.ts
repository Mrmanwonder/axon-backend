/**
 * A small stand-in for the Supabase client, for worker entrypoint tests (AXO-296).
 *
 * Reads: `from(table)` returns a chain whose filters are recorded and ignored; `maybeSingle()`
 * resolves to the table's single row (or the first of its rows), and awaiting the chain resolves
 * to all of them. A table given as a function receives the recorded filters, so a test can answer
 * differently per query. Writes (`insert`, `update`, `upsert`, `delete`) and `rpc` calls are
 * recorded; an rpc answers with whatever `rpc(name, args)` returns, and a write with `onWrite(query)`.
 *
 * Not matched by shared's own test glob (`src/__tests__/*.test.ts`), so it is support code only.
 */

export interface FakeQuery {
  table: string;
  filters: Array<[string, string, unknown]>;
  op: "select" | "insert" | "update" | "upsert" | "delete";
  payload?: unknown;
}

export type TableValue = unknown[] | Record<string, unknown> | null | ((q: FakeQuery) => unknown);

export interface FakeDb {
  sb: any;
  rpcCalls: Array<{ name: string; args: any }>;
  writes: FakeQuery[];
  reads: FakeQuery[];
  rpcNamed(name: string): any[];
}

export function fakeDb(
  tables: Record<string, TableValue> = {},
  rpc: (name: string, args: any) => unknown = () => null,
  /** What a write returns as `data` (e.g. the rows an update matched). Default: null. */
  onWrite: (q: FakeQuery) => unknown = () => null,
): FakeDb {
  const rpcCalls: Array<{ name: string; args: any }> = [];
  const writes: FakeQuery[] = [];
  const reads: FakeQuery[] = [];

  const resolve = (q: FakeQuery): unknown => {
    const value = tables[q.table];
    return typeof value === "function" ? (value as (q: FakeQuery) => unknown)(q) : value ?? null;
  };

  function chain(table: string): any {
    const q: FakeQuery = { table, filters: [], op: "select" };
    let recorded = false;
    const record = () => {
      if (recorded) return;
      recorded = true;
      (q.op === "select" ? reads : writes).push(q);
    };
    const filter = (kind: string) => (column: string, value?: unknown) => { q.filters.push([kind, column, value]); return b; };
    const b: any = {
      select: () => b,
      eq: filter("eq"), neq: filter("neq"), in: filter("in"), is: filter("is"), not: filter("not"),
      lt: filter("lt"), lte: filter("lte"), gt: filter("gt"), gte: filter("gte"), match: filter("match"),
      or: filter("or"), filter: filter("filter"), contains: filter("contains"),
      order: () => b, limit: () => b, range: () => b, returns: () => b,
      insert: (payload: unknown) => { q.op = "insert"; q.payload = payload; return b; },
      update: (payload: unknown) => { q.op = "update"; q.payload = payload; return b; },
      upsert: (payload: unknown) => { q.op = "upsert"; q.payload = payload; return b; },
      delete: () => { q.op = "delete"; return b; },
      maybeSingle: async () => {
        record();
        if (q.op !== "select") return { data: onWrite(q) ?? null, error: null };
        const value = resolve(q);
        return { data: Array.isArray(value) ? value[0] ?? null : value, error: null };
      },
      single: async () => b.maybeSingle(),
      then: (yes: any, no: any) => {
        record();
        if (q.op !== "select") return Promise.resolve({ data: onWrite(q) ?? null, error: null }).then(yes, no);
        const value = resolve(q);
        const data = value === null ? [] : Array.isArray(value) ? value : [value];
        return Promise.resolve({ data, error: null }).then(yes, no);
      },
    };
    return b;
  }

  const sb = {
    from: (table: string) => chain(table),
    rpc: async (name: string, args: any) => {
      rpcCalls.push({ name, args });
      const data = rpc(name, args);
      return { data: data === undefined ? null : data, error: null };
    },
  };

  return { sb, rpcCalls, writes, reads, rpcNamed: (name) => rpcCalls.filter((c) => c.name === name).map((c) => c.args) };
}

/** A pipeline_write answer: the fence applied the write. */
export const applied = { applied: true };
/** A pipeline_write answer: the run moved on, so the write was refused as stale. */
export const stale = { applied: false };
