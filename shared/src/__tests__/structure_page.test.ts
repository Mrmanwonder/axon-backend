import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createClient } from '@supabase/supabase-js';
import { loadStructurePage } from '../structure-page.js';
import { ConfigurationError, RetryableError } from '../errors.js';
import { processQueueMessage } from '../worker.js';
import type { Env } from '../env.js';

// Persisted schema from the incident. A nonexistent select field must reject
// the entire request, just as PostgREST does, rather than returning a fake page.
const persisted = new Set('id paper_id student_id page_number r2_bucket r2_key mask_key structure_status layer_fallback teacher_marks conditioning_meta quality_signals'.split(' '));
function client(response: unknown, status = 200) {
  return createClient('https://database.example', 'test-key', {
    auth: { persistSession: false },
    global: { fetch: async (input) => {
      const url = new URL(String(input));
      if (url.pathname.endsWith('/rpc/run_heartbeat')) return Response.json(null);
      const columns = url.searchParams.get('select')?.split(',') ?? [];
      assert.ok(columns.length);
      const absent = columns.find(column => !persisted.has(column.trim()));
      return Response.json(absent ? { code: '42703', message: `column ${absent} does not exist` } : response,
        { status: absent ? 400 : status });
    } },
  });
}

test('persisted pending page loads without selecting device-only margin_band', async () => {
  const page = { id: 'page-1', structure_status: 'pending', page_number: 1 };
  assert.deepEqual(await loadStructurePage(client([page]), 'page-1'), page);
});

test('genuinely deleted page is distinguishable from a database error', async () => {
  assert.equal(await loadStructurePage(client([]), 'page-1'), null);
});

test('schema rejection surfaces as configuration failure instead of no such page', async () => {
  await assert.rejects(loadStructurePage(client({ code: '42703', message: 'missing column' }, 400), 'page-1'), ConfigurationError);
});

test('transient read failure remains retryable', async () => {
  await assert.rejects(loadStructurePage(client({ code: '57014', message: 'query canceled' }, 503), 'page-1'), RetryableError);
});

test('queue retains work when page lookup fails and no replacement is durable', async () => {
  let acknowledged = false;
  let retried = false;
  await processQueueMessage(
    { body: { run_id: 'run-1' }, attempts: 1, ack: () => { acknowledged = true; }, retry: () => { retried = true; } },
    client({ code: '57014', message: 'query canceled' }, 503),
    {} as Env,
    async ({ sb }) => { await loadStructurePage(sb, 'page-1'); },
  );
  assert.equal(acknowledged, false);
  assert.equal(retried, true);
});

test('schema failure is acknowledged only after durable failure recording', async () => {
  const events: string[] = [];
  await processQueueMessage(
    { body: { run_id: 'run-1' }, attempts: 1, ack: () => { events.push('ack'); }, retry: () => { events.push('retry'); } },
    client({ code: '42703', message: 'missing column' }, 400),
    {} as Env,
    async ({ sb }) => { await loadStructurePage(sb, 'page-1'); },
    async (_, error) => { assert.ok(error instanceof ConfigurationError); events.push('recorded'); },
  );
  assert.deepEqual(events, ['recorded', 'ack']);
});
