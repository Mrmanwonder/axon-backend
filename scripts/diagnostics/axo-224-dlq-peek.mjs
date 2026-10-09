// AXO-224: non-leasing, read-only peek into content and structure DLQs.
// DO NOT log queue message bodies, refs, region IDs, paper IDs, UUIDs, token,
// prompts, student content, or R2 keys. No ACK, PULL, PURGE or SEND calls.
// Cloudflare docs: POST /queues/{queue_id}/messages/peek leaves messages untouched.
const token = process.env.CLOUDFLARE_API_TOKEN;
const account = process.env.CLOUDFLARE_ACCOUNT_ID;
if (!token || !account) throw new Error("Cloudflare metrics credentials missing");
const base = "https://api.cloudflare.com/client/v4/accounts/" + encodeURIComponent(account);
const headers = { authorization: "Bearer " + token, "content-type": "application/json" };
const response = await fetch(base + "/queues?per_page=100", {
  headers: { authorization: headers.authorization }, signal: AbortSignal.timeout(30000),
});
if (!response.ok) { console.log(JSON.stringify({phase:"list",httpStatus:response.status})); process.exit(1); }
const list = await response.json();
for (const queueName of ["content-dlq", "structure-dlq"]) {
const queue = (list.result ?? []).find(q => q.queue_name === queueName);
if (!queue?.queue_id) throw new Error("Requested DLQ absent from returned queue names");
const peek = await fetch(base + "/queues/" + encodeURIComponent(queue.queue_id) + "/messages/peek", {
  method:"POST", headers, body:JSON.stringify({batch_size:100}), signal:AbortSignal.timeout(30000),
});
if (!peek.ok) { console.log(JSON.stringify({phase:"peek",httpStatus:peek.status})); process.exit(1); }
const result = await peek.json();
if (!result.success) {
  console.log(JSON.stringify({phase:"peek",errorCodes:(result.errors ?? []).map(e=>e.code).slice(0,5)}));
  process.exit(1);
}
const messages = result.result?.messages ?? [];
const byRun = new Map();
const attempts = {};
const manualRetries = {};
let malformed = 0;
let noRun = 0;
let noWork = 0;
for (const m of messages) {
  // The body is handled in memory exclusively, never printed, stored or logged.
  let body;
  try { body = JSON.parse(m.body ?? ""); } catch { malformed++; continue; }
  if (!body || typeof body !== "object") { malformed++; continue; }
  const run = typeof body.run_id === "string" ? body.run_id : null;
  const work = typeof (queueName === "content-dlq" ? body.region_id : body.page_id) === "string"
    ? (queueName === "content-dlq" ? body.region_id : body.page_id) : null;
  if (!run) noRun++;
  if (!work) noWork++;
  if (run) byRun.set(run,(byRun.get(run)??0)+1);
  const n = Number.isInteger(m.attempts) ? m.attempts : "unknown";
  attempts[n] = (attempts[n] ?? 0) + 1;
  const retry = Number.isInteger(body._retries) ? body._retries : "unset";
  manualRetries[retry] = (manualRetries[retry] ?? 0) + 1;
}
console.log(JSON.stringify({
  phase:"peek_aggregate",queue:queueName,peeked:messages.length,malformed,noRun,noWork,
  // Do not expose individual run UUIDs. The sorted group counts are enough
  // to compare to database aggregates (17 and 24) without leaking identifiers.
  distinctRunCount:byRun.size,runMessageCounts:[...byRun.values()].sort((a,b)=>a-b),
  attempts,manualRetries,
}));

}
