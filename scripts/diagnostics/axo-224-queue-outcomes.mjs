// AXO-224 forensic probe. Read-only Cloudflare queue operations and backlog.
// Output only aggregate dimensions/retention/configuration. Never read, pull,
// peek, acknowledge, purge, or log any message body or student data.
const token = process.env.CLOUDFLARE_API_TOKEN;
const account = process.env.CLOUDFLARE_ACCOUNT_ID;
if (!token || !account) throw new Error("Queue diagnosis credentials unavailable");
const api = "https://api.cloudflare.com/client/v4";
const headers = { authorization: "Bearer " + token };
const wanted = new Set(["content-queue","content-dlq","structure-queue","structure-dlq","triage-queue","triage-dlq","crop-queue","crop-dlq","reconcile-queue","reconcile-dlq"]);

async function get(path) {
  const response = await fetch(api + path, { headers, signal: AbortSignal.timeout(30000) });
  if (!response.ok) return { httpStatus: response.status };
  const json = await response.json();
  return json.success ? { result: json.result } : { errorCodes: (json.errors ?? []).map(e => e.code).slice(0,5) };
}
const found = await get("/accounts/" + encodeURIComponent(account) + "/queues?per_page=100");
if (!Array.isArray(found.result)) {
  console.log(JSON.stringify({ phase: "queue_list", httpStatus: found.httpStatus, errorCodes: found.errorCodes ?? [] }));
  process.exitCode = 1;
} else {
  for (const queue of found.result.filter(q => wanted.has(q.queue_name))) {
    const name = queue.queue_name;
    const metrics = await get("/accounts/" + encodeURIComponent(account) + "/queues/" + encodeURIComponent(queue.queue_id) + "/metrics");
    const consumer = (queue.consumers ?? []).find(c => c.type === "worker");
    console.log(JSON.stringify({
      queue: name,
      phase: "current",
      paused: queue.settings?.delivery_paused ?? null,
      retentionSeconds: queue.settings?.message_retention_period ?? null,
      batchSize: consumer?.settings?.batch_size ?? null,
      maxConcurrency: consumer?.settings?.max_concurrency ?? null,
      maxRetries: consumer?.settings?.max_retries ?? null,
      deadLetterQueue: consumer?.dead_letter_queue ?? null,
      backlogCount: metrics.result?.backlog_count ?? null,
      oldestMessageAt: Number.isFinite(metrics.result?.oldest_message_timestamp_ms) && metrics.result.oldest_message_timestamp_ms > 0
        ? new Date(metrics.result.oldest_message_timestamp_ms).toISOString() : null,
      metricsHttpStatus: metrics.httpStatus ?? null,
    }));

    // Read-only verification after AXO-224 containment merge/deploy.
    // A green backend workflow does not prove the live Queue consumer adopted
    // its new Wrangler settings; verify the Cloudflare effective configuration.
    if (name === "content-queue" || name === "structure-queue") {
      const actualBatch = consumer?.settings?.batch_size ?? null;
      const actualMaxConcurrency = consumer?.settings?.max_concurrency ?? null;
      const applied = actualBatch === 2 && actualMaxConcurrency === 6;
      console.log(JSON.stringify({queue:name,phase:"effective_config_check",
        expectedBatch:2,actualBatch,expectedMaxConcurrency:6,actualMaxConcurrency,applied}));
      if (!applied) process.exitCode = 1;
    }

    // Docs: https://developers.cloudflare.com/queues/observability/metrics/
    // One day only; filter down to incident window after fetching aggregates.
    const query = `query($accountTag:string!,$queueId:string!,$datetimeStart:Date!,$datetimeEnd:Date!){
      viewer { accounts(filter:{accountTag:$accountTag}) {
        queueMessageOperationsAdaptiveGroups(limit:1000,
          filter:{queueId:$queueId,datetime_geq:$datetimeStart,datetime_leq:$datetimeEnd},
          orderBy:[datetimeMinute_ASC]){
          count
          sum { bytes }
          dimensions { datetimeMinute actionType outcome }
        }
      }}
    }`;
    const response = await fetch(api + "/graphql", {
      method: "POST", headers: { ...headers, "content-type":"application/json" },
      body: JSON.stringify({query, variables:{accountTag:account,queueId:queue.queue_id,
        datetimeStart:"2026-10-09",datetimeEnd:"2026-10-10"}}),
      signal: AbortSignal.timeout(30000),
    });
    if (!response.ok) {
      console.log(JSON.stringify({queue:name,phase:"operations",httpStatus:response.status}));
      continue;
    }
    const result = await response.json();
    if (result.errors?.length) {
      // Never log raw GraphQL errors, which can contain query/header arguments.
      console.log(JSON.stringify({queue:name,phase:"operations",errorCodes:result.errors.map(e=>e.extensions?.code ?? "graphql_error").slice(0,5)}));
      continue;
    }
    const rows = result.data?.viewer?.accounts?.[0]?.queueMessageOperationsAdaptiveGroups;
    if (!Array.isArray(rows)) {
      console.log(JSON.stringify({queue:name,phase:"operations",responseShape:"unavailable"}));
      continue;
    }
    for (const row of rows) {
      const at = String(row.dimensions?.datetimeMinute ?? "");
      if (at >= "2026-10-09T11:00" && at <= "2026-10-09T11:55") {
        console.log(JSON.stringify({queue:name,phase:"operations",at,action:row.dimensions?.actionType,
          outcome:row.dimensions?.outcome ?? null,count:row.count,bytes:row.sum?.bytes ?? null}));
      }
    }
    console.log(JSON.stringify({queue:name,phase:"operations_total_rows",count:rows.length}));
  }
}
