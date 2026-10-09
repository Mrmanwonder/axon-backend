// Read-only, bounded operational metrics. Never print headers, bindings, tokens or paper data.
const token = process.env.CLOUDFLARE_API_TOKEN;
const account = process.env.CLOUDFLARE_ACCOUNT_ID;
if (!token || !account) throw new Error("Cloudflare diagnostics credentials unavailable");
const query = `query GetWorkersAnalytics($accountTag: string, $datetimeStart: string, $datetimeEnd: string, $scriptName: string) {
  viewer { accounts(filter: {accountTag: $accountTag}) {
    workersInvocationsAdaptive(limit: 100, filter: {scriptName: $scriptName, datetime_geq: $datetimeStart, datetime_leq: $datetimeEnd}) {
      sum { subrequests requests errors }
      quantiles { cpuTimeP50 cpuTimeP99 }
      dimensions { datetime scriptName status }
    }
  }}
}`;
for (const scriptName of ["mastery-triage","mastery-structure","mastery-crop","mastery-content"]) {
  const response = await fetch("https://api.cloudflare.com/client/v4/graphql", {
    method: "POST", headers: { authorization: "Bearer " + token, "content-type": "application/json" },
    body: JSON.stringify({ query, variables: { accountTag: account, datetimeStart: "2026-10-09T11:00:00Z", datetimeEnd: "2026-10-09T11:25:00Z", scriptName } }),
    signal: AbortSignal.timeout(30000),
  });
  if (!response.ok) { console.log(JSON.stringify({scriptName, httpStatus: response.status})); process.exitCode = 1; continue; }
  const result = await response.json();
  if (result.errors?.length) {
    // Provider errors may echo inputs: report only codes, never the raw error.
    console.log(JSON.stringify({scriptName, errorCodes: result.errors.map(e => e.extensions?.code ?? "graphql_error")}));
    process.exitCode = 1; continue;
  }
  const rows = result.data?.viewer?.accounts?.[0]?.workersInvocationsAdaptive;
  if (!Array.isArray(rows)) throw new Error("Unexpected metrics response shape");
  console.log(JSON.stringify({scriptName, rows: rows.map(r => ({
    at: r.dimensions?.datetime, status: r.dimensions?.status,
    requests: r.sum?.requests, errors: r.sum?.errors, subrequests: r.sum?.subrequests,
    cpuP50: r.quantiles?.cpuTimeP50, cpuP99: r.quantiles?.cpuTimeP99,
  }))}));
}
