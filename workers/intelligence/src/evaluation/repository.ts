import type { CaseResult, EvalCase, EvalRunSummary } from "./types";

export async function persistEvalSuite(db: D1Database, values: { id: string; name: string; version: string; category: string; cases: readonly EvalCase[] }): Promise<void> {
  const createdAt = new Date().toISOString();
  const statements: D1PreparedStatement[] = [
    db.prepare("INSERT OR IGNORE INTO eval_suite (id, name, version, category, created_at) VALUES (?, ?, ?, ?, ?)").bind(values.id, values.name, values.version, values.category, createdAt)
  ];
  for (const item of values.cases) statements.push(db.prepare("INSERT OR IGNORE INTO eval_case (id, suite_id, input_json, expected_json, risk) VALUES (?, ?, ?, ?, ?)")
    .bind(item.id, values.id, JSON.stringify(item.input), JSON.stringify(item.expected), item.risk));
  await db.batch(statements);
}

export async function persistEvalRun(db: D1Database, values: {
  suiteId: string; baselineConfig?: string; candidateConfig: string; deploymentSha: string;
  startedAt: string; summary: EvalRunSummary; results: readonly CaseResult[];
}): Promise<string> {
  const runId = crypto.randomUUID();
  const statements: D1PreparedStatement[] = [
    db.prepare("INSERT INTO eval_run (id, suite_id, baseline_config, candidate_config, deployment_sha, started_at, completed_at, passed) VALUES (?, ?, ?, ?, ?, ?, ?, ?)")
      .bind(runId, values.suiteId, values.baselineConfig ?? null, values.candidateConfig, values.deploymentSha, values.startedAt, new Date().toISOString(), values.summary.releaseAllowed ? 1 : 0)
  ];
  for (const result of values.results) statements.push(db.prepare("INSERT INTO eval_result (id, run_id, case_id, output_json, metrics_json, passed) VALUES (?, ?, ?, NULL, ?, ?)")
    .bind(crypto.randomUUID(), runId, result.caseId, JSON.stringify({ ...result.metrics, latencyMs: result.latencyMs, failures: result.failures }), result.passed ? 1 : 0));
  await db.batch(statements);
  return runId;
}
