PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS ai_config_revision (
  revision_id TEXT PRIMARY KEY, parent_revision_id TEXT REFERENCES ai_config_revision(revision_id),
  git_sha TEXT NOT NULL, author TEXT NOT NULL, reason TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS ai_deployment (
  deployment_sha TEXT PRIMARY KEY, config_revision TEXT NOT NULL REFERENCES ai_config_revision(revision_id),
  pipeline_version TEXT NOT NULL, rollout_percent INTEGER NOT NULL CHECK (rollout_percent BETWEEN 0 AND 100),
  state TEXT NOT NULL CHECK (state IN ('BENCHMARK','INTERNAL','CANARY','ACTIVE','HALTED','ROLLED_BACK')), created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS prompt_artifact (
  prompt_id TEXT NOT NULL, prompt_hash TEXT NOT NULL, schema_id TEXT NOT NULL, schema_hash TEXT NOT NULL,
  deployment_sha TEXT NOT NULL, created_at TEXT NOT NULL, PRIMARY KEY (prompt_id, prompt_hash)
);
CREATE TABLE IF NOT EXISTS schema_artifact (
  schema_id TEXT NOT NULL, schema_hash TEXT NOT NULL, schema_json TEXT NOT NULL, created_at TEXT NOT NULL,
  PRIMARY KEY (schema_id, schema_hash)
);
CREATE TABLE IF NOT EXISTS ai_route (
  id TEXT PRIMARY KEY, capability TEXT NOT NULL, risk TEXT NOT NULL, privacy_policy TEXT NOT NULL,
  provider TEXT NOT NULL, model TEXT NOT NULL, thinking_level TEXT NOT NULL CHECK (thinking_level IN ('minimal','low','medium','high')),
  prompt_id TEXT NOT NULL, timeout_ms INTEGER NOT NULL, enabled INTEGER NOT NULL DEFAULT 1,
  config_revision TEXT NOT NULL REFERENCES ai_config_revision(revision_id)
);
CREATE TABLE IF NOT EXISTS ai_trace (
  trace_id TEXT PRIMARY KEY, paper_id TEXT, question_id TEXT, stage TEXT NOT NULL, capability TEXT NOT NULL,
  deployment_sha TEXT NOT NULL, config_revision TEXT NOT NULL, pipeline_version TEXT NOT NULL,
  provider TEXT, requested_model TEXT, served_model TEXT, thinking_level TEXT, prompt_id TEXT, prompt_hash TEXT,
  schema_id TEXT, schema_hash TEXT, input_artifact_hashes TEXT NOT NULL DEFAULT '[]', tool_calls TEXT NOT NULL DEFAULT '[]',
  retrieval_used INTEGER NOT NULL DEFAULT 0, verification_status TEXT NOT NULL, repair_attempted INTEGER NOT NULL DEFAULT 0,
  confidence REAL, latency_ms INTEGER, input_tokens INTEGER, output_tokens INTEGER, estimated_cost REAL, error TEXT,
  transport_success INTEGER, schema_success INTEGER, semantic_validation_success INTEGER, benchmark_correctness INTEGER,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS ai_trace_created_idx ON ai_trace(created_at);
CREATE TABLE IF NOT EXISTS evidence (
  id TEXT PRIMARY KEY, trace_id TEXT NOT NULL REFERENCES ai_trace(trace_id), information_class TEXT NOT NULL,
  source TEXT NOT NULL, authority TEXT NOT NULL, value_json TEXT NOT NULL, provenance_json TEXT NOT NULL,
  verification TEXT NOT NULL, confidence REAL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS claim (
  id TEXT PRIMARY KEY, trace_id TEXT NOT NULL REFERENCES ai_trace(trace_id), text TEXT NOT NULL, type TEXT NOT NULL,
  risk TEXT NOT NULL, verification_status TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS claim_evidence (
  claim_id TEXT NOT NULL REFERENCES claim(id), evidence_id TEXT NOT NULL REFERENCES evidence(id), PRIMARY KEY (claim_id, evidence_id)
);
CREATE TABLE IF NOT EXISTS paper_page (
  page_id TEXT PRIMARY KEY, paper_id TEXT NOT NULL, student_id TEXT NOT NULL, original_hash TEXT NOT NULL,
  perceptual_hash TEXT, source_type TEXT NOT NULL, object_key TEXT NOT NULL, processing_state TEXT NOT NULL,
  quality_class TEXT, quality_json TEXT, created_at TEXT NOT NULL, updated_at TEXT,
  UNIQUE(student_id, original_hash)
);
CREATE TABLE IF NOT EXISTS student_correction (
  id TEXT PRIMARY KEY, field TEXT NOT NULL, predicted_json TEXT NOT NULL, corrected_json TEXT NOT NULL,
  accepted_json TEXT NOT NULL, artifact_id TEXT NOT NULL, pipeline_version TEXT NOT NULL, model TEXT NOT NULL,
  prompt_hash TEXT NOT NULL, context_metadata_json TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS provider_health (
  provider TEXT NOT NULL, model TEXT NOT NULL, state TEXT NOT NULL CHECK (state IN ('CLOSED','HALF_OPEN','OPEN')),
  timeout_rate REAL NOT NULL, rate_limit_rate REAL NOT NULL, server_error_rate REAL NOT NULL,
  schema_failure_rate REAL NOT NULL, semantic_failure_rate REAL NOT NULL, p95_latency_ms INTEGER NOT NULL,
  updated_at TEXT NOT NULL, PRIMARY KEY(provider, model)
);
CREATE TABLE IF NOT EXISTS eval_suite (id TEXT PRIMARY KEY, name TEXT NOT NULL, version TEXT NOT NULL, category TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS eval_case (id TEXT PRIMARY KEY, suite_id TEXT NOT NULL REFERENCES eval_suite(id), input_json TEXT NOT NULL, expected_json TEXT NOT NULL, risk TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS eval_run (id TEXT PRIMARY KEY, suite_id TEXT NOT NULL REFERENCES eval_suite(id), baseline_config TEXT, candidate_config TEXT NOT NULL, deployment_sha TEXT NOT NULL, started_at TEXT NOT NULL, completed_at TEXT, passed INTEGER);
CREATE TABLE IF NOT EXISTS eval_result (id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES eval_run(id), case_id TEXT NOT NULL REFERENCES eval_case(id), output_json TEXT, metrics_json TEXT NOT NULL, passed INTEGER NOT NULL);
