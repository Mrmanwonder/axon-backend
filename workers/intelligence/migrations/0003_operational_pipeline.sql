PRAGMA foreign_keys = ON;

ALTER TABLE paper_page ADD COLUMN review_reasons_json TEXT NOT NULL DEFAULT '[]';
ALTER TABLE paper_page ADD COLUMN conditioned_hash TEXT;
ALTER TABLE paper_page ADD COLUMN conditioned_object_key TEXT;
ALTER TABLE paper_page ADD COLUMN pipeline_version TEXT;

CREATE TABLE IF NOT EXISTS paper_stage_event (
  id TEXT PRIMARY KEY,
  page_id TEXT NOT NULL REFERENCES paper_page(page_id),
  stage TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('STARTED','COMPLETED','FAILED','REVIEW_REQUIRED')),
  input_hash TEXT NOT NULL,
  output_hash TEXT,
  details_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS paper_stage_event_page_idx ON paper_stage_event(page_id, created_at);

CREATE TABLE IF NOT EXISTS provider_observation (
  id TEXT PRIMARY KEY,
  provider TEXT NOT NULL,
  model TEXT NOT NULL,
  success INTEGER NOT NULL,
  timeout INTEGER NOT NULL DEFAULT 0,
  rate_limited INTEGER NOT NULL DEFAULT 0,
  server_error INTEGER NOT NULL DEFAULT 0,
  schema_failure INTEGER NOT NULL DEFAULT 0,
  semantic_failure INTEGER NOT NULL DEFAULT 0,
  latency_ms INTEGER NOT NULL,
  created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS provider_observation_recent_idx ON provider_observation(provider, model, created_at DESC);

CREATE TABLE IF NOT EXISTS shadow_result (
  id TEXT PRIMARY KEY,
  trace_id TEXT NOT NULL,
  candidate_config_revision TEXT NOT NULL,
  candidate_model TEXT NOT NULL,
  verification_status TEXT NOT NULL,
  metrics_json TEXT NOT NULL,
  output_hash TEXT,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS capability_probe (
  id TEXT PRIMARY KEY,
  provider TEXT NOT NULL,
  model TEXT NOT NULL,
  capability TEXT NOT NULL,
  passed INTEGER NOT NULL,
  details_json TEXT NOT NULL,
  probed_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS capability_probe_latest_idx ON capability_probe(provider, model, capability, probed_at DESC);
