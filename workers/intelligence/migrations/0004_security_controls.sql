PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS api_rate_window (
  identity_hash TEXT NOT NULL,
  route TEXT NOT NULL,
  window_start INTEGER NOT NULL,
  request_count INTEGER NOT NULL,
  PRIMARY KEY(identity_hash, route, window_start)
);

CREATE TABLE IF NOT EXISTS request_idempotency (
  idempotency_key_hash TEXT NOT NULL,
  route TEXT NOT NULL,
  request_hash TEXT NOT NULL,
  status_code INTEGER NOT NULL,
  response_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  expires_at TEXT NOT NULL,
  PRIMARY KEY(idempotency_key_hash, route)
);
