PRAGMA foreign_keys = ON;

-- One reviewed correction may improve several independent subsystems. These
-- targets contain only routing metadata; the original and accepted values stay
-- in student_correction and are never copied into aggregate learning tables.
CREATE TABLE IF NOT EXISTS active_learning_target (
  id TEXT PRIMARY KEY,
  correction_id TEXT NOT NULL REFERENCES student_correction(id),
  target TEXT NOT NULL CHECK (target IN (
    'BENCHMARK_EXPANSION',
    'CONFIDENCE_RECALIBRATION',
    'LAYOUT_TRAINING',
    'HTR_DATASET',
    'PROMPT_REGRESSION',
    'ERROR_CLUSTERING'
  )),
  status TEXT NOT NULL CHECK (status IN ('QUEUED','LABELLED','READY','EXPORTED','REJECTED')),
  created_at TEXT NOT NULL,
  updated_at TEXT,
  UNIQUE(correction_id, target)
);

CREATE INDEX IF NOT EXISTS active_learning_target_queue_idx
  ON active_learning_target(target, status, created_at);

-- Signatures are one-way hashes of bounded categorical metadata. No predicted,
-- corrected, or accepted student content is stored in this aggregate.
CREATE TABLE IF NOT EXISTS correction_error_cluster (
  signature TEXT PRIMARY KEY,
  field TEXT NOT NULL,
  reasons_json TEXT NOT NULL,
  correction_count INTEGER NOT NULL,
  high_confidence_count INTEGER NOT NULL,
  first_seen_at TEXT NOT NULL,
  last_seen_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS correction_error_cluster_priority_idx
  ON correction_error_cluster(correction_count DESC, high_confidence_count DESC, last_seen_at DESC);

CREATE TABLE IF NOT EXISTS confidence_calibration_observation (
  id TEXT PRIMARY KEY,
  correction_id TEXT NOT NULL UNIQUE REFERENCES student_correction(id),
  confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
  confidence_bucket INTEGER NOT NULL CHECK (confidence_bucket >= 0 AND confidence_bucket <= 9),
  prediction_correct INTEGER NOT NULL CHECK (prediction_correct IN (0, 1)),
  created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS confidence_calibration_bucket_idx
  ON confidence_calibration_observation(confidence_bucket, created_at);
