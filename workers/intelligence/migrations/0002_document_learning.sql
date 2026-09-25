PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS layout_region (
  id TEXT PRIMARY KEY, page_id TEXT NOT NULL REFERENCES paper_page(page_id), class TEXT NOT NULL,
  box_json TEXT NOT NULL, confidence REAL NOT NULL, ink_class TEXT, text_value TEXT, trust_state TEXT NOT NULL DEFAULT 'UNVERIFIED'
);
CREATE TABLE IF NOT EXISTS question_node (
  id TEXT PRIMARY KEY, paper_id TEXT NOT NULL, label TEXT NOT NULL, parent_id TEXT REFERENCES question_node(id),
  page_ids_json TEXT NOT NULL, reading_order INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS question_region (
  question_id TEXT NOT NULL REFERENCES question_node(id), region_id TEXT NOT NULL REFERENCES layout_region(id),
  relationship TEXT NOT NULL, PRIMARY KEY(question_id, region_id)
);
CREATE TABLE IF NOT EXISTS mark_assignment (
  mark_region_id TEXT NOT NULL REFERENCES layout_region(id), question_id TEXT REFERENCES question_node(id),
  confidence REAL NOT NULL, second_best_gap REAL NOT NULL, features_json TEXT NOT NULL,
  trust_state TEXT NOT NULL DEFAULT 'UNVERIFIED', PRIMARY KEY(mark_region_id)
);
CREATE TABLE IF NOT EXISTS recognition_read (
  id TEXT PRIMARY KEY, region_id TEXT NOT NULL REFERENCES layout_region(id), value_text TEXT,
  alternatives_json TEXT NOT NULL, status TEXT NOT NULL, reader_ids_json TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS trusted_field (
  id TEXT PRIMARY KEY, artifact_id TEXT NOT NULL, field_name TEXT NOT NULL, value_json TEXT,
  trust_state TEXT NOT NULL CHECK (trust_state IN ('AUTO_VERIFIED','STUDENT_VERIFIED','UNVERIFIED','UNKNOWN')),
  evidence_ids_json TEXT NOT NULL, updated_at TEXT NOT NULL, UNIQUE(artifact_id, field_name)
);
CREATE TABLE IF NOT EXISTS active_learning_queue (
  id TEXT PRIMARY KEY, correction_id TEXT NOT NULL REFERENCES student_correction(id), priority REAL NOT NULL,
  reasons_json TEXT NOT NULL, status TEXT NOT NULL CHECK (status IN ('QUEUED','LABELLED','PROMOTED','REJECTED')),
  created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS active_learning_priority_idx ON active_learning_queue(status, priority DESC);
CREATE TABLE IF NOT EXISTS concept_taxonomy (
  id TEXT PRIMARY KEY, subject TEXT NOT NULL, parent_id TEXT REFERENCES concept_taxonomy(id),
  aliases_json TEXT NOT NULL, curricula_json TEXT NOT NULL, revision TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS insight_observation (
  id TEXT PRIMARY KEY, student_id TEXT NOT NULL, concept_id TEXT NOT NULL REFERENCES concept_taxonomy(id),
  paper_id TEXT NOT NULL, correct INTEGER NOT NULL, confidence REAL NOT NULL, trust_state TEXT NOT NULL,
  evidence_ids_json TEXT NOT NULL, created_at TEXT NOT NULL
);
