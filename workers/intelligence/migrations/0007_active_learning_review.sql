PRAGMA foreign_keys = ON;

ALTER TABLE active_learning_queue ADD COLUMN reviewer TEXT;
ALTER TABLE active_learning_queue ADD COLUMN evidence_uri TEXT;
ALTER TABLE active_learning_queue ADD COLUMN eval_run_id TEXT REFERENCES eval_run(id);
ALTER TABLE active_learning_queue ADD COLUMN updated_at TEXT;
