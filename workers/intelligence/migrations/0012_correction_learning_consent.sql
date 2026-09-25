PRAGMA foreign_keys = ON;

-- Corrections remain part of the student's operational record regardless of
-- optional product-improvement consent. New correction writes attach only a
-- keyed pseudonym and the authoritative consent decision observed at receipt.
ALTER TABLE student_correction ADD COLUMN student_id TEXT;
ALTER TABLE student_correction ADD COLUMN learning_consent_granted INTEGER NOT NULL DEFAULT 0
  CHECK (learning_consent_granted IN (0, 1));
ALTER TABLE student_correction ADD COLUMN learning_consent_seq INTEGER;
ALTER TABLE student_correction ADD COLUMN learning_consent_notice_version TEXT;

CREATE INDEX IF NOT EXISTS student_correction_student_created_idx
  ON student_correction(student_id, created_at);
