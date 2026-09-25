PRAGMA foreign_keys = ON;

-- These fields complete the Intelligence Layer v2 audit contract. They are
-- structured metadata only; raw student prompts and model answers stay out of
-- the general trace ledger.
ALTER TABLE ai_trace ADD COLUMN intent TEXT;
ALTER TABLE ai_trace ADD COLUMN verification_failures TEXT NOT NULL DEFAULT '[]';
ALTER TABLE ai_trace ADD COLUMN grounding_used INTEGER NOT NULL DEFAULT 0;
ALTER TABLE ai_trace ADD COLUMN answer_status TEXT;
