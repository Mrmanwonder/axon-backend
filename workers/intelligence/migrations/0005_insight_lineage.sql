PRAGMA foreign_keys = ON;

ALTER TABLE insight_observation ADD COLUMN trusted_field_id TEXT REFERENCES trusted_field(id);
CREATE UNIQUE INDEX IF NOT EXISTS insight_observation_field_idx ON insight_observation(trusted_field_id) WHERE trusted_field_id IS NOT NULL;
