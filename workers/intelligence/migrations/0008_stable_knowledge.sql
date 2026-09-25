PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS stable_knowledge (
  id TEXT PRIMARY KEY,
  subject TEXT NOT NULL,
  statement TEXT NOT NULL,
  revision TEXT NOT NULL,
  source_hash TEXT NOT NULL,
  source_uri TEXT,
  active INTEGER NOT NULL DEFAULT 1,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS stable_knowledge_alias (
  knowledge_id TEXT NOT NULL REFERENCES stable_knowledge(id),
  alias_norm TEXT NOT NULL,
  PRIMARY KEY(knowledge_id, alias_norm)
);
CREATE INDEX IF NOT EXISTS stable_knowledge_alias_lookup_idx ON stable_knowledge_alias(alias_norm);
