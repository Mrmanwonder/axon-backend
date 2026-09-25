PRAGMA foreign_keys = ON;

ALTER TABLE paper_page ADD COLUMN page_index INTEGER NOT NULL DEFAULT 0;
CREATE INDEX IF NOT EXISTS paper_page_order_idx ON paper_page(paper_id, page_index, created_at);
CREATE UNIQUE INDEX IF NOT EXISTS paper_page_unique_order_idx ON paper_page(paper_id, page_index);
