ALTER TABLE capability_probe ADD COLUMN deployment_sha TEXT;
ALTER TABLE capability_probe ADD COLUMN config_revision TEXT;

CREATE INDEX IF NOT EXISTS capability_probe_release_idx
  ON capability_probe(provider, capability, deployment_sha, config_revision, probed_at DESC);
