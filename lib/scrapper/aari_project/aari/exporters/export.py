"""
AARI — Export Engine
Generates:
  • batch_upload.json  — Firestore-ready payload for ResourceCrawlerService
  • manifest.lock      — idempotent sync manifest (uid + sha256 + synced flag)

Both files use strict ISO-8601 timestamps and match the Axon Firestore
Security Rules (`isOwner`, `validUserDoc`) field requirements.
"""

from __future__ import annotations
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from aari.models.resource import AARIResource, CrawlSession, ManifestEntry

logger = logging.getLogger("aari.exporter")


class ExportEngine:

    def __init__(self, output_dir: Path) -> None:
        self._out = output_dir
        self._out.mkdir(parents=True, exist_ok=True)

    # ── batch_upload.json ─────────────────────────────────────────────────────

    def write_batch_upload(
        self,
        resources: list[AARIResource],
        crawl:     CrawlSession,
    ) -> Path:
        """
        Firestore batch-upload format.
        Each entry is a /resources/{uid} document ready for the
        ResourceCrawlerService to upsert via a WriteBatch.

        Shape
        -----
        {
          "meta": { session_id, generated_at, total },
          "resources": [ { uid, board, subject_code, ... }, ... ]
        }
        """
        payload = {
            "meta": {
                "session_id":    crawl.session_id,
                "generated_at":  datetime.now(timezone.utc).isoformat(),
                "total":         len(resources),
                "new":           crawl.total_new,
                "dupes":         crawl.total_dupes,
                "errors":        crawl.total_errors,
            },
            "resources": [r.to_firestore_dict() for r in resources],
        }

        dest = self._out / "batch_upload.json"
        dest.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        logger.info("Wrote batch_upload.json (%d resources) → %s", len(resources), dest)
        return dest

    # ── manifest.lock ─────────────────────────────────────────────────────────

    def write_manifest(
        self,
        resources: list[AARIResource],
        crawl:     CrawlSession,
        existing:  dict[str, ManifestEntry] | None = None,
    ) -> Path:
        """
        Idempotent manifest for incremental sync.

        • Resources already in the manifest and unchanged (same SHA-256)
          have ``synced: true`` preserved.
        • New or updated resources have ``synced: false``.

        The ResourceCrawlerService reads this file on each background sync
        and skips entries where synced == true.
        """
        existing = existing or {}
        entries: dict[str, dict] = {}

        for r in resources:
            if not r.sha256:
                continue   # not yet downloaded / validated
            prev = existing.get(r.uid)
            synced = (
                prev is not None
                and prev.sha256 == r.sha256
                and prev.synced
            )
            entry = ManifestEntry(
                uid           = r.uid,
                sha256        = r.sha256,
                file_url      = r.file_url,
                resource_type = r.resource_type,
                crawled_at    = r.crawled_at,
                synced        = synced,
            )
            entries[r.uid] = entry.model_dump()

        manifest = {
            "lock_version":  "1.0",
            "session_id":    crawl.session_id,
            "locked_at":     datetime.now(timezone.utc).isoformat(),
            "total_entries": len(entries),
            "entries":       entries,
        }

        dest = self._out / "manifest.lock"
        dest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
        logger.info("Wrote manifest.lock (%d entries) → %s", len(entries), dest)
        return dest

    # ── Load existing manifest ────────────────────────────────────────────────

    def load_manifest(self) -> dict[str, ManifestEntry]:
        path = self._out / "manifest.lock"
        if not path.exists():
            return {}
        try:
            data    = json.loads(path.read_text())
            entries = data.get("entries", {})
            return {
                uid: ManifestEntry(**entry)
                for uid, entry in entries.items()
            }
        except Exception as exc:
            logger.warning("Could not load manifest.lock: %s", exc)
            return {}
