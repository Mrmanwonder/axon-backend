"""
AARI — Deduplication Engine
SHA-256 content-addressed dedup with a persistent bloom-filter-backed
seen-set so large crawls don't re-download known files.
"""

from __future__ import annotations
import hashlib
import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("aari.dedup")


class DeduplicationEngine:
    """
    Two-level deduplication:
      1. URL-level  — skip if exact URL was already crawled this session.
      2. Content-level — skip if SHA-256 matches a previously ingested file
                         (catches mirrors and re-uploads of the same content).

    The seen-set is persisted to ``cache_path`` between runs, enabling
    idempotent incremental crawls.
    """

    def __init__(self, cache_path: Path) -> None:
        self._cache_path = cache_path
        self._url_seen:     set[str] = set()
        self._sha256_seen:  set[str] = set()
        self._stats = {"urls_skipped": 0, "content_dupes": 0, "new_items": 0}
        self._load()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self) -> None:
        if self._cache_path.exists():
            try:
                data = json.loads(self._cache_path.read_text())
                self._url_seen    = set(data.get("urls", []))
                self._sha256_seen = set(data.get("sha256s", []))
                logger.info(
                    "DeduplicationEngine loaded %d URLs, %d hashes from cache",
                    len(self._url_seen), len(self._sha256_seen),
                )
            except Exception as exc:
                logger.warning("Could not load dedup cache: %s", exc)

    def save(self) -> None:
        """Persist the seen-sets to disk. Call at end of crawl."""
        try:
            self._cache_path.write_text(json.dumps({
                "urls":    list(self._url_seen),
                "sha256s": list(self._sha256_seen),
            }, indent=2))
        except Exception as exc:
            logger.error("Could not save dedup cache: %s", exc)

    # ── URL-level ─────────────────────────────────────────────────────────────

    def is_url_seen(self, url: str) -> bool:
        return url in self._url_seen

    def mark_url_seen(self, url: str) -> None:
        self._url_seen.add(url)

    # ── Content-level ─────────────────────────────────────────────────────────

    @staticmethod
    def compute_sha256(raw_bytes: bytes) -> str:
        return hashlib.sha256(raw_bytes).hexdigest()

    def is_content_seen(self, sha256: str) -> bool:
        return sha256 in self._sha256_seen

    def mark_content_seen(self, sha256: str) -> None:
        self._sha256_seen.add(sha256)

    # ── Combined check ────────────────────────────────────────────────────────

    def check_and_register_url(self, url: str) -> bool:
        """
        Returns True if the URL is a *duplicate* (should be skipped).
        Registers it as seen on first encounter.
        """
        if self.is_url_seen(url):
            self._stats["urls_skipped"] += 1
            return True
        self.mark_url_seen(url)
        return False

    def check_and_register_content(self, raw_bytes: bytes) -> tuple[bool, str]:
        """
        Hashes ``raw_bytes`` and checks against seen digests.
        Returns (is_duplicate, sha256_hex).
        """
        digest = self.compute_sha256(raw_bytes)
        if self.is_content_seen(digest):
            self._stats["content_dupes"] += 1
            return True, digest
        self.mark_content_seen(digest)
        self._stats["new_items"] += 1
        return False, digest

    @property
    def stats(self) -> dict:
        return {**self._stats, "total_seen_urls": len(self._url_seen),
                "total_seen_hashes": len(self._sha256_seen)}
