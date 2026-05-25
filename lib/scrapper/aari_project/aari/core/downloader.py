"""
AARI — Async File Downloader
Downloads discovered PDF URLs, runs deduplication and PDF validation,
and attaches integrity hashes to the AARIResource model.
"""

from __future__ import annotations
import asyncio
import logging
from pathlib import Path
from typing import Optional

import httpx

from aari.config import CRAWL_CONFIG
from aari.core.dedup import DeduplicationEngine
from aari.core.polite import PoliteLimiter, RobotsCache, fetch_with_retry
from aari.models.resource import AARIResource, VerifiedStatus
from aari.validators.pdf_validator import PDFValidator
from aari.validators.quality import compute_quality_score

logger = logging.getLogger("aari.downloader")


class ResourceDownloader:
    """
    Concurrently downloads AARIResource file URLs, validates each PDF,
    and enriches the resource record with sha256 + quality_score.

    Files are saved to ``output_dir / board / subject_code / filename``.
    """

    def __init__(
        self,
        output_dir: Path,
        dedup:      DeduplicationEngine,
        limiter:    PoliteLimiter,
        robots:     RobotsCache,
        validator:  PDFValidator,
    ) -> None:
        self._out       = output_dir
        self._dedup     = dedup
        self._limiter   = limiter
        self._robots    = robots
        self._validator = validator

    async def download_all(
        self,
        resources: list[AARIResource],
        concurrency: int = 5,
    ) -> list[AARIResource]:
        """Download all resources with bounded concurrency. Returns enriched list."""
        sem     = asyncio.Semaphore(concurrency)
        headers = {"User-Agent": CRAWL_CONFIG.user_agent}

        async with httpx.AsyncClient(headers=headers, timeout=60) as client:
            tasks = [
                self._download_one(r, client, sem, resources)
                for r in resources
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        final = []
        for r, res in zip(resources, results):
            if isinstance(res, Exception):
                logger.error("Download failed for %s: %s", r.file_url, res)
            elif res is not None:
                final.append(res)
        return final

    async def _download_one(
        self,
        resource:    AARIResource,
        client:      httpx.AsyncClient,
        sem:         asyncio.Semaphore,
        all_resources: list[AARIResource],
    ) -> Optional[AARIResource]:

        async with sem:
            resp = await fetch_with_retry(
                client, resource.file_url,
                self._limiter, self._robots,
                max_retries=CRAWL_CONFIG.max_retries,
            )
            if resp is None:
                return None

            raw = resp.content

            # Content-level dedup
            is_dupe, sha256 = self._dedup.check_and_register_content(raw)
            if is_dupe:
                logger.debug("Content dupe (SHA-256 match): %s", resource.file_url)
                return None

            # PDF structural validation
            vr = self._validator.validate(raw, resource.file_url)
            if not vr.is_valid:
                logger.warning("Invalid PDF [%s]: %s", resource.file_url, vr.error)
                resource.verified_status = VerifiedStatus.FLAGGED
                return resource   # still include — let operator decide

            # Attach integrity data
            resource.sha256          = sha256
            resource.file_size_bytes = vr.file_size

            # Compute quality score (uses all_resources for paired-MS/ER check)
            resource.quality_score = compute_quality_score(
                resource, all_resources, pdf_quality_sig=vr.quality_signal
            )

            # Persist to disk
            await self._save(resource, raw)

            logger.info(
                "✓ %s [%s/%s %s] Q=%d",
                resource.subject_code,
                resource.board,
                resource.year,
                resource.resource_type,
                resource.quality_score,
            )
            return resource

    async def _save(self, resource: AARIResource, raw: bytes) -> Path:
        dest_dir = (
            self._out
            / resource.board
            / resource.subject_code
            / str(resource.year)
        )
        dest_dir.mkdir(parents=True, exist_ok=True)

        filename = resource.file_url.split("/")[-1].split("?")[0] or f"{resource.uid}.pdf"
        dest     = dest_dir / filename

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, dest.write_bytes, raw)
        return dest
