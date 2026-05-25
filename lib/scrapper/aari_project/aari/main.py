"""
AARI — Main Orchestrator
Wires the full pipeline:
  Discovery → Download → Validate → Score → Dedup → Export

Usage
-----
    python -m aari.main --subjects "0580:Mathematics:IGCSE:CAIE" --years 2023 2024 2025
    python -m aari.main --config subjects.yaml
"""

from __future__ import annotations
import argparse
import asyncio
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

from aari.config import (
    CACHE_DIR, LOG_DIR, OUTPUT_DIR, CRAWL_CONFIG,
    OFFICIAL_SOURCES, COMMUNITY_SOURCES,
)
from aari.core.dedup import DeduplicationEngine
from aari.core.downloader import ResourceDownloader
from aari.core.polite import PoliteLimiter, RobotsCache
from aari.exporters.export import ExportEngine
from aari.models.resource import Board, CrawlSession, Qualification
from aari.scrapers.official import OfficialBoardScraper
from aari.scrapers.community import CommunityRepoScraper
from aari.validators.pdf_validator import PDFValidator

# ─── Logging Setup ────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "aari.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("aari.main")


# ─── Subject Spec ─────────────────────────────────────────────────────────────

@dataclass
class SubjectSpec:
    code:          str
    name:          str
    qualification: Qualification
    board:         Board

    @classmethod
    def from_string(cls, s: str) -> "SubjectSpec":
        """Parse 'CODE:Name:Qualification:Board' — e.g. '0580:Mathematics:IGCSE:CAIE'."""
        parts = s.split(":")
        if len(parts) != 4:
            raise ValueError(f"Expected CODE:Name:Qual:Board, got: {s!r}")
        code, name, qual, board = parts
        return cls(
            code          = code.strip().upper(),
            name          = name.strip(),
            qualification = Qualification(qual.strip()),
            board         = Board(board.strip()),
        )


# ─── Pipeline ─────────────────────────────────────────────────────────────────

class AARIPipeline:

    def __init__(self, target_years: list[int]) -> None:
        self._years   = target_years
        self._dedup   = DeduplicationEngine(CACHE_DIR / "dedup_cache.json")
        self._limiter = PoliteLimiter(
            CRAWL_CONFIG.request_delay_min,
            CRAWL_CONFIG.request_delay_max,
        )
        self._robots  = RobotsCache(CRAWL_CONFIG.user_agent)
        self._official  = OfficialBoardScraper(self._dedup, self._limiter, self._robots)
        self._community = CommunityRepoScraper(self._dedup, self._limiter, self._robots)
        self._downloader = ResourceDownloader(
            output_dir = OUTPUT_DIR / "files",
            dedup      = self._dedup,
            limiter    = self._limiter,
            robots     = self._robots,
            validator  = PDFValidator(),
        )
        self._exporter = ExportEngine(OUTPUT_DIR)

    async def run(self, subjects: list[SubjectSpec]) -> None:
        crawl = CrawlSession()
        crawl.sources_crawled = list(OFFICIAL_SOURCES) + list(COMMUNITY_SOURCES)
        all_resources = []

        logger.info("═" * 60)
        logger.info(" AARI Crawl Session: %s", crawl.session_id)
        logger.info(" Subjects: %d | Target years: %s", len(subjects), self._years)
        logger.info("═" * 60)

        # ── Stage 1: Discovery ─────────────────────────────────────────────
        for spec in subjects:
            logger.info("── Discovering: %s [%s/%s]", spec.name, spec.board, spec.qualification)

            # Official board sources
            board_source_key = {
                Board.CAIE:    "cambridge",
                Board.IBO:     "ibo",
                Board.Edexcel: "pearson",
            }.get(spec.board)

            if board_source_key:
                try:
                    found = await self._official.crawl_source(
                        board_source_key, spec.code, spec.name, spec.qualification
                    )
                    logger.info("  Official: %d resources found", len(found))
                    all_resources.extend(found)
                except Exception as exc:
                    logger.error("  Official crawl error: %s", exc)
                    crawl.errors.append({"source": board_source_key, "error": str(exc)})

            # Community repositories
            for key in COMMUNITY_SOURCES:
                cfg = COMMUNITY_SOURCES[key]
                if spec.board.value not in cfg["board_map"]:
                    continue
                try:
                    found = await self._community.crawl(
                        key, spec.code, spec.name, spec.board,
                        spec.qualification, self._years,
                    )
                    logger.info("  Community [%s]: %d resources found", key, len(found))
                    all_resources.extend(found)
                except Exception as exc:
                    logger.error("  Community crawl error [%s]: %s", key, exc)
                    crawl.errors.append({"source": key, "error": str(exc)})

        crawl.total_found = len(all_resources)
        logger.info("Discovery complete. Total candidates: %d", crawl.total_found)

        # ── Stage 2: Download + Validate ──────────────────────────────────
        logger.info("── Downloading and validating PDFs …")
        validated = await self._downloader.download_all(
            all_resources, concurrency=CRAWL_CONFIG.max_concurrency
        )
        crawl.total_new   = self._dedup.stats["new_items"]
        crawl.total_dupes = (
            self._dedup.stats["urls_skipped"] + self._dedup.stats["content_dupes"]
        )
        crawl.total_errors = len(crawl.errors)
        logger.info(
            "Download complete — new: %d  dupes: %d  errors: %d",
            crawl.total_new, crawl.total_dupes, crawl.total_errors,
        )

        # ── Stage 3: Export ───────────────────────────────────────────────
        crawl.close()
        existing_manifest = self._exporter.load_manifest()
        self._exporter.write_batch_upload(validated, crawl)
        self._exporter.write_manifest(validated, crawl, existing_manifest)
        self._dedup.save()

        logger.info("═" * 60)
        logger.info(" Session complete: %s", crawl.session_id)
        logger.info(" batch_upload.json + manifest.lock written to: %s", OUTPUT_DIR)
        logger.info(" Dedup cache saved to: %s", CACHE_DIR / "dedup_cache.json")
        logger.info("═" * 60)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AARI — Axon Autonomous Resource Intelligence",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "--subjects", nargs="+",
        metavar="CODE:Name:Qual:Board",
        default=["0580:Mathematics:IGCSE:CAIE"],
        help="Subject specs. Example: 0580:Mathematics:IGCSE:CAIE",
    )
    p.add_argument(
        "--years", nargs="+", type=int,
        default=[2023, 2024, 2025, 2026],
        help="Target years to collect (default: 2023–2026)",
    )
    return p.parse_args()


async def _main() -> None:
    args     = _parse_args()
    subjects = [SubjectSpec.from_string(s) for s in args.subjects]
    pipeline = AARIPipeline(target_years=args.years)
    await pipeline.run(subjects)


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()
