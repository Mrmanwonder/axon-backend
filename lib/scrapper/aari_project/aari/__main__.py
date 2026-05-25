"""
AARI — CLI Entrypoint (YAML config support)
Extends main.py to accept a --config subjects.yaml file.

Usage
-----
    python -m aari --config subjects.yaml
    python -m aari --subjects "0580:Mathematics:IGCSE:CAIE" --years 2024 2025
"""

from __future__ import annotations
import argparse
import asyncio
import logging
import sys
from pathlib import Path

from aari.main import AARIPipeline, SubjectSpec
from aari.config import LOG_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "aari.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("aari")


def _load_yaml_config(path: Path) -> tuple[list[SubjectSpec], list[int]]:
    try:
        import yaml
    except ImportError:
        logger.error("PyYAML not installed. Run: pip install pyyaml")
        sys.exit(1)

    data     = yaml.safe_load(path.read_text())
    years    = data.get("target_years", [2023, 2024, 2025, 2026])
    subjects = []
    for entry in data.get("subjects", []):
        spec = SubjectSpec(
            code          = entry["code"].strip().upper(),
            name          = entry["name"].strip(),
            qualification = entry["qualification"].strip(),
            board         = entry["board"].strip(),
        )
        subjects.append(spec)
    return subjects, years


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m aari",
        description="AARI — Axon Autonomous Resource Intelligence",
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--config", type=Path, metavar="FILE",
        help="Path to a subjects.yaml config file",
    )
    group.add_argument(
        "--subjects", nargs="+", metavar="CODE:Name:Qual:Board",
        help="Inline subject specs",
    )
    p.add_argument(
        "--years", nargs="+", type=int,
        default=[2023, 2024, 2025, 2026],
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Discover URLs but skip downloading files",
    )
    return p.parse_args()


async def _run() -> None:
    args = _parse()

    if args.config:
        subjects, years = _load_yaml_config(args.config)
    else:
        subjects = [SubjectSpec.from_string(s) for s in args.subjects]
        years    = args.years

    logger.info("AARI starting — %d subjects, years %s", len(subjects), years)

    pipeline = AARIPipeline(target_years=years)

    if args.dry_run:
        logger.info("[DRY RUN] Discovery only — files will not be downloaded")
        # Monkey-patch downloader to no-op
        async def _noop(resources, **kw):
            logger.info("[DRY RUN] Would download %d files", len(resources))
            return resources
        pipeline._downloader.download_all = _noop  # type: ignore[assignment]

    await pipeline.run(subjects)


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
