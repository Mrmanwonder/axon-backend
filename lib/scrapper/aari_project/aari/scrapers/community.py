"""
AARI — Community Repository Scraper
Crawls community hubs (PapaCambridge, GCE Guide) that publicly list
past-paper PDFs and respect scraper traffic (verified via robots.txt).

Note: cloudscraper / undetected-chromedriver are NOT used here.
If a site actively blocks crawlers, that is a ToS signal to respect — not
a technical challenge to solve. Resources from such sites are excluded.
"""

from __future__ import annotations
import asyncio
import logging
import re
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from aari.config import COMMUNITY_SOURCES, CRAWL_CONFIG
from aari.core.polite import PoliteLimiter, RobotsCache, fetch_with_retry
from aari.core.healer import self_heal
from aari.core.dedup import DeduplicationEngine
from aari.models.resource import (
    AARIResource, Board, Qualification, ResourceType,
    Session, SourceType, VerifiedStatus,
)

logger = logging.getLogger("aari.community_scraper")


_YEAR_RE    = re.compile(r"\b(20\d{2})\b")
_SESSION_MAP = {
    "feb": Session.FEB_MAR, "mar": Session.FEB_MAR,
    "may": Session.MAY_JUN, "jun": Session.MAY_JUN,
    "oct": Session.OCT_NOV, "nov": Session.OCT_NOV,
}


def _guess_session(text: str) -> Optional[Session]:
    for token, sess in _SESSION_MAP.items():
        if token in text.lower():
            return sess
    return None


def _guess_rtype(text: str) -> ResourceType:
    t = text.lower()
    if "mark" in t or " ms" in t:
        return ResourceType.MARK_SCHEME
    if "examiner" in t or " er" in t:
        return ResourceType.EXAMINER_REPORT
    if "syllabus" in t or "specification" in t:
        return ResourceType.SYLLABUS
    return ResourceType.PAST_PAPER


class CommunityRepoScraper:
    """
    Scrapes community repositories for subject-specific past paper links.

    Only targets sites confirmed to allow indexing in their robots.txt.
    Falls back to self-healing if a URL pattern has changed.
    """

    def __init__(
        self,
        dedup:   DeduplicationEngine,
        limiter: PoliteLimiter,
        robots:  RobotsCache,
    ) -> None:
        self._dedup   = dedup
        self._limiter = limiter
        self._robots  = robots

    async def crawl(
        self,
        source_key:    str,
        subject_code:  str,
        subject_name:  str,
        board:         Board,
        qualification: Qualification,
        target_years:  list[int],
    ) -> list[AARIResource]:

        cfg = COMMUNITY_SOURCES.get(source_key)
        if not cfg:
            raise ValueError(f"Unknown community source: {source_key}")

        board_path = cfg["board_map"].get(board.value, "/")
        start_url  = cfg["base_url"] + board_path + subject_code.lower() + "/"
        results    = []

        headers = {"User-Agent": CRAWL_CONFIG.user_agent}

        async with httpx.AsyncClient(headers=headers, timeout=30) as client:
            resp = await fetch_with_retry(
                client, start_url, self._limiter, self._robots
            )

            # Self-heal if URL pattern has changed
            if resp is None:
                logger.info("[Healer] Attempting self-heal for %s", start_url)
                healed = await self_heal(
                    start_url, f"{subject_code} {subject_name}", client
                )
                if healed:
                    logger.info("[Healer] Resolved to %s", healed)
                    resp = await fetch_with_retry(
                        client, healed, self._limiter, self._robots
                    )

            if resp is None:
                logger.warning("Could not access %s — skipping", start_url)
                return []

            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup.find_all("a", href=True):
                href = urljoin(start_url, tag["href"])
                text = tag.get_text(strip=True)

                if not href.lower().endswith(".pdf"):
                    continue
                if self._dedup.check_and_register_url(href):
                    continue

                year_m = _YEAR_RE.search(href + " " + text)
                year   = int(year_m.group(1)) if year_m else datetime.now().year

                if year not in target_years:
                    continue

                resource = AARIResource(
                    board          = board,
                    qualification  = qualification,
                    subject_code   = subject_code,
                    subject_name   = subject_name,
                    resource_type  = _guess_rtype(text + " " + href),
                    year           = year,
                    session        = _guess_session(href + " " + text),
                    source_type    = SourceType.COMMUNITY,
                    source_url     = resp.url if hasattr(resp, "url") else start_url,
                    file_url       = href,
                    verified_status= VerifiedStatus.UNVERIFIED,
                    crawled_at     = datetime.now(timezone.utc).isoformat(),
                )
                results.append(resource)
                logger.info("Community: %s — %s", subject_code, href)

        return results
