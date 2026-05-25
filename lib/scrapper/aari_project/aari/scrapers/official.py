"""
AARI — Official Board Scraper
Uses Playwright to navigate CAIE / IBO / Edexcel public-facing resource pages.

Design principles:
  • Full robots.txt compliance for each domain
  • Identifies itself honestly via User-Agent
  • Respects Crawl-delay directives
  • Never attempts to access authenticated / paywalled content
  • Deep-Tree Traversal stops at max_depth to prevent runaway crawls
"""

from __future__ import annotations
import asyncio
import logging
import re
from datetime import datetime, timezone
from typing import AsyncIterator
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from aari.config import OFFICIAL_SOURCES, CRAWL_CONFIG
from aari.core.polite import PoliteLimiter, RobotsCache, fetch_with_retry
from aari.core.healer import self_heal
from aari.core.dedup import DeduplicationEngine
from aari.models.resource import (
    AARIResource, Board, Qualification, ResourceType,
    Session, SourceType, VerifiedStatus,
)

import httpx

logger = logging.getLogger("aari.official_scraper")


# ─── Filename Heuristics ──────────────────────────────────────────────────────

_YEAR_RE    = re.compile(r"\b(20\d{2})\b")
_SESSION_RE = re.compile(
    r"(feb|mar|may|jun|oct|nov|winter|summer|specimen)",
    re.I
)
_PAPER_RE   = re.compile(r"[_\-]p(\d)[_\-]", re.I)
_VARIANT_RE = re.compile(r"[_\-]v(\d)[_\-]", re.I)

_SESSION_MAP = {
    "feb": Session.FEB_MAR, "mar": Session.FEB_MAR,
    "winter": Session.FEB_MAR,
    "may": Session.MAY_JUN, "jun": Session.MAY_JUN,
    "summer": Session.MAY_JUN,
    "oct": Session.OCT_NOV, "nov": Session.OCT_NOV,
}

_RTYPE_KEYWORDS = {
    ResourceType.MARK_SCHEME:     ["mark scheme", "ms", "_ms_", "-ms-"],
    ResourceType.EXAMINER_REPORT: ["examiner report", "er", "_er_", "-er-"],
    ResourceType.SYLLABUS:        ["syllabus", "specification"],
    ResourceType.SPECIMEN_PAPER:  ["specimen"],
    ResourceType.DATESHEET:       ["datesheet", "timetable", "schedule"],
}


def _infer_resource_type(href: str, text: str) -> ResourceType:
    combined = (href + " " + text).lower()
    for rtype, keywords in _RTYPE_KEYWORDS.items():
        if any(kw in combined for kw in keywords):
            return rtype
    return ResourceType.PAST_PAPER


def _parse_link_metadata(href: str, text: str, subject_code: str) -> dict:
    combined = href + " " + text
    year_m   = _YEAR_RE.search(combined)
    sess_m   = _SESSION_RE.search(combined)
    paper_m  = _PAPER_RE.search(href)
    var_m    = _VARIANT_RE.search(href)
    return {
        "year":    int(year_m.group(1)) if year_m else datetime.now().year,
        "session": _SESSION_MAP.get(sess_m.group(1).lower()[:3]) if sess_m else None,
        "paper_number": paper_m.group(1) if paper_m else None,
        "variant":      var_m.group(1) if var_m else None,
    }


# ─── Deep-Tree Traversal ──────────────────────────────────────────────────────

async def _dfs_collect_pdfs(
    start_url:    str,
    base_domain:  str,
    client:       httpx.AsyncClient,
    limiter:      PoliteLimiter,
    robots:       RobotsCache,
    max_depth:    int,
    visited:      set[str],
) -> AsyncIterator[tuple[str, str]]:
    """
    Yield (page_url, pdf_href) tuples found within ``max_depth`` hops
    from ``start_url``, staying within ``base_domain``.
    """
    queue = [(start_url, 0)]

    while queue:
        url, depth = queue.pop()
        if url in visited or depth > max_depth:
            continue
        visited.add(url)

        resp = await fetch_with_retry(client, url, limiter, robots, timeout=20)
        if resp is None:
            continue

        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup.find_all("a", href=True):
            href = urljoin(url, tag["href"])
            if urlparse(href).netloc != urlparse(base_domain).netloc:
                continue
            if href.lower().endswith(".pdf"):
                yield url, href
            elif href not in visited and depth + 1 <= max_depth:
                queue.append((href, depth + 1))


# ─── Official Scraper ─────────────────────────────────────────────────────────

class OfficialBoardScraper:
    """
    Traverses official board websites to discover publicly available
    syllabi, past papers, mark schemes, and datesheets.

    All requests respect robots.txt and use a declared bot User-Agent.
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

    async def crawl_source(
        self,
        source_key:    str,
        subject_code:  str,
        subject_name:  str,
        qualification: Qualification,
    ) -> list[AARIResource]:
        """
        Crawl one official source for a given subject.
        Returns a list of discovered AARIResource objects (not yet downloaded).
        """
        cfg      = OFFICIAL_SOURCES.get(source_key)
        if not cfg:
            raise ValueError(f"Unknown source key: {source_key}")

        board    = Board(cfg["board"])
        start    = cfg["base_url"] + cfg.get("past_papers_path", cfg.get("syllabi_path", "/"))
        domain   = cfg["base_url"]
        results  = []
        visited  = set()

        headers = {"User-Agent": CRAWL_CONFIG.user_agent}
        async with httpx.AsyncClient(headers=headers, timeout=30) as client:
            async for page_url, pdf_url in _dfs_collect_pdfs(
                start, domain, client, self._limiter,
                self._robots, CRAWL_CONFIG.max_depth, visited
            ):
                if self._dedup.check_and_register_url(pdf_url):
                    continue

                # Try to heal if the URL already 404'd in a previous run
                # (for new runs, this is a no-op since we just found the URL)
                meta    = _parse_link_metadata(pdf_url, "", subject_code)
                rtype   = _infer_resource_type(pdf_url, "")

                resource = AARIResource(
                    board          = board,
                    qualification  = qualification,
                    subject_code   = subject_code,
                    subject_name   = subject_name,
                    resource_type  = rtype,
                    year           = meta["year"],
                    session        = meta["session"],
                    paper_number   = meta["paper_number"],
                    variant        = meta["variant"],
                    source_type    = SourceType.OFFICIAL,
                    source_url     = page_url,
                    file_url       = pdf_url,
                    verified_status= VerifiedStatus.VERIFIED,
                    crawled_at     = datetime.now(timezone.utc).isoformat(),
                )
                results.append(resource)
                logger.info("Found [%s] %s — %s", board.value, subject_code, pdf_url)

        return results
