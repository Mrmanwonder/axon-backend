"""
AARI — Self-Healing Link Resolver
'Observation Loop': when a known URL pattern 404s, the resolver walks
the parent directory's DOM, scores anchor text + href with fuzzy string
matching (difflib), and attempts to locate the moved resource.

This is entirely above-board — it's what a human would do when a bookmark
breaks: navigate up one level and look for the file.
"""

from __future__ import annotations
import logging
import re
from difflib import SequenceMatcher
from typing import Optional
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger("aari.healer")


def _parent_url(url: str) -> Optional[str]:
    """Return the parent directory URL, or None if already at root."""
    parsed = urlparse(url)
    path   = parsed.path.rstrip("/")
    if not path or path == "/":
        return None
    parent_path = "/".join(path.split("/")[:-1]) or "/"
    return parsed._replace(path=parent_path, query="", fragment="").geturl()


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _extract_candidates(html: str, base_url: str, hint: str) -> list[tuple[float, str]]:
    """
    Parse all <a href> links from ``html``, score them against ``hint``,
    return sorted list of (score, absolute_url).
    """
    soup  = BeautifulSoup(html, "html.parser")
    seen  = set()
    items = []
    for tag in soup.find_all("a", href=True):
        href = urljoin(base_url, tag["href"])
        if href in seen:
            continue
        seen.add(href)
        text  = (tag.get_text(strip=True) + " " + tag["href"])
        score = _similarity(hint, text)
        if score > 0.25:                    # floor: discard obviously unrelated
            items.append((score, href))
    items.sort(key=lambda x: x[0], reverse=True)
    return items[:10]


async def self_heal(
    original_url: str,
    hint:         str,
    client:       httpx.AsyncClient,
    max_ascent:   int = 3,
) -> Optional[str]:
    """
    Attempt to find a relocated resource.

    Parameters
    ----------
    original_url : str
        The URL that returned 404 / 403.
    hint : str
        A string describing the resource (e.g. filename or subject name).
        Used as the fuzzy-match target against anchor text in the parent page.
    client : httpx.AsyncClient
        Shared HTTP client (already configured with headers/timeout).
    max_ascent : int
        How many directory levels to climb before giving up.

    Returns
    -------
    str | None
        The best candidate URL found, or None if healing failed.
    """
    current = original_url
    for level in range(1, max_ascent + 1):
        parent = _parent_url(current)
        if parent is None:
            break
        logger.info("[Healer L%d] Ascending to %s (hint: %r)", level, parent, hint)
        try:
            resp = await client.get(parent, follow_redirects=True, timeout=15)
        except Exception as exc:
            logger.warning("[Healer L%d] Could not fetch parent %s: %s", level, parent, exc)
            current = parent
            continue

        if resp.status_code != 200:
            current = parent
            continue

        candidates = _extract_candidates(resp.text, parent, hint)
        if candidates:
            best_score, best_url = candidates[0]
            logger.info("[Healer L%d] Best match (%.2f): %s", level, best_score, best_url)
            if best_score >= 0.55:
                return best_url

        current = parent

    logger.warning("[Healer] Could not resolve %s after %d levels", original_url, max_ascent)
    return None
