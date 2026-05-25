"""
AARI — Polite Crawler Primitives
• robots.txt enforcement
• Staggered async rate-limiting (polite, not evasion-oriented)
• Retry logic with exponential back-off
"""

from __future__ import annotations
import asyncio
import logging
import random
import time
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser
from typing import Dict, Optional

import httpx

logger = logging.getLogger("aari.polite")


# ─── robots.txt Cache ─────────────────────────────────────────────────────────

class RobotsCache:
    """
    Fetches and caches robots.txt per domain.
    Enforces restrictions before any HTTP request is dispatched.
    """

    def __init__(self, user_agent: str) -> None:
        self._ua      = user_agent
        self._parsers: Dict[str, RobotFileParser] = {}

    async def fetch(self, base_url: str, client: httpx.AsyncClient) -> RobotFileParser:
        parsed = urlparse(base_url)
        domain = f"{parsed.scheme}://{parsed.netloc}"
        if domain not in self._parsers:
            robots_url = f"{domain}/robots.txt"
            rp = RobotFileParser()
            rp.set_url(robots_url)
            try:
                resp = await client.get(robots_url, timeout=10)
                if resp.status_code == 200:
                    rp.parse(resp.text.splitlines())
                    logger.debug("Loaded robots.txt for %s", domain)
                else:
                    # No robots.txt → assume fully permissive
                    logger.debug("No robots.txt at %s (HTTP %s)", domain, resp.status_code)
            except Exception as exc:
                logger.warning("Could not fetch robots.txt for %s: %s", domain, exc)
            self._parsers[domain] = rp
        return self._parsers[domain]

    async def can_fetch(self, url: str, client: httpx.AsyncClient) -> bool:
        base = "{0.scheme}://{0.netloc}".format(urlparse(url))
        rp   = await self.fetch(base, client)
        allowed = rp.can_fetch(self._ua, url)
        if not allowed:
            logger.info("robots.txt disallows: %s", url)
        return allowed


# ─── Polite Rate Limiter ──────────────────────────────────────────────────────

class PoliteLimiter:
    """
    Per-domain bucket that enforces a minimum inter-request delay.
    Delays are randomised within [min, max] to appear less mechanical
    while staying well above scraping-evasion territory —
    the intent is courtesy, not concealment.
    """

    def __init__(self, delay_min: float = 1.5, delay_max: float = 4.0) -> None:
        self._min   = delay_min
        self._max   = delay_max
        self._last:  Dict[str, float] = {}
        self._lock:  asyncio.Lock = asyncio.Lock()

    async def wait(self, url: str) -> None:
        domain = urlparse(url).netloc
        async with self._lock:
            last = self._last.get(domain, 0.0)
            now  = time.monotonic()
            gap  = random.uniform(self._min, self._max)
            wait = max(0.0, (last + gap) - now)
            self._last[domain] = now + wait
        if wait > 0:
            logger.debug("Polite delay %.2fs for %s", wait, domain)
            await asyncio.sleep(wait)


# ─── Retry Helper ─────────────────────────────────────────────────────────────

async def fetch_with_retry(
    client:      httpx.AsyncClient,
    url:         str,
    limiter:     PoliteLimiter,
    robots:      RobotsCache,
    max_retries: int   = 3,
    backoff:     float = 2.0,
    **kwargs,
) -> Optional[httpx.Response]:
    """
    Polite GET with:
      • robots.txt gate
      • per-domain rate limiting
      • exponential backoff on transient errors (5xx, timeout, network)
    """
    if not await robots.can_fetch(url, client):
        return None

    for attempt in range(1, max_retries + 1):
        await limiter.wait(url)
        try:
            resp = await client.get(url, follow_redirects=True, **kwargs)
            if resp.status_code == 200:
                return resp
            if resp.status_code in (429, 503):
                wait = backoff ** attempt
                logger.warning("Rate-limited (%s) on %s — backing off %.1fs",
                               resp.status_code, url, wait)
                await asyncio.sleep(wait)
            elif resp.status_code in (404, 403, 410):
                logger.info("Terminal HTTP %s for %s", resp.status_code, url)
                return None
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            wait = backoff ** attempt
            logger.warning("Attempt %d/%d failed for %s: %s — retrying in %.1fs",
                           attempt, max_retries, url, exc, wait)
            await asyncio.sleep(wait)

    logger.error("All %d attempts exhausted for %s", max_retries, url)
    return None
