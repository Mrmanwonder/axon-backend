import os
import time
from typing import Set

# In-memory dictionary for sliding window rate limiting
# Format: { "key": [timestamp1, timestamp2, ...] }
_rate_limits = {}


def cors_allowed_origins() -> Set[str]:
    """Get allowed CORS origins from environment variable or default."""
    origins_str = os.environ.get("CORS_ALLOWED_ORIGINS")
    if origins_str:
        return {origin.strip() for origin in origins_str.split(",") if origin.strip()}
    # Fallback default if not specified (e.g. for local development)
    return {"*"}


def is_rate_limited(key: str, limit: int = 60, window_seconds: int = 60) -> bool:
    """
    Check if a key (e.g. IP address or User ID) has exceeded its rate limit.
    Uses a simple in-memory sliding window.
    """
    now = time.time()

    # Initialize list if missing
    if key not in _rate_limits:
        _rate_limits[key] = []

    # Remove timestamps older than our window
    _rate_limits[key] = [ts for ts in _rate_limits[key] if now - ts < window_seconds]

    # Check if we exceed the limit
    if len(_rate_limits[key]) >= limit:
        return True

    # Register the request
    _rate_limits[key].append(now)
    return False
