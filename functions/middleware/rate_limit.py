# Stub middleware to prevent import errors
# The rate_limit middleware is not deployed on Render

from typing import Set

def cors_allowed_origins() -> Set[str]:
    """Get allowed CORS origins (stub for local/dev only)."""
    return {"*"}

def is_rate_limited(key: str, limit: int = 60, window_seconds: int = 60) -> bool:
    """Check if key is rate limited (stub for local/dev only)."""
    return False