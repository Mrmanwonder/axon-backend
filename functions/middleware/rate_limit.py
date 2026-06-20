import os
import time
from typing import Set

_rate_limits = {}

def cors_allowed_origins() -> Set[str]:
    origins_str = os.environ.get("CORS_ALLOWED_ORIGINS")
    if origins_str:
        return {origin.strip() for origin in origins_str.split(",") if origin.strip()}
    return {"*"}

# Fix the bug where a path might be passed to 'limit' argument
def is_rate_limited(key: str, path: str = None, limit: int = 60, window_seconds: int = 60) -> bool:
    now = time.time()
    
    if key not in _rate_limits:
        _rate_limits[key] = []
        
    _rate_limits[key] = [ts for ts in _rate_limits[key] if now - ts < window_seconds]

    if len(_rate_limits[key]) >= limit:
        return True
        
    _rate_limits[key].append(now)
    return False
