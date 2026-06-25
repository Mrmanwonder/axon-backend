from __future__ import annotations

import os
import time
from typing import Any, Dict, Tuple

_SYLLABUS_CACHE_TTL_SECONDS = int(os.environ.get("SYLLABUS_CACHE_TTL_SECONDS", "900"))
_SYLLABUS_ALL_CACHE: Tuple[float, Dict[str, dict[str, Any]]] | None = None


def get_cached_syllabus_maps(db) -> Dict[str, dict[str, Any]]:
    """
    Fetches the entire 'syllabus_maps' collection from Firestore and caches it
    in-memory for a limited time to avoid redundant DB reads across requests.
    Returns a mapping of document ID to document data.
    """
    global _SYLLABUS_ALL_CACHE
    now = time.time()

    if _SYLLABUS_ALL_CACHE and now - _SYLLABUS_ALL_CACHE[0] < _SYLLABUS_CACHE_TTL_SECONDS:
        return {k: dict(v) for k, v in _SYLLABUS_ALL_CACHE[1].items()}

    snaps = db.collection("syllabus_maps").stream()
    all_maps = {s.id: (s.to_dict() or {}) for s in snaps}

    _SYLLABUS_ALL_CACHE = (now, all_maps)

    # Return a copy to prevent accidental mutation by callers
    return {k: dict(v) for k, v in all_maps.items()}
