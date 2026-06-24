import time
import threading
from typing import Any, Dict

_SYLLABUS_MAPS_CACHE: Dict[str, dict] = {}
_SYLLABUS_CACHE_TIMESTAMP: float = 0.0
_SYLLABUS_CACHE_TTL_SECONDS = 3600
_SYLLABUS_CACHE_LOCK = threading.Lock()

def get_syllabus_map(db, objective_id: str) -> dict[str, Any]:
    """
    Thread-safe TTL cache fetch for a single syllabus objective by its code.
    Prevents N+1 database lookups when hydrating grading context.
    """
    global _SYLLABUS_MAPS_CACHE, _SYLLABUS_CACHE_TIMESTAMP
    now = time.time()

    with _SYLLABUS_CACHE_LOCK:
        if not _SYLLABUS_MAPS_CACHE or (now - _SYLLABUS_CACHE_TIMESTAMP) > _SYLLABUS_CACHE_TTL_SECONDS:
            # Build cache
            snaps = db.collection("syllabus_maps").stream()
            new_cache = {}
            for snap in snaps:
                data = snap.to_dict() or {}
                # Index by code, fallback to objective_id, id
                code = data.get("code")
                if code:
                    new_cache[str(code)] = data
            _SYLLABUS_MAPS_CACHE = new_cache
            _SYLLABUS_CACHE_TIMESTAMP = now

    return _SYLLABUS_MAPS_CACHE.get(objective_id, {})

def get_all_syllabus_docs(db) -> list[dict[str, Any]]:
    """
    Thread-safe TTL cache fetch for all syllabus docs.
    """
    global _SYLLABUS_MAPS_CACHE, _SYLLABUS_CACHE_TIMESTAMP
    now = time.time()

    with _SYLLABUS_CACHE_LOCK:
        if not _SYLLABUS_MAPS_CACHE or (now - _SYLLABUS_CACHE_TIMESTAMP) > _SYLLABUS_CACHE_TTL_SECONDS:
            snaps = db.collection("syllabus_maps").stream()
            new_cache = {}
            for snap in snaps:
                data = snap.to_dict() or {}
                code = data.get("code")
                if code:
                    new_cache[str(code)] = data
            _SYLLABUS_MAPS_CACHE = new_cache
            _SYLLABUS_CACHE_TIMESTAMP = now

    return list(_SYLLABUS_MAPS_CACHE.values())
