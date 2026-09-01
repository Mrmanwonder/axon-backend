import os
import time as time_module
from typing import Dict, Tuple, Optional

_SYLLABUS_CACHE_TTL_SECONDS = int(os.environ.get("SYLLABUS_CACHE_TTL_SECONDS", "900"))
_SYLLABUS_ALL_CACHE: Optional[Tuple[float, Dict[str, dict]]] = None

def get_all_syllabus_maps(db) -> Dict[str, dict]:
    global _SYLLABUS_ALL_CACHE
    now = time_module.time()

    if _SYLLABUS_ALL_CACHE and now - _SYLLABUS_ALL_CACHE[0] < _SYLLABUS_CACHE_TTL_SECONDS:
        return {k: dict(v) for k, v in _SYLLABUS_ALL_CACHE[1].items()}

    snaps = db.collection("syllabus_maps").stream()
    all_maps = {s.id: s.to_dict() for s in snaps}
    _SYLLABUS_ALL_CACHE = (now, all_maps)

    return {k: dict(v) for k, v in all_maps.items()}
