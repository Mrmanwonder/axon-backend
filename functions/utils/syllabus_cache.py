import os
import time

_SYLLABUS_CACHE_TTL_SECONDS = int(os.environ.get("SYLLABUS_CACHE_TTL_SECONDS", "3600"))

# Cache for all syllabus maps
_SYLLABUS_ALL_CACHE = None  # Tuple[float, dict]
_SYLLABUS_CODE_INDEX = None  # Tuple[float, dict]


def get_all_syllabus_maps(db) -> dict:
    global _SYLLABUS_ALL_CACHE
    now = time.time()

    if _SYLLABUS_ALL_CACHE and now - _SYLLABUS_ALL_CACHE[0] < _SYLLABUS_CACHE_TTL_SECONDS:
        return _SYLLABUS_ALL_CACHE[1]

    snaps = db.collection("syllabus_maps").stream()
    all_maps = {}
    for snap in snaps:
        data = snap.to_dict() or {}
        all_maps[snap.id] = data

    _SYLLABUS_ALL_CACHE = (now, all_maps)
    return all_maps


def get_syllabus_map(db, code: str) -> dict:
    global _SYLLABUS_CODE_INDEX

    all_maps = get_all_syllabus_maps(db)
    now = time.time()

    if _SYLLABUS_CODE_INDEX and now - _SYLLABUS_CODE_INDEX[0] < _SYLLABUS_CACHE_TTL_SECONDS:
        return _SYLLABUS_CODE_INDEX[1].get(code, {})

    code_index = {}
    for doc_id, data in all_maps.items():
        c = data.get("code") or data.get("objective_id")
        if c:
            code_index[c] = data

    _SYLLABUS_CODE_INDEX = (now, code_index)
    return code_index.get(code, {})
