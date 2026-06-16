# Stub middleware to prevent import errors
# The supabase_scope middleware is not deployed on Render

from typing import Any

# Allowlist - maps to unified curriculum table
PUBLIC_READ_TABLES = {
    "boards",
    "subjects",
    "chapters",
    "Datesheet",
    "PYQs",
    "curriculum",  # Unified table (not curriculum_subjects/chapters/subchapters/syllabi)
    "global_notes",
    "subchapter_notes",
    "papers",
    "topics",
}

USER_SCOPED_TABLES = {
    "user_notes",
    "user_subjects",
    "user_subchapter_progress",
    "user_pyqs",
    "user_mocks",
    "study_progress",
    "user_bookmarks",
    "user_recent_papers",
    "user_personal_index",
}

def apply_user_scope(
    table: str,
    method: str,
    params: dict[str, Any],
    user_id: str,
) -> dict[str, Any]:
    """Apply user scope to params (stub for local/dev only)."""
    return params

def assert_table_access(table: str, method: str) -> None:
    """Assert table access (stub for local/dev only)."""
    # No-op stub - all access allowed
    pass