# Stub middleware to prevent import errors
# The supabase_scope middleware is not deployed on Render

from typing import Any

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