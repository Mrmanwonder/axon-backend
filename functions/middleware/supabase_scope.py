from typing import Any

USER_COLUMN_BY_TABLE = {
    "user_bookmarks": "user_id",
    "user_recent_papers": "user_id",
}

def _user_column(table: str) -> str:
    return USER_COLUMN_BY_TABLE.get(table, "user_id")

PUBLIC_READ_TABLES = {
    "boards",
    "subjects",
    "chapters",
    "Datesheet",
    "PYQs",
    "curriculum",
    "global_notes",
    "subchapter_notes",
    "papers",
    "topics",
    "sme_questions",
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

def _ensure_body_user_scope(body: Any, user_column: str, user_id: str) -> Any:
    if isinstance(body, list):
        return [_ensure_body_user_scope(item, user_column, user_id) for item in body]
    if isinstance(body, dict):
        item = dict(body)
        existing = item.get(user_column)
        if existing not in (None, user_id):
            raise PermissionError("Cannot write another user's data")
        item[user_column] = user_id
        return item
    return body

def apply_user_scope(
    table: str,
    method: str,
    params: dict[str, Any],
    user_id: str,
) -> dict[str, Any]:
    """Apply user scope to params to prevent accessing or modifying other users' data."""
    if table not in USER_SCOPED_TABLES:
        return params

    scoped_params = dict(params)
    user_column = _user_column(table)

    if method in ("select", "update", "delete"):
        # We should NOT remove "or" and "and" keys because that breaks legitimate filters (e.g. deleting notes where status=draft OR status=archived).
        # PostgREST naturally ANDs top-level query parameters. So adding `user_id=eq.user_id` will safely enforce `WHERE user_id=... AND (...)`.

        # However, we DO need to remove any direct top-level keys that conflict with the `user_column` to avoid
        # ambiguity or error. For example, if the user explicitly provided `user_id=eq.456`, we just overwrite it
        # by simply setting `scoped_params[user_column] = f"eq.{user_id}"`.

        # We only remove the old buggy `eq.user_id` if it was somehow passed in, to avoid the PGRST100 error.
        if f"eq.{user_column}" in scoped_params:
            del scoped_params[f"eq.{user_column}"]

        # PostgREST uses `column=eq.value` for exact match filtering in query params
        scoped_params[user_column] = f"eq.{user_id}"
    
    if method in ("insert", "update", "upsert") and "body" in scoped_params:
        # Enforce that any inserted/updated data correctly identifies the user
        scoped_params["body"] = _ensure_body_user_scope(scoped_params["body"], user_column, user_id)

    return scoped_params

def assert_table_access(table: str, method: str) -> None:
    """Assert table access and ensure read-only operations on public tables."""
    if table in PUBLIC_READ_TABLES:
        if method != "select":
            raise PermissionError(f"Cannot {method} on public table '{table}'")
    elif table in USER_SCOPED_TABLES:
        pass
    else:
        raise PermissionError(f"Access to table '{table}' is forbidden")
