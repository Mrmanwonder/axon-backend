import pytest
from middleware.supabase_scope import apply_user_scope, assert_table_access

def test_apply_user_scope_public_table():
    params = {"select": "*"}
    result = apply_user_scope("subjects", "select", params, "user123")
    # Public tables should not have scope applied
    assert result == {"select": "*"}

def test_apply_user_scope_basic_select():
    params = {"select": "*"}
    result = apply_user_scope("user_notes", "select", params, "user123")
    assert result == {"select": "*", "user_id": "eq.user123"}
    assert "eq.user_id" not in result

def test_apply_user_scope_preserves_logical_operators():
    # User tries to access their notes using logical operators
    params = {
        "select": "*",
        "or": "(status.eq.draft,status.eq.archived)",
        "and": "(priority.eq.high)"
    }
    result = apply_user_scope("user_notes", "select", params, "user123")

    # Authorized filter should be injected, preserving the other logical ops
    assert result == {
        "select": "*",
        "or": "(status.eq.draft,status.eq.archived)",
        "and": "(priority.eq.high)",
        "user_id": "eq.user123"
    }

def test_apply_user_scope_fixes_legacy_eq_bug():
    # If a legacy `eq.user_id` is passed, it should be removed and fixed to `user_id: eq.xyz`
    params = {"select": "*", "eq.user_id": "malicious_user"}
    result = apply_user_scope("user_notes", "select", params, "user123")
    assert result == {"select": "*", "user_id": "eq.user123"}
    assert "eq.user_id" not in result

def test_apply_user_scope_insert_body():
    # Test that inserts correctly enforce the user_id in the body
    params = {
        "body": {
            "title": "My Note",
            "content": "Hello",
            "user_id": "user123" # Matching user
        }
    }
    result = apply_user_scope("user_notes", "insert", params, "user123")
    assert result["body"]["user_id"] == "user123"

def test_apply_user_scope_insert_body_malicious():
    # Test that inserts raise PermissionError when trying to insert for another user
    params = {
        "body": {
            "title": "My Note",
            "user_id": "malicious_user" # Wrong user
        }
    }
    with pytest.raises(PermissionError):
        apply_user_scope("user_notes", "insert", params, "user123")

def test_apply_user_scope_insert_body_missing_user():
    # Test that inserts auto-inject the user_id if missing
    params = {
        "body": {
            "title": "My Note",
        }
    }
    result = apply_user_scope("user_notes", "insert", params, "user123")
    assert result["body"]["user_id"] == "user123"

def test_assert_table_access():
    # Select on public table is allowed
    assert_table_access("subjects", "select")

    # Insert on public table is forbidden
    with pytest.raises(PermissionError):
        assert_table_access("subjects", "insert")

    # Any operation on user table is allowed (it will be scoped by apply_user_scope)
    assert_table_access("user_notes", "insert")
    assert_table_access("user_notes", "delete")
    assert_table_access("user_notes", "select")

    # Unknown tables are forbidden
    with pytest.raises(PermissionError):
        assert_table_access("unknown_table", "select")
