import pytest
from app import build_syllabus_context
from unittest.mock import patch, MagicMock

@patch("app.syllabus_maps_collection")
def test_build_syllabus_context_empty(mock_collection):
    assert build_syllabus_context([]) == ""
    mock_collection.assert_not_called()

@patch("app.syllabus_maps_collection")
def test_build_syllabus_context_with_ids(mock_collection):
    mock_query = MagicMock()
    mock_collection.return_value.where.return_value.get.return_value = [
        MagicMock(
            exists=True,
            to_dict=lambda: {"board": "B1", "subject": "S1", "paper": "P1", "topic": "T1", "code": "obj_1", "description": "D1"}
        )
    ]

    # We clear cache for deterministic test
    import app
    app._syllabus_cache.clear()

    res = build_syllabus_context(["obj_1"])

    # Ensure Firestore 'in' query was used with chunks
    mock_collection.return_value.where.assert_called_with("code", "in", ["obj_1"])

    assert res == "B1 / S1 / P1 / T1 / obj_1: D1"

    # Check that second call hits cache
    mock_collection.reset_mock()
    res2 = build_syllabus_context(["obj_1"])
    assert res2 == "B1 / S1 / P1 / T1 / obj_1: D1"
    mock_collection.assert_not_called()
