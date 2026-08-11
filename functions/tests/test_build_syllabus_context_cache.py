import pytest
from unittest.mock import MagicMock, patch
import app
import time

def test_build_syllabus_context_caching():
    # Setup mock
    mock_collection = MagicMock()
    mock_query = MagicMock()
    mock_snapshot = MagicMock()

    mock_doc1 = MagicMock()
    mock_doc1.to_dict.return_value = {"code": "A1", "board": "B", "subject": "S", "paper": "P", "topic": "T", "description": "D1"}

    mock_snapshot.__iter__.return_value = [mock_doc1]
    mock_query.get.return_value = mock_snapshot
    mock_collection.where.return_value = mock_query

    # Clear cache
    app._SYLLABUS_CACHE.clear()

    with patch("app.syllabus_maps_collection", return_value=mock_collection):
        res1 = app.build_syllabus_context(["A1", "A2"])
        # A1 is found, A2 is missing
        assert "A1: D1" in res1
        assert "A2" not in res1
        assert mock_collection.where.call_count == 1

        # Second call should use cache and NOT make db calls
        res2 = app.build_syllabus_context(["A1", "A2"])
        assert "A1: D1" in res2
        assert "A2" not in res2
        assert mock_collection.where.call_count == 1 # count didn't increase

if __name__ == "__main__":
    test_build_syllabus_context_caching()
