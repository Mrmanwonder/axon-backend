import pytest
from unittest.mock import MagicMock, patch
import app

def test_build_syllabus_context():
    # Setup mock
    mock_collection = MagicMock()
    mock_query = MagicMock()
    mock_snapshot = MagicMock()

    mock_doc1 = MagicMock()
    mock_doc1.to_dict.return_value = {"code": "A1", "board": "B", "subject": "S", "paper": "P", "topic": "T", "description": "D1"}
    mock_doc2 = MagicMock()
    mock_doc2.to_dict.return_value = {"code": "A2", "board": "B", "subject": "S", "paper": "P", "topic": "T", "description": "D2"}

    mock_snapshot.__iter__.return_value = [mock_doc1, mock_doc2]
    mock_query.get.return_value = mock_snapshot
    mock_collection.where.return_value = mock_query

    with patch("app.syllabus_maps_collection", return_value=mock_collection):
        res = app.build_syllabus_context(["A1", "A2"])

    print(res)

if __name__ == "__main__":
    test_build_syllabus_context()
