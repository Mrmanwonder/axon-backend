import pytest
from unittest.mock import MagicMock
import time

from app import build_syllabus_context

class MockDoc:
    def __init__(self, data: dict):
        self._data = data

    def to_dict(self):
        return self._data

@pytest.fixture
def mock_syllabus_collection(mocker):
    collection_mock = MagicMock()
    mocker.patch("app.syllabus_maps_collection", return_value=collection_mock)

    # We will mock the batch 'in' query behavior which we are about to implement,
    # as well as the sequential 'where' query to make sure both can pass (if we test before and after)
    def mock_where(*args, **kwargs):
        query_mock = MagicMock()

        # for batch: where("code", "in", chunk)
        if len(args) == 3 and args[0] == "code" and args[1] == "in":
            codes = args[2]
            docs = []
            for code in codes:
                if code == "MATH1":
                    docs.append(MockDoc({"board": "CAIE", "subject": "Math", "paper": "P1", "topic": "Algebra", "code": "MATH1", "description": "Basic Algebra"}))
                elif code == "PHY1":
                    docs.append(MockDoc({"board": "CAIE", "subject": "Physics", "paper": "P2", "topic": "Mechanics", "code": "PHY1", "description": "Basic Mechanics"}))

            def mock_stream():
                return docs
            query_mock.stream = mock_stream
            return query_mock

        # for sequential: where("code", is_equal_to="...")
        if args and args[0] == "code" and kwargs.get("is_equal_to"):
            code = kwargs.get("is_equal_to")
            docs = []
            if code == "MATH1":
                docs.append(MockDoc({"board": "CAIE", "subject": "Math", "paper": "P1", "topic": "Algebra", "code": "MATH1", "description": "Basic Algebra"}))
            elif code == "PHY1":
                docs.append(MockDoc({"board": "CAIE", "subject": "Physics", "paper": "P2", "topic": "Mechanics", "code": "PHY1", "description": "Basic Mechanics"}))

            def mock_get():
                return docs
            limit_mock = MagicMock()
            limit_mock.get = mock_get
            query_mock.limit.return_value = limit_mock
            return query_mock

        return query_mock

    collection_mock.where = mock_where
    return collection_mock

@pytest.fixture(autouse=True)
def reset_cache():
    import app
    if hasattr(app, "_syllabus_cache"):
        app._syllabus_cache.clear()

def test_build_syllabus_context_empty():
    assert build_syllabus_context([]) == ""

def test_build_syllabus_context_single(mock_syllabus_collection):
    result = build_syllabus_context(["MATH1"])
    assert result == "CAIE / Math / P1 / Algebra / MATH1: Basic Algebra"

def test_build_syllabus_context_multiple(mock_syllabus_collection):
    result = build_syllabus_context(["MATH1", "PHY1"])
    assert "CAIE / Math / P1 / Algebra / MATH1: Basic Algebra" in result
    assert "CAIE / Physics / P2 / Mechanics / PHY1: Basic Mechanics" in result

def test_build_syllabus_context_missing(mock_syllabus_collection):
    # Test with an ID that isn't in our mock
    result = build_syllabus_context(["UNKNOWN1"])
    assert result == ""

def test_build_syllabus_context_order(mock_syllabus_collection):
    # Should preserve the order of the input IDs
    result1 = build_syllabus_context(["MATH1", "PHY1"])
    result2 = build_syllabus_context(["PHY1", "MATH1"])

    assert result1 == "CAIE / Math / P1 / Algebra / MATH1: Basic Algebra | CAIE / Physics / P2 / Mechanics / PHY1: Basic Mechanics"
    assert result2 == "CAIE / Physics / P2 / Mechanics / PHY1: Basic Mechanics | CAIE / Math / P1 / Algebra / MATH1: Basic Algebra"
