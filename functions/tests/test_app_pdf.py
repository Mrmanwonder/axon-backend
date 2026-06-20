import pytest
from fastapi.testclient import TestClient
from functions.app import app, load_pdf_bytes, AnalyzePdfRequest
from unittest.mock import patch, MagicMock, AsyncMock
import httpx
import pytest_asyncio

@pytest.mark.asyncio
async def test_load_pdf_bytes_base64():
    import base64
    content = b"pdf content"
    b64 = base64.b64encode(content).decode("utf-8")
    payload = AnalyzePdfRequest(pdf_base64=b64, filename="test.pdf")

    result_content, filename = await load_pdf_bytes(payload)
    assert result_content == content
    assert filename == "test.pdf"

@pytest.mark.asyncio
async def test_load_pdf_bytes_url():
    payload = AnalyzePdfRequest(pdf_url="https://example.com/test.pdf")

    with patch("httpx.AsyncClient") as mock_client:
        mock_response = MagicMock()
        mock_response.content = b"pdf content"
        mock_response.status_code = 200

        mock_instance = MagicMock()
        mock_instance.get = AsyncMock(return_value=mock_response)
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock()

        mock_client.return_value = mock_instance

        with patch("functions.app.assert_safe_https_url"):
            result_content, filename = await load_pdf_bytes(payload)

            assert result_content == b"pdf content"
            assert filename == "test.pdf"
