import asyncio
import time
import pytest
from unittest.mock import patch, MagicMock

from services.exam_dates_service import OfficialExamDatesService

@pytest.fixture
def service():
    mock_db = MagicMock()
    return OfficialExamDatesService(db=mock_db)

def test_fetch_dates_performance(service):
    # Create mock HTML response with many PDF links
    num_pdfs = 20
    html_content = "<html><body>"
    for i in range(num_pdfs):
        html_content += f'<a href="http://example.com/pdf{i}.pdf">June 2024 Zone {i}.pdf</a>\n'
    html_content += "</body></html>"

    mock_html_response = MagicMock()
    mock_html_response.text = html_content
    mock_html_response.raise_for_status = MagicMock()

    # Create mock PDF response
    mock_pdf_response = MagicMock()
    mock_pdf_response.content = b"Mock PDF Content"
    mock_pdf_response.raise_for_status = MagicMock()

    # We want requests.get to be slow to simulate network latency for the HTML fetch
    def delayed_requests_get(url, *args, **kwargs):
        if "pdf" in url:
            time.sleep(0.1) # Simulate network delay for old sequential logic
            return mock_pdf_response
        return mock_html_response

    # Mock httpx.AsyncClient.get for the PDF fetches
    async def delayed_httpx_get(*args, **kwargs):
        await asyncio.sleep(0.1) # Simulate network delay
        return mock_pdf_response

    with patch('requests.get', side_effect=delayed_requests_get), \
         patch('httpx.AsyncClient.get', side_effect=delayed_httpx_get):

        start_time = time.time()

        events = service.fetch_official_exam_dates(
            board="caie_igcse",
            subjects=["Math", "Physics"],
            year=2024,
            series="june",
            administrative_zone=None
        )

        end_time = time.time()
        duration = end_time - start_time

        print(f"\nTime taken for {num_pdfs} requests: {duration:.4f} seconds")
        print(f"Events found: {len(events)}")

        # Expected baseline for 20 PDFs sequentially would be ~2.0 seconds
        expected_sequential = 2.0
        print(f"Expected sequential time: {expected_sequential:.4f} seconds")
        print(f"Improvement: {expected_sequential / duration:.2f}x faster")

        # Assert concurrent execution was significantly faster than sequential
        assert duration < 1.0, f"Expected concurrent execution to be fast, took {duration}s"

if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])
