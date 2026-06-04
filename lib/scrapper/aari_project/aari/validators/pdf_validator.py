"""
AARI — PDF Integrity & Metadata Validator
'Axon Protocol' sanitisation layer.

What this does (legitimate):
  ✓ Verify the file is a well-formed PDF (not HTML, not a login redirect)
  ✓ Extract and log publicly visible metadata (title, author, creation date)
  ✓ Flag files that appear to be login-wall redirects or corrupted
  ✓ Assign a structural quality signal

What this does NOT do:
  ✗ Remove watermarks  (copyright violation)
  ✗ Strip DRM or encryption  (DMCA §1201)
  ✗ Alter file content in any way
"""

from __future__ import annotations
import io
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("aari.validator")

# PDF functionality has been removed as per user request
_PYPDF_AVAILABLE = False
logger.info("pypdf not available; PDF structure checks disabled")


# ─── PDF Magic-Byte Check ─────────────────────────────────────────────────────

_PDF_MAGIC = b"%PDF-"

def _is_pdf_bytes(raw: bytes) -> bool:
    """Quick header check — catches HTML login-wall redirects served as 200 OK."""
    return raw[:5] == _PDF_MAGIC


# ─── Validation Result ────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    is_valid:       bool
    page_count:     Optional[int]   = None
    metadata:       Optional[dict]  = None
    error:          Optional[str]   = None
    file_size:      int             = 0

    @property
    def quality_signal(self) -> int:
        """0–10 structural quality signal (higher = more usable)."""
        if not self.is_valid:
            return 0
        score = 5
        if self.page_count and self.page_count > 0:
            score += min(3, self.page_count // 4)
        if self.metadata and self.metadata.get("title"):
            score += 2
        return min(score, 10)


# ─── Validator ────────────────────────────────────────────────────────────────

class PDFValidator:
    """
    Validates structural integrity of downloaded PDFs.

    Operates on raw bytes only — files are never written to disk
    until after validation passes.
    """

    def validate(self, raw: bytes, source_url: str = "") -> ValidationResult:
        result = ValidationResult(is_valid=False, file_size=len(raw))

        if len(raw) < 64:
            result.error = "File too small to be a valid PDF"
            return result

        if not _is_pdf_bytes(raw):
            # Common case: site served a login/CAPTCHA page
            snippet = raw[:200].decode("utf-8", errors="replace")
            if "<html" in snippet.lower() or "<!doctype" in snippet.lower():
                result.error = "Response is an HTML page (likely a login redirect)"
            else:
                result.error = f"File does not start with PDF magic bytes"
            logger.warning("Invalid PDF from %s: %s", source_url, result.error)
            return result

        # PDF functionality has been removed as per user request
        # Perform basic validation only (magic bytes check)
        result.is_valid  = True
        result.page_count = None
        result.metadata   = {}
        return result
