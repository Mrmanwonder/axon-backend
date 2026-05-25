"""
AARI — Quality Scorer
Computes the `quality_score` (0–100) for each AARIResource based on
provenance, paired resources, recency, and structural validity.
"""

from __future__ import annotations
from datetime import datetime, timezone
from typing import Collection

from aari.models.resource import AARIResource, ResourceType, SourceType
from aari.config import QUALITY_WEIGHTS


CURRENT_YEAR = datetime.now(timezone.utc).year
RECENCY_WINDOW = 4    # years over which recency score is normalised


def _recency_score(year: int, weight: int) -> int:
    """Newer papers score closer to ``weight``; older ones approach 0."""
    delta  = max(0, CURRENT_YEAR - year)
    factor = max(0.0, 1.0 - delta / RECENCY_WINDOW)
    return round(factor * weight)


def compute_quality_score(
    resource:        AARIResource,
    all_resources:   Collection[AARIResource],
    pdf_quality_sig: int = 0,           # 0–10 from PDFValidator
) -> int:
    """
    Score breakdown  (max 100):
      40 — Official board source
      20 — Paired Mark Scheme exists in the same batch
      15 — Paired Examiner Report exists
      15 — Recency (normalised over RECENCY_WINDOW years)
      10 — Structural PDF validity (from PDFValidator.quality_signal)
    """
    score = 0

    # 1. Official source
    if resource.source_type == SourceType.OFFICIAL:
        score += QUALITY_WEIGHTS["is_official"]

    # 2. Paired Mark Scheme
    paired_ms = any(
        r.subject_code   == resource.subject_code
        and r.year       == resource.year
        and r.session    == resource.session
        and r.resource_type == ResourceType.MARK_SCHEME
        for r in all_resources
        if r.uid != resource.uid
    )
    if paired_ms:
        score += QUALITY_WEIGHTS["has_mark_scheme"]

    # 3. Paired Examiner Report
    paired_er = any(
        r.subject_code   == resource.subject_code
        and r.year       == resource.year
        and r.resource_type == ResourceType.EXAMINER_REPORT
        for r in all_resources
        if r.uid != resource.uid
    )
    if paired_er:
        score += QUALITY_WEIGHTS["has_er"]

    # 4. Recency
    score += _recency_score(resource.year, QUALITY_WEIGHTS["year_recency"])

    # 5. Structural PDF validity (normalised from 0–10 to 0–10)
    score += min(pdf_quality_sig, QUALITY_WEIGHTS["file_valid"])

    return min(score, 100)
