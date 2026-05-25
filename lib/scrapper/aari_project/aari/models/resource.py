"""
AARI — Data Models
Pydantic schemas that map to Axon's Firestore Security Rules.
Every field documented here corresponds to a Firestore path validated
by `isOwner` / `validUserDoc` guards in the Axon backend.
"""

from __future__ import annotations
from enum import Enum
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4
import hashlib

from pydantic import BaseModel, Field, HttpUrl, field_validator, model_validator


# ─── Enumerations ─────────────────────────────────────────────────────────────

class Board(str, Enum):
    CAIE    = "CAIE"
    IBO     = "IBO"
    Edexcel = "Edexcel"

class Qualification(str, Enum):
    IGCSE     = "IGCSE"
    O_LEVEL   = "O-Level"
    A_LEVEL   = "A-Level"
    AS_LEVEL  = "AS-Level"
    IB_DP     = "IB-DP"

class ResourceType(str, Enum):
    PAST_PAPER      = "Past Paper"
    MARK_SCHEME     = "Mark Scheme"
    EXAMINER_REPORT = "Examiner Report"
    SYLLABUS        = "Syllabus"
    SPECIMEN_PAPER  = "Specimen Paper"
    DATESHEET       = "Datesheet"
    NOTES           = "Notes"

class Session(str, Enum):
    FEB_MAR = "Feb-Mar"
    MAY_JUN = "May-Jun"
    OCT_NOV = "Oct-Nov"

class SourceType(str, Enum):
    OFFICIAL  = "Official"
    COMMUNITY = "Community"

class VerifiedStatus(str, Enum):
    VERIFIED   = "verified"
    UNVERIFIED = "unverified"
    FLAGGED    = "flagged"


# ─── Core Resource Model ──────────────────────────────────────────────────────

class AARIResource(BaseModel):
    """
    Single educational resource — maps directly to a Firestore document
    under /resources/{uid}.

    Firestore Security Rules compatibility:
        isOwner:      uid is set server-side; client writes require auth.
        validUserDoc: all required fields must be present and non-empty.
    """

    # Identity
    uid:            str = Field(default_factory=lambda: str(uuid4()))

    # Classification
    board:          Board
    qualification:  Qualification
    subject_code:   str   = Field(..., min_length=3, max_length=12,
                                  description="e.g. '0580', '9709', 'MATH HL'")
    subject_name:   str   = Field(..., min_length=2, max_length=120)
    resource_type:  ResourceType
    year:           int   = Field(..., ge=2000, le=2030)
    session:        Optional[Session]   = None
    paper_number:   Optional[str]       = None   # "1", "2", "3", "4"
    variant:        Optional[str]       = None   # "1", "2", "3"

    # Provenance
    source_type:    SourceType
    source_url:     str   = Field(..., description="Page URL where resource was discovered")
    file_url:       str   = Field(..., description="Direct download / CDN link")

    # Integrity
    sha256:         Optional[str] = None   # populated after download
    file_size_bytes: Optional[int] = None

    # Axon Protocol fields
    verified_status: VerifiedStatus = VerifiedStatus.UNVERIFIED
    quality_score:   int = Field(default=0, ge=0, le=100)
    crawled_at:      str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @field_validator("year")
    @classmethod
    def year_must_be_plausible(cls, v: int) -> int:
        if v < 1990 or v > 2030:
            raise ValueError(f"Year {v} is outside plausible range")
        return v

    @field_validator("sha256")
    @classmethod
    def sha256_format(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and len(v) != 64:
            raise ValueError("sha256 must be a 64-character hex string")
        return v

    def compute_sha256(self, raw_bytes: bytes) -> "AARIResource":
        """Attach SHA-256 digest after file download."""
        self.sha256 = hashlib.sha256(raw_bytes).hexdigest()
        self.file_size_bytes = len(raw_bytes)
        return self

    def to_firestore_dict(self) -> dict:
        """Serialise to a flat dict matching Firestore document schema."""
        d = self.model_dump(mode="json")
        # Firestore doesn't accept Python Enum objects — already str via mode="json"
        return d

    model_config = {"use_enum_values": True}


# ─── Crawl Session Metadata ───────────────────────────────────────────────────

class CrawlSession(BaseModel):
    """Metadata block written to manifest.lock."""
    session_id:     str = Field(default_factory=lambda: str(uuid4()))
    started_at:     str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    completed_at:   Optional[str] = None
    total_found:    int = 0
    total_new:      int = 0
    total_dupes:    int = 0
    total_errors:   int = 0
    sources_crawled: list[str] = Field(default_factory=list)
    errors:          list[dict] = Field(default_factory=list)

    def close(self) -> "CrawlSession":
        self.completed_at = datetime.now(timezone.utc).isoformat()
        return self


# ─── Manifest Lock Entry ──────────────────────────────────────────────────────

class ManifestEntry(BaseModel):
    """One entry in manifest.lock — used for idempotent re-sync."""
    uid:        str
    sha256:     str
    file_url:   str
    resource_type: str
    crawled_at: str
    synced:     bool = False
