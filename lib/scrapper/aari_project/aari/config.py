"""
AARI — Axon Autonomous Resource Intelligence
Configuration & Constants
"""

from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).parent
OUTPUT_DIR     = BASE_DIR / "output"
CACHE_DIR      = BASE_DIR / ".cache"
LOG_DIR        = BASE_DIR / "logs"

OUTPUT_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# ─── Crawl Targets ────────────────────────────────────────────────────────────
# Only sources whose robots.txt / ToS permit automated access or public APIs.
OFFICIAL_SOURCES = {
    "cambridge": {
        "name": "Cambridge Assessment International Education",
        "base_url": "https://www.cambridgeinternational.org",
        "syllabi_path": "/programmes-and-qualifications/cambridge-igcse/",
        "past_papers_path": "/programmes-and-qualifications/cambridge-igcse/past-papers/",
        "board": "CAIE",
        "respect_robots": True,
    },
    "ibo": {
        "name": "International Baccalaureate Organization",
        "base_url": "https://www.ibo.org",
        "syllabi_path": "/programmes/diploma-programme/curriculum/",
        "board": "IBO",
        "respect_robots": True,
    },
    "pearson": {
        "name": "Pearson Edexcel",
        "base_url": "https://qualifications.pearson.com",
        "syllabi_path": "/en/qualifications/edexcel-international-advanced-levels/",
        "board": "Edexcel",
        "respect_robots": True,
    },
}

# Community sources that publicly allow indexing / linking
COMMUNITY_SOURCES = {
    "papacambridge": {
        "name": "PapaCambridge",
        "base_url": "https://papacambridge.com",
        "board_map": {"CAIE": "/o-level/", "A-Level": "/a-and-as-level/"},
        "source_type": "Community",
        "respect_robots": True,
    },
    "gceguide": {
        "name": "GCE Guide",
        "base_url": "https://gceguide.com",
        "board_map": {"CAIE": "/cambridge-a-and-as-level/"},
        "source_type": "Community",
        "respect_robots": True,
    },
}

# ─── Crawl Behaviour ──────────────────────────────────────────────────────────
@dataclass
class CrawlConfig:
    max_concurrency: int          = 5
    request_delay_min: float      = 1.5    # seconds — polite floor
    request_delay_max: float      = 4.0    # seconds — polite ceiling
    request_timeout:  int         = 30
    max_retries:      int         = 3
    retry_backoff:    float       = 2.0    # exponential base
    respect_robots:   bool        = True
    user_agent: str = (
        "AARI-EducationalBot/1.0 "
        "(+https://axon.app/bot; research@axon.app)"
    )
    max_depth: int = 4
    sessions: List[str] = field(default_factory=lambda: [
        "2023", "2024", "2025", "2026"
    ])

CRAWL_CONFIG = CrawlConfig()

# ─── Quality Scoring Weights ──────────────────────────────────────────────────
QUALITY_WEIGHTS = {
    "is_official":      40,   # Official board source
    "has_mark_scheme":  20,   # Paired MS available
    "has_er":           15,   # Examiner Report present
    "year_recency":     15,   # Newer = higher score (normalised to 0-15)
    "file_valid":       10,   # PDF is well-formed, non-empty
}

# ─── Supported Qualifications ─────────────────────────────────────────────────
BOARDS    = ["CAIE", "IBO", "Edexcel"]
QUAL_TYPES = ["IGCSE", "O-Level", "A-Level", "AS-Level", "IB-DP"]
RESOURCE_TYPES = [
    "Past Paper", "Mark Scheme", "Examiner Report",
    "Syllabus", "Specimen Paper", "Datesheet", "Notes",
]
SESSIONS  = ["Feb-Mar", "May-Jun", "Oct-Nov"]

# ─── Firestore Schema Field Names ─────────────────────────────────────────────
FIRESTORE_FIELDS = [
    "uid", "board", "qualification", "subject_code", "subject_name",
    "resource_type", "year", "session", "paper_number", "variant",
    "source_type", "source_url", "file_url", "sha256",
    "verified_status", "quality_score", "crawled_at", "file_size_bytes",
]
