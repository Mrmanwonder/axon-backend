"""
AARI — Demo / Smoke-Test Script
Generates realistic sample output files (batch_upload.json, manifest.lock)
using synthetic data so you can inspect the exact Firestore payload
shape before running a live crawl.

Run:  python demo.py
"""

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).parent))

from aari.models.resource import (
    AARIResource, Board, CrawlSession, ManifestEntry,
    Qualification, ResourceType, Session, SourceType, VerifiedStatus,
)
from aari.exporters.export import ExportEngine
from aari.validators.quality import compute_quality_score
from aari.config import OUTPUT_DIR


# ── Synthetic resource factory ────────────────────────────────────────────────

def _fake_bytes(seed: str) -> bytes:
    """Deterministic fake 'PDF' bytes for hashing."""
    return (f"%PDF-1.4 AARI_DEMO_{seed}").encode() + b"\x00" * 128


def make_resource(
    subject_code: str,
    subject_name: str,
    board: Board,
    qual: Qualification,
    rtype: ResourceType,
    year: int,
    session: Session,
    source_type: SourceType,
    paper: str = "1",
    variant: str = "1",
) -> AARIResource:
    base_url = {
        Board.CAIE:    "https://www.cambridgeinternational.org",
        Board.IBO:     "https://www.ibo.org",
        Board.Edexcel: "https://qualifications.pearson.com",
    }[board]

    sess_slug = session.replace("-", "_").lower() if session else "specimen"
    rtype_slug = {
        ResourceType.PAST_PAPER:      "qp",
        ResourceType.MARK_SCHEME:     "ms",
        ResourceType.EXAMINER_REPORT: "er",
        ResourceType.SYLLABUS:        "syllabus",
        ResourceType.DATESHEET:       "datesheet",
        ResourceType.SPECIMEN_PAPER:  "sp",
    }.get(rtype, "qp")

    filename  = f"{subject_code}_{str(year)[2:]}_{sess_slug}_{rtype_slug}_{paper}{variant}.pdf"
    file_url  = f"{base_url}/past-papers/{subject_code}/{filename}"
    page_url  = f"{base_url}/past-papers/{subject_code}/"

    r = AARIResource(
        board          = board,
        qualification  = qual,
        subject_code   = subject_code,
        subject_name   = subject_name,
        resource_type  = rtype,
        year           = year,
        session        = session,
        paper_number   = paper,
        variant        = variant,
        source_type    = source_type,
        source_url     = page_url,
        file_url       = file_url,
        verified_status= VerifiedStatus.VERIFIED if source_type == SourceType.OFFICIAL else VerifiedStatus.UNVERIFIED,
        crawled_at     = datetime.now(timezone.utc).isoformat(),
    )

    raw = _fake_bytes(filename)
    r.compute_sha256(raw)
    return r


# ── Build a rich synthetic dataset ───────────────────────────────────────────

def build_demo_resources() -> list[AARIResource]:
    resources = []

    # CAIE IGCSE Mathematics — full set: QP + MS + ER, multiple years & sessions
    for year in [2023, 2024, 2025]:
        for sess in [Session.MAY_JUN, Session.OCT_NOV]:
            for paper in ["1", "2", "4"]:
                for variant in ["1", "2"]:
                    qp = make_resource(
                        "0580", "Mathematics", Board.CAIE, Qualification.IGCSE,
                        ResourceType.PAST_PAPER, year, sess, SourceType.OFFICIAL,
                        paper, variant,
                    )
                    ms = make_resource(
                        "0580", "Mathematics", Board.CAIE, Qualification.IGCSE,
                        ResourceType.MARK_SCHEME, year, sess, SourceType.OFFICIAL,
                        paper, variant,
                    )
                    resources.extend([qp, ms])

    # CAIE IGCSE Mathematics — Examiner Reports (one per year)
    for year in [2023, 2024]:
        er = make_resource(
            "0580", "Mathematics", Board.CAIE, Qualification.IGCSE,
            ResourceType.EXAMINER_REPORT, year, Session.MAY_JUN,
            SourceType.OFFICIAL, "0", "0",
        )
        resources.append(er)

    # CAIE A-Level Physics — official source
    for year in [2024, 2025]:
        for sess in [Session.MAY_JUN]:
            for paper in ["1", "2", "3", "4", "5"]:
                qp = make_resource(
                    "9702", "Physics", Board.CAIE, Qualification.A_LEVEL,
                    ResourceType.PAST_PAPER, year, sess, SourceType.OFFICIAL,
                    paper, "1",
                )
                ms = make_resource(
                    "9702", "Physics", Board.CAIE, Qualification.A_LEVEL,
                    ResourceType.MARK_SCHEME, year, sess, SourceType.OFFICIAL,
                    paper, "1",
                )
                resources.extend([qp, ms])

    # CAIE A-Level Physics — Syllabus
    syllabus = make_resource(
        "9702", "Physics", Board.CAIE, Qualification.A_LEVEL,
        ResourceType.SYLLABUS, 2025, None, SourceType.OFFICIAL,
        "0", "0",
    )
    resources.append(syllabus)

    # IB Mathematics AA HL — official + community mix
    for year in [2023, 2024]:
        for sess in [Session.MAY_JUN, Session.OCT_NOV]:
            qp = make_resource(
                "MATH-AA-HL", "Math Analysis & Approaches HL",
                Board.IBO, Qualification.IB_DP,
                ResourceType.PAST_PAPER, year, sess, SourceType.OFFICIAL,
                "1", "1",
            )
            ms = make_resource(
                "MATH-AA-HL", "Math Analysis & Approaches HL",
                Board.IBO, Qualification.IB_DP,
                ResourceType.MARK_SCHEME, year, sess, SourceType.COMMUNITY,
                "1", "1",
            )
            resources.extend([qp, ms])

    # Edexcel A-Level Chemistry — community source
    for year in [2023, 2024]:
        qp = make_resource(
            "YCH01", "Chemistry", Board.Edexcel, Qualification.A_LEVEL,
            ResourceType.PAST_PAPER, year, Session.MAY_JUN, SourceType.COMMUNITY,
            "1", "1",
        )
        resources.append(qp)

    # 2026 Datesheet — official only
    ds = make_resource(
        "DATESHEET", "2026 Session Timetable", Board.CAIE, Qualification.A_LEVEL,
        ResourceType.DATESHEET, 2026, Session.MAY_JUN, SourceType.OFFICIAL,
        "0", "0",
    )
    resources.append(ds)

    return resources


# ── Score all resources ───────────────────────────────────────────────────────

def score_all(resources: list[AARIResource]) -> list[AARIResource]:
    for r in resources:
        r.quality_score = compute_quality_score(r, resources, pdf_quality_sig=8)
    return resources


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("AARI Demo — generating sample output files …\n")

    resources = build_demo_resources()
    resources = score_all(resources)

    crawl = CrawlSession(
        total_found  = len(resources),
        total_new    = len(resources),
        total_dupes  = 4,   # simulated
        total_errors = 0,
        sources_crawled = ["cambridge", "ibo", "pearson", "papacambridge", "gceguide"],
    )
    crawl.close()

    exporter = ExportEngine(OUTPUT_DIR)
    batch_path    = exporter.write_batch_upload(resources, crawl)
    manifest_path = exporter.write_manifest(resources, crawl, {})

    # ── Pretty-print a summary ────────────────────────────────────────────
    by_board = {}
    by_rtype = {}
    by_source = {}
    for r in resources:
        by_board[r.board]         = by_board.get(r.board, 0) + 1
        by_rtype[r.resource_type] = by_rtype.get(r.resource_type, 0) + 1
        by_source[r.source_type]  = by_source.get(r.source_type, 0) + 1

    scores = [r.quality_score for r in resources]

    print(f"  Total resources   : {len(resources)}")
    print(f"  Session ID        : {crawl.session_id}")
    print()
    print("  By board:")
    for k, v in sorted(by_board.items()):       print(f"    {k:<12} {v}")
    print("  By resource type:")
    for k, v in sorted(by_rtype.items()):       print(f"    {k:<22} {v}")
    print("  By source type:")
    for k, v in sorted(by_source.items()):      print(f"    {k:<12} {v}")
    print()
    print(f"  Quality scores    : min={min(scores)}  avg={sum(scores)//len(scores)}  max={max(scores)}")
    print()
    print(f"  batch_upload.json -> {batch_path}")
    print(f"  manifest.lock     -> {manifest_path}")
    print()

    # ── Inline preview of first resource ─────────────────────────────────
    sample = resources[0].to_firestore_dict()
    print("  Sample Firestore document (first resource):")
    print("  " + json.dumps(sample, indent=4).replace("\n", "\n  "))


if __name__ == "__main__":
    main()
