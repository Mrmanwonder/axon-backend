"""
AARI — Unit Tests
Run with: pytest tests/ -v
"""

import hashlib
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

# ── Deduplication ─────────────────────────────────────────────────────────────
from aari.core.dedup import DeduplicationEngine


def test_dedup_url_first_visit():
    with tempfile.TemporaryDirectory() as td:
        eng = DeduplicationEngine(Path(td) / "cache.json")
        assert eng.check_and_register_url("https://example.com/a.pdf") is False


def test_dedup_url_second_visit():
    with tempfile.TemporaryDirectory() as td:
        eng = DeduplicationEngine(Path(td) / "cache.json")
        eng.check_and_register_url("https://example.com/a.pdf")
        assert eng.check_and_register_url("https://example.com/a.pdf") is True


def test_dedup_content_new():
    with tempfile.TemporaryDirectory() as td:
        eng = DeduplicationEngine(Path(td) / "cache.json")
        raw = b"%PDF-fake content"
        is_dupe, digest = eng.check_and_register_content(raw)
        assert is_dupe is False
        assert digest == hashlib.sha256(raw).hexdigest()


def test_dedup_content_duplicate():
    with tempfile.TemporaryDirectory() as td:
        eng = DeduplicationEngine(Path(td) / "cache.json")
        raw = b"%PDF-same content here"
        eng.check_and_register_content(raw)
        is_dupe, _ = eng.check_and_register_content(raw)
        assert is_dupe is True


def test_dedup_persistence():
    with tempfile.TemporaryDirectory() as td:
        cache = Path(td) / "cache.json"
        eng1  = DeduplicationEngine(cache)
        eng1.check_and_register_url("https://example.com/b.pdf")
        eng1.save()

        eng2 = DeduplicationEngine(cache)
        assert eng2.check_and_register_url("https://example.com/b.pdf") is True


# ── PDF Validator ─────────────────────────────────────────────────────────────
from aari.validators.pdf_validator import PDFValidator, _is_pdf_bytes


def test_pdf_magic_valid():
    assert _is_pdf_bytes(b"%PDF-1.4 content") is True


def test_pdf_magic_html_page():
    assert _is_pdf_bytes(b"<!DOCTYPE html><html>") is False


def test_validator_rejects_html():
    v   = PDFValidator()
    # Pad to exceed the 64-byte floor so we reach the HTML detection branch
    payload = b"<html><body>" + b"Login page - please sign in to continue. " * 5 + b"</body></html>"
    res = v.validate(payload, "https://example.com/file.pdf")
    assert res.is_valid is False
    assert res.error is not None


def test_validator_rejects_tiny():
    v   = PDFValidator()
    res = v.validate(b"%PDF", "https://example.com/file.pdf")
    assert res.is_valid is False


# ── Quality Scorer ────────────────────────────────────────────────────────────
from aari.validators.quality import compute_quality_score
from aari.models.resource import (
    AARIResource, Board, Qualification, ResourceType,
    Session, SourceType, VerifiedStatus,
)


def _make_resource(**overrides) -> AARIResource:
    defaults = dict(
        board          = Board.CAIE,
        qualification  = Qualification.IGCSE,
        subject_code   = "0580",
        subject_name   = "Mathematics",
        resource_type  = ResourceType.PAST_PAPER,
        year           = 2024,
        session        = Session.MAY_JUN,
        source_type    = SourceType.OFFICIAL,
        source_url     = "https://cambridgeinternational.org/test",
        file_url       = "https://cambridgeinternational.org/test/0580_s24_qp_1.pdf",
        verified_status= VerifiedStatus.VERIFIED,
    )
    defaults.update(overrides)
    return AARIResource(**defaults)


def test_quality_official_source():
    r     = _make_resource()
    score = compute_quality_score(r, [], pdf_quality_sig=10)
    assert score >= 40   # at least official source weight


def test_quality_community_lower():
    r_off  = _make_resource(source_type=SourceType.OFFICIAL)
    r_comm = _make_resource(source_type=SourceType.COMMUNITY)
    s_off  = compute_quality_score(r_off,  [], pdf_quality_sig=5)
    s_comm = compute_quality_score(r_comm, [], pdf_quality_sig=5)
    assert s_off > s_comm


def test_quality_paired_ms_boost():
    paper = _make_resource(resource_type=ResourceType.PAST_PAPER)
    ms    = _make_resource(resource_type=ResourceType.MARK_SCHEME)
    without = compute_quality_score(paper, [],     pdf_quality_sig=0)
    with_ms = compute_quality_score(paper, [ms],   pdf_quality_sig=0)
    assert with_ms > without


def test_quality_max_100():
    r  = _make_resource()
    ms = _make_resource(resource_type=ResourceType.MARK_SCHEME)
    er = _make_resource(resource_type=ResourceType.EXAMINER_REPORT)
    score = compute_quality_score(r, [ms, er], pdf_quality_sig=10)
    assert score <= 100


# ── Resource Model ────────────────────────────────────────────────────────────

def test_resource_firestore_dict_has_required_fields():
    r = _make_resource()
    d = r.to_firestore_dict()
    for field in ["uid", "board", "subject_code", "resource_type",
                  "year", "source_type", "verified_status", "crawled_at"]:
        assert field in d, f"Missing Firestore field: {field}"


def test_resource_sha256_attached():
    r   = _make_resource()
    raw = b"%PDF-1.4 fake"
    r.compute_sha256(raw)
    assert r.sha256 == hashlib.sha256(raw).hexdigest()
    assert r.file_size_bytes == len(raw)


def test_resource_invalid_sha256_length():
    with pytest.raises(Exception):
        _make_resource(sha256="tooshort")


# ── Export Engine ─────────────────────────────────────────────────────────────
from aari.exporters.export import ExportEngine
from aari.models.resource import CrawlSession


def test_export_batch_upload_structure():
    with tempfile.TemporaryDirectory() as td:
        eng    = ExportEngine(Path(td))
        crawl  = CrawlSession()
        r      = _make_resource()
        r.compute_sha256(b"%PDF-fake")
        dest   = eng.write_batch_upload([r], crawl)
        data   = json.loads(dest.read_text())
        assert "meta" in data
        assert "resources" in data
        assert len(data["resources"]) == 1
        assert data["resources"][0]["uid"] == r.uid


def test_export_manifest_idempotent():
    with tempfile.TemporaryDirectory() as td:
        eng   = ExportEngine(Path(td))
        crawl = CrawlSession()
        r     = _make_resource()
        r.compute_sha256(b"%PDF-fake")

        # First write — synced=False
        eng.write_manifest([r], crawl, {})
        loaded = eng.load_manifest()
        assert loaded[r.uid].synced is False

        # Simulate sync completion — mark synced externally
        loaded[r.uid].synced = True

        # Second write — same sha256 → preserved as synced=True
        eng.write_manifest([r], crawl, loaded)
        reloaded = eng.load_manifest()
        assert reloaded[r.uid].synced is True
