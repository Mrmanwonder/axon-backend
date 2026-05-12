from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _normalize_zone(zone: str | None) -> str | None:
    if not zone:
        return None
    normalized = _normalize(zone).replace(" ", "")
    if normalized in {"uk", "zone3uk"}:
        return "uk"
    if normalized.startswith("zone"):
        return normalized
    digits = re.sub(r"[^0-9]", "", normalized)
    return f"zone{digits}" if digits else normalized


def _parse_date_from_text(value: str, *, year: int) -> str | None:
    cleaned = " ".join(value.replace("\n", " ").split())
    if not cleaned:
        return None

    candidates = [
        cleaned,
        f"{cleaned} {year}",
    ]
    for candidate in candidates:
        parsed = pd.to_datetime(candidate, errors="coerce", dayfirst=True)
        if pd.notna(parsed):
            return parsed.to_pydatetime().date().isoformat()
    return None


def _extract_pdf_text(pdf_bytes: bytes) -> str:
    from io import BytesIO

    reader = PdfReader(BytesIO(pdf_bytes))
    chunks: list[str] = []
    for page in reader.pages:
        chunks.append(page.extract_text() or "")
    return "\n".join(chunks)


@dataclass(frozen=True)
class ScrapedExamEvent:
    board: str
    subject: str
    paper: str
    label: str
    exam_date: str
    source_url: str
    source_title: str
    source_type: str = "OFFICIAL_DATESHEET_SCRAPER"

    def to_deadline_doc(self, *, administrative_zone: str | None, series: str, year: int) -> dict[str, Any]:
        return {
            "board": self.board,
            "subject": self.subject,
            "paper": self.paper,
            "label": self.label,
            "exam_date": self.exam_date,
            "source_url": self.source_url,
            "source_title": self.source_title,
            "source_type": self.source_type,
            "series": series,
            "year": year,
            "administrative_zone": administrative_zone,
            "status": "scheduled",
            "scraped_at": _utc_now(),
        }


class OfficialExamDatesService:
    CAMBRIDGE_TIMETABLES_URL = "https://www.cambridgeinternational.org/timetables"
    IB_DP_SCHEDULE_URL = (
        "https://www.ibo.org/programmes/diploma-programme/assessment-and-exams/exam-schedule/"
    )

    def __init__(self, db) -> None:
        self._db = db

    def sync_user_deadlines(
        self,
        *,
        user_id: str,
        board: str,
        subjects: list[str],
        year: int | None = None,
        series: str | None = None,
        administrative_zone: str | None = None,
        persist: bool = True,
    ) -> dict[str, Any]:
        target_year = year or datetime.now(timezone.utc).year
        target_series = (series or self._default_series()).lower()
        target_zone = _normalize_zone(administrative_zone)

        events = self.fetch_official_exam_dates(
            board=board,
            subjects=subjects,
            year=target_year,
            series=target_series,
            administrative_zone=target_zone,
        )

        persisted = 0
        if persist:
            deadlines_ref = (
                self._db.collection("users_private").document(user_id).collection("deadlines")
            )
            for event in events:
                payload = event.to_deadline_doc(
                    administrative_zone=target_zone,
                    series=target_series,
                    year=target_year,
                )
                doc_id = hashlib.sha1(
                    json.dumps(
                        [
                            user_id,
                            payload["board"],
                            payload["subject"],
                            payload["paper"],
                            payload["exam_date"],
                            payload.get("administrative_zone") or "",
                        ],
                        sort_keys=True,
                    ).encode("utf-8")
                ).hexdigest()
                deadlines_ref.document(doc_id).set(payload, merge=True)
                persisted += 1

        return {
            "board": self._canonical_board(board),
            "subjects": subjects,
            "year": target_year,
            "series": target_series,
            "administrative_zone": target_zone,
            "persisted_count": persisted,
            "events": [event.to_deadline_doc(
                administrative_zone=target_zone,
                series=target_series,
                year=target_year,
            ) for event in events],
        }

    def fetch_official_exam_dates(
        self,
        *,
        board: str,
        subjects: list[str],
        year: int,
        series: str,
        administrative_zone: str | None,
    ) -> list[ScrapedExamEvent]:
        canonical_board = self._canonical_board(board)
        clean_subjects = [subject.strip() for subject in subjects if subject.strip()]
        if not clean_subjects:
            return []

        if canonical_board in {"IGCSE", "A_LEVEL"}:
            return self._scrape_cambridge_dates(
                board=canonical_board,
                subjects=clean_subjects,
                year=year,
                series=series,
                administrative_zone=administrative_zone,
            )
        if canonical_board == "IB":
            return self._scrape_ib_dates(
                board=canonical_board,
                subjects=clean_subjects,
                year=year,
                series=series,
            )
        return []

    def _default_series(self) -> str:
        month = datetime.now(timezone.utc).month
        if month <= 6:
            return "may"
        return "november"

    def _canonical_board(self, board: str) -> str:
        normalized = _normalize(board)
        if "ib" == normalized or "international baccalaureate" in normalized:
            return "IB"
        if "ocr" in normalized or "edexcel" in normalized:
            return "UNSUPPORTED"
        if "a level" in normalized or "as level" in normalized:
            return "A_LEVEL"
        return "IGCSE"

    def _scrape_cambridge_dates(
        self,
        *,
        board: str,
        subjects: list[str],
        year: int,
        series: str,
        administrative_zone: str | None,
    ) -> list[ScrapedExamEvent]:
        response = requests.get(self.CAMBRIDGE_TIMETABLES_URL, timeout=30)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        pdf_links: list[tuple[str, str]] = []
        for link in soup.find_all("a", href=True):
            href = link["href"]
            text = " ".join(link.get_text(" ", strip=True).split())
            if ".pdf" not in href.lower():
                continue
            normalized_text = _normalize(text)
            if str(year) not in text:
                continue
            if series.startswith("may") or series.startswith("jun"):
                if "june" not in normalized_text and "may" not in normalized_text:
                    continue
            elif series.startswith("nov"):
                if "november" not in normalized_text:
                    continue
            if administrative_zone == "uk":
                if "uk" not in normalized_text:
                    continue
            elif administrative_zone and administrative_zone not in normalized_text.replace(" ", ""):
                continue
            pdf_links.append((text, urljoin(self.CAMBRIDGE_TIMETABLES_URL, href)))

        events: list[ScrapedExamEvent] = []
        seen: set[tuple[str, str, str]] = set()
        for title, pdf_url in pdf_links:
            pdf_response = requests.get(pdf_url, timeout=40)
            pdf_response.raise_for_status()
            text = _extract_pdf_text(pdf_response.content)
            events.extend(
                self._parse_timetable_text(
                    board=board,
                    subjects=subjects,
                    text=text,
                    source_url=pdf_url,
                    source_title=title,
                    year=year,
                    seen=seen,
                )
            )
        return events

    def _scrape_ib_dates(
        self,
        *,
        board: str,
        subjects: list[str],
        year: int,
        series: str,
    ) -> list[ScrapedExamEvent]:
        response = requests.get(self.IB_DP_SCHEDULE_URL, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        target_series = "may" if series.startswith("may") else "november"
        pdf_links: list[tuple[str, str]] = []
        for link in soup.find_all("a", href=True):
            href = link["href"]
            text = " ".join(link.get_text(" ", strip=True).split())
            normalized_text = _normalize(text)
            if ".pdf" not in href.lower():
                continue
            if str(year) not in text:
                continue
            if target_series not in normalized_text:
                continue
            pdf_links.append((text, urljoin(self.IB_DP_SCHEDULE_URL, href)))

        events: list[ScrapedExamEvent] = []
        seen: set[tuple[str, str, str]] = set()
        for title, pdf_url in pdf_links:
            pdf_response = requests.get(pdf_url, timeout=40)
            pdf_response.raise_for_status()
            text = _extract_pdf_text(pdf_response.content)
            events.extend(
                self._parse_timetable_text(
                    board=board,
                    subjects=subjects,
                    text=text,
                    source_url=pdf_url,
                    source_title=title,
                    year=year,
                    seen=seen,
                )
            )
        return events

    def _parse_timetable_text(
        self,
        *,
        board: str,
        subjects: list[str],
        text: str,
        source_url: str,
        source_title: str,
        year: int,
        seen: set[tuple[str, str, str]],
    ) -> list[ScrapedExamEvent]:
        subject_aliases = {subject: _normalize(subject) for subject in subjects}
        lines = [" ".join(line.split()) for line in text.splitlines()]
        events: list[ScrapedExamEvent] = []
        current_date: str | None = None

        date_patterns = [
            re.compile(r"\b\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}\b"),
            re.compile(r"\b\d{1,2}\s+[A-Za-z]{3,9}\b"),
        ]
        paper_patterns = [
            re.compile(r"\bpaper\s+(\d+[A-Z]?)\b", re.IGNORECASE),
            re.compile(r"\bp(\d+[A-Z]?)\b", re.IGNORECASE),
        ]

        for line in lines:
            if not line:
                continue

            for pattern in date_patterns:
                match = pattern.search(line)
                if match:
                    parsed_date = _parse_date_from_text(match.group(0), year=year)
                    if parsed_date:
                        current_date = parsed_date
                    break

            normalized_line = _normalize(line)
            if current_date is None:
                continue

            for subject, alias in subject_aliases.items():
                if alias not in normalized_line:
                    continue
                paper = "Main Paper"
                for paper_pattern in paper_patterns:
                    paper_match = paper_pattern.search(line)
                    if paper_match:
                        paper = f"Paper {paper_match.group(1).upper()}"
                        break

                event_key = (subject, paper, current_date)
                if event_key in seen:
                    continue
                seen.add(event_key)
                events.append(
                    ScrapedExamEvent(
                        board=board,
                        subject=subject,
                        paper=paper,
                        label=line[:220],
                        exam_date=current_date,
                        source_url=source_url,
                        source_title=source_title,
                    )
                )
        return events
