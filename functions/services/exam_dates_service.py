from __future__ import annotations

import hashlib
import asyncio
import httpx
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup


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
    # PDF functionality has been removed as per user request
    # Return empty string to maintain compatibility but disable PDF text extraction
    return ""


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

        print(f"[ExamDates] sync_user_deadlines: user={user_id}, board={board}, subjects={subjects}, year={target_year}, series={target_series}, zone={target_zone}")

        deadlines_ref = (
            self._db.collection("users_private").document(user_id).collection("deadlines")
        )
        sync_meta_ref = (
            self._db.collection("users_private").document(user_id).collection("_sync_meta").document("exam_dates")
        )

        sync_key = hashlib.sha1(
            json.dumps([user_id, board, sorted(subjects), target_year, target_series, target_zone], sort_keys=True).encode("utf-8")
        ).hexdigest()

        sync_meta = sync_meta_ref.get()
        if sync_meta.exists:
            meta = sync_meta.to_dict() or {}
            if meta.get("sync_key") == sync_key:
                print(f"[ExamDates] Sync meta matches (key={sync_key[:12]}...), skipping scrape")
                existing_snapshots = list(deadlines_ref.stream())
                existing_subjects = {str((s.to_dict() or {}).get("subject", "")) for s in existing_snapshots}
                stale = existing_subjects - set(subjects)
                if stale:
                    for snap in existing_snapshots:
                        data = snap.to_dict() or {}
                        if str(data.get("subject", "")) in stale:
                            try:
                                snap.reference.delete()
                            except AttributeError:
                                deadlines_ref.document(snap.id).delete()
                    print(f"[ExamDates] Removed {len(stale)} stale subjects: {stale}")
                return {
                    "board": self._canonical_board(board),
                    "subjects": subjects,
                    "year": target_year,
                    "series": target_series,
                    "administrative_zone": target_zone,
                    "persisted_count": 0,
                    "cached": True,
                    "events": [],
                }

        existing_snapshots = list(deadlines_ref.stream())

        try:
            events = self.fetch_official_exam_dates(
                board=board,
                subjects=subjects,
                year=target_year,
                series=target_series,
                administrative_zone=target_zone,
            )
        except Exception as e:
            print(f"[ExamDates] Exception during fetch_official_exam_dates: {e}")
            events = []

        print(f"[ExamDates] Fetched {len(events)} events")

        persisted = 0
        if persist:
            try:
                for snap in existing_snapshots:
                    data = snap.to_dict() or {}
                    if data.get("source_type") == "OFFICIAL_DATESHEET_SCRAPER":
                        try:
                            snap.reference.delete()
                        except AttributeError:
                            deadlines_ref.document(snap.id).delete()

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

                sync_meta_ref.set({
                    "sync_key": sync_key,
                    "subject_count": len(subjects),
                    "event_count": len(events),
                    "synced_at": _utc_now(),
                })
                print(f"[ExamDates] Persisted {persisted} deadlines to Firestore (replaced {len(existing_snapshots)} old)")
            except Exception as e:
                print(f"[ExamDates] Failed to persist deadlines: {e}")

        return {
            "board": self._canonical_board(board),
            "subjects": subjects,
            "year": target_year,
            "series": target_series,
            "administrative_zone": target_zone,
            "persisted_count": persisted,
            "cached": False,
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

        # Parse canonical board into board_id + level
        parts = canonical_board.split("_", 1)
        board_id = parts[0] if len(parts) > 0 else canonical_board
        level = parts[1] if len(parts) > 1 else "igcse"

        # Cambridge International (CIE) — IGCSE, AS Level, A Level
        # Scrapes from cambridgeinternational.org/timetables
        if board_id == "caie":
            return self._scrape_cambridge_dates(
                board=canonical_board,
                subjects=clean_subjects,
                year=year,
                series=series,
                administrative_zone=administrative_zone,
            )

        print(f"[ExamDates] Unrecognized board: '{canonical_board}' (original: '{board}')")
        return []

    def _default_series(self) -> str:
        month = datetime.now(timezone.utc).month
        # Cambridge: March (India only), May/June, October/November
        if month <= 2:
            return "march"
        if month <= 7:
            return "may"
        return "november"

    def _canonical_board(self, board: str) -> str:
        """
        Two-step board resolution:
          1. Extract exam board: only CAIE (Cambridge International)
          2. Extract level: igcse, as_level, a_level

        Returns a combined canonical identifier like 'caie_igcse', 'caie_a_level', etc.
        """
        normalized = _normalize(board)

        # Step 1: Identify the exam board — only CAIE
        board_id = None
        if "caie" in normalized or "cambridge" in normalized or "cie" in normalized:
            board_id = "caie"

        # Step 2: Identify the qualification level
        if "as level" in normalized and "a level" not in normalized:
            level = "as_level"
        elif "a level" in normalized or "alevel" in normalized or "ial" in normalized or "international advanced" in normalized:
            level = "a_level"
        elif "igcse" in normalized or "international gcse" in normalized:
            level = "igcse"
        elif "o level" in normalized or "olevel" in normalized:
            level = "igcse"
        elif "gcse" in normalized:
            level = "igcse"
        else:
            level = "igcse"  # Default

        if board_id:
            return f"{board_id}_{level}"

        # Fallback — all unrecognized input defaults to CAIE
        if "a level" in normalized or "as level" in normalized:
            return "caie_a_level"
        return "caie_igcse"

    async def _fetch_all_pdfs(self, pdf_links: list[tuple[str, str]]) -> list[tuple[str, str, bytes | None]]:
        # Limit concurrency to 10 simultaneous downloads to prevent OOM or rate limits
        semaphore = asyncio.Semaphore(10)

        async with httpx.AsyncClient(timeout=40.0) as client:
            async def fetch_one(title: str, url: str) -> tuple[str, str, bytes | None]:
                async with semaphore:
                    try:
                        response = await client.get(url)
                        response.raise_for_status()
                        return title, url, response.content
                    except Exception as e:
                        print(f"[ExamDates] Failed to download PDF {url}: {e}")
                        return title, url, None

            tasks = [fetch_one(title, url) for title, url in pdf_links]
            return await asyncio.gather(*tasks)

    def _scrape_cambridge_dates(
        self,
        *,
        board: str,
        subjects: list[str],
        year: int,
        series: str,
        administrative_zone: str | None,
    ) -> list[ScrapedExamEvent]:
        try:
            response = requests.get(self.CAMBRIDGE_TIMETABLES_URL, timeout=30)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"[ExamDates] Failed to fetch Cambridge timetables page: {e}")
            return []

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
            elif series.startswith("mar"):
                if "march" not in normalized_text:
                    continue
            if administrative_zone == "uk":
                if "uk" not in normalized_text:
                    continue
            elif administrative_zone and administrative_zone not in normalized_text.replace(" ", ""):
                continue
            pdf_links.append((text, urljoin(self.CAMBRIDGE_TIMETABLES_URL, href)))

        if not pdf_links:
            print(f"[ExamDates] No Cambridge PDF links found for year={year}, series={series}, zone={administrative_zone}")

        events: list[ScrapedExamEvent] = []
        seen: set[tuple[str, str, str]] = set()

        # Concurrent fetching of PDFs
        try:
            results = asyncio.run(self._fetch_all_pdfs(pdf_links))
        except RuntimeError as e:
            # If we're already running in an event loop (e.g. from FastAPI), we can't use asyncio.run
            # We must gracefully fail or provide an alternative.
            print(f"[ExamDates] RuntimeError executing concurrent fetches (likely already in an event loop): {e}")
            raise
        except Exception as e:
            print(f"[ExamDates] Failed to execute concurrent fetches: {e}")
            raise

        for title, pdf_url, content in results:
            if content is None:
                continue
            try:
                text = _extract_pdf_text(content)
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
            except Exception as e:
                print(f"[ExamDates] Failed to parse PDF {pdf_url}: {e}")
                continue
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
