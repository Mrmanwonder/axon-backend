from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class CurriculumCatalogService:
    def __init__(self, catalog_path: Path | None = None) -> None:
        self._catalog_path = catalog_path or (
            Path(__file__).resolve().parents[2] / "data" / "curriculum_catalog.json"
        )
        self._cache: dict[str, Any] | None = None

    def _load(self) -> dict[str, Any]:
        if self._cache is not None:
            return self._cache
        with self._catalog_path.open("r", encoding="utf-8") as handle:
            self._cache = json.load(handle)
        return self._cache

    def find_board(self, raw_board: str) -> dict[str, Any] | None:
        normalized = self._normalize(raw_board)
        boards = self._load().get("boards", [])
        for board in boards:
            if self._normalize(board.get("id", "")) == normalized:
                return board
            if self._normalize(board.get("label", "")) == normalized:
                return board
            for alias in board.get("aliases", []):
                if self._normalize(str(alias)) == normalized:
                    return board
        if "o level" in normalized or "olevel" in normalized:
            return self.find_board("caie_o_level")
        if "a level" in normalized or "as level" in normalized:
            return self.find_board("caie_a_level")
        if "igcse" in normalized or "cambridge" in normalized:
            return self.find_board("caie_igcse")
        return None

    def find_subject(self, board: str, subject: str) -> dict[str, Any] | None:
        board_entry = self.find_board(board)
        if board_entry is None:
            return None
        normalized = self._normalize(subject)
        for item in board_entry.get("subjects", []):
            if self._normalize(item.get("name", "")) == normalized:
                return item
            for alias in item.get("aliases", []):
                if self._normalize(str(alias)) == normalized:
                    return item
        return None

    def list_available(self) -> dict[str, Any]:
        boards = self._load().get("boards", [])
        return {
            "boards": [
                {
                    "id": board.get("id", ""),
                    "label": board.get("label", ""),
                    "subject_count": len(board.get("subjects", [])),
                    "subjects": [
                        item.get("name", "") for item in board.get("subjects", [])
                    ],
                }
                for board in boards
            ]
        }

    def syllabus_payload(self, board: str, subject: str) -> dict[str, Any] | None:
        board_entry = self.find_board(board)
        subject_entry = self.find_subject(board, subject)
        if board_entry is None or subject_entry is None:
            return None
        return {
            "board": board_entry.get("label", board),
            "board_id": board_entry.get("id", ""),
            "subject": subject_entry.get("name", subject),
            "aliases": subject_entry.get("aliases", []),
            "chapters": subject_entry.get("chapters", []),
            "chapter_count": len(subject_entry.get("chapters", [])),
        }

    def _normalize(self, value: str) -> str:
        return " ".join(value.strip().lower().split())
