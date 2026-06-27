from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

try:
    import fitz
except Exception:  # pragma: no cover - optional at import time
    fitz = None


QUESTION_HEADER = re.compile(
    r"^(?:Q(?:uestion)?\.?\s*)?(\d{1,3})(?:[\.\):]|\s+-|\s)",
    re.IGNORECASE,
)
PART_HEADER = re.compile(r"^\(?([a-z])\)?[\)\.]?\s+", re.IGNORECASE)


@dataclass(frozen=True)
class LayoutLine:
    page_number: int
    text: str
    x0: float
    y0: float
    x1: float
    y1: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_number": self.page_number,
            "text": self.text,
            "x0": self.x0,
            "y0": self.y0,
            "x1": self.x1,
            "y1": self.y1,
        }


class MarkSchemeLayoutService:
    def analyze_pdf_bytes(self, pdf_bytes: bytes) -> list[dict[str, Any]]:
        # PDF functionality has been removed as per user request
        # Return empty list to maintain compatibility but disable PDF processing
        return []

    def segment_batch(
        self,
        *,
        mark_scheme_text: str,
        page_layouts: list[dict[str, Any]],
        questions: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return [
            {
                "item_id": question["item_id"],
                "snippet": self.segment_question(
                    mark_scheme_text=mark_scheme_text,
                    page_layouts=page_layouts,
                    question=question,
                ),
                "strategy": "layout-aware" if page_layouts else "text-fallback",
            }
            for question in questions
        ]

    def segment_question(
        self,
        *,
        mark_scheme_text: str,
        page_layouts: list[dict[str, Any]],
        question: dict[str, Any],
    ) -> str:
        layout_snippet = self._segment_from_layout(page_layouts, question)
        if layout_snippet.strip():
            return layout_snippet.strip()
        return self._segment_from_text(mark_scheme_text, question).strip()

    def _extract_lines(self, page, page_number: int) -> list[LayoutLine]:
        text_dict = page.get_text("dict")
        lines: list[LayoutLine] = []
        for block in text_dict.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                text = " ".join((span.get("text") or "").strip() for span in spans).strip()
                if not text:
                    continue
                xs = [span["bbox"][0] for span in spans if span.get("bbox")]
                ys = [span["bbox"][1] for span in spans if span.get("bbox")]
                x2s = [span["bbox"][2] for span in spans if span.get("bbox")]
                y2s = [span["bbox"][3] for span in spans if span.get("bbox")]
                if not xs or not ys or not x2s or not y2s:
                    continue
                lines.append(
                    LayoutLine(
                        page_number=page_number,
                        text=text,
                        x0=min(xs),
                        y0=min(ys),
                        x1=max(x2s),
                        y1=max(y2s),
                    )
                )
        lines.sort(key=lambda item: (item.page_number, item.y0, item.x0))
        return lines

    def _segment_from_layout(
        self,
        page_layouts: list[dict[str, Any]],
        question: dict[str, Any],
    ) -> str:
        if not page_layouts:
            return ""

        page_number = int(question.get("page_number") or 0)
        question_number = int(question.get("question_number") or 0)
        part_labels = [str(item).lower() for item in (question.get("parts") or []) if str(item).strip()]
        normalized_bounds = (
            (question.get("spatial_metadata") or {}).get("normalized_bounds")
            if isinstance(question.get("spatial_metadata"), dict)
            else None
        ) or {}

        candidate_lines = []
        for page in page_layouts:
            if page_number > 0 and int(page.get("page_number") or 0) < page_number:
                continue
            width = float(page.get("width") or 1.0)
            height = float(page.get("height") or 1.0)
            for raw_line in page.get("lines", []):
                line = LayoutLine(
                    page_number=int(raw_line.get("page_number") or page.get("page_number") or 0),
                    text=str(raw_line.get("text") or "").strip(),
                    x0=float(raw_line.get("x0") or 0),
                    y0=float(raw_line.get("y0") or 0),
                    x1=float(raw_line.get("x1") or 0),
                    y1=float(raw_line.get("y1") or 0),
                )
                if not line.text:
                    continue
                score = 0.0
                header_match = QUESTION_HEADER.match(line.text)
                if header_match and int(header_match.group(1)) == question_number:
                    score += 100
                if normalized_bounds:
                    score += self._geometry_bonus(line, normalized_bounds, width, height)
                if part_labels and any(self._line_matches_part(line.text, label) for label in part_labels):
                    score += 12
                candidate_lines.append((score, line))

        candidate_lines.sort(key=lambda item: item[0], reverse=True)
        if not candidate_lines or candidate_lines[0][0] <= 0:
            return ""

        anchor_line = candidate_lines[0][1]
        anchor_page = anchor_line.page_number

        linear_lines: list[LayoutLine] = []
        for page in page_layouts:
            if int(page.get("page_number") or 0) < anchor_page:
                continue
            for raw_line in page.get("lines", []):
                linear_lines.append(
                    LayoutLine(
                        page_number=int(raw_line.get("page_number") or page.get("page_number") or 0),
                        text=str(raw_line.get("text") or "").strip(),
                        x0=float(raw_line.get("x0") or 0),
                        y0=float(raw_line.get("y0") or 0),
                        x1=float(raw_line.get("x1") or 0),
                        y1=float(raw_line.get("y1") or 0),
                    )
                )
        linear_lines.sort(key=lambda item: (item.page_number, item.y0, item.x0))
        try:
            start_index = linear_lines.index(anchor_line)
        except ValueError:
            return ""

        collected: list[str] = []
        for index in range(start_index, len(linear_lines)):
            line = linear_lines[index]
            if index > start_index:
                next_match = QUESTION_HEADER.match(line.text)
                if next_match and int(next_match.group(1)) != question_number:
                    break
            if part_labels:
                if QUESTION_HEADER.match(line.text):
                    collected.append(line.text)
                    continue
                if PART_HEADER.match(line.text):
                    part_label = PART_HEADER.match(line.text).group(1).lower()
                    if part_label in part_labels:
                        collected.append(line.text)
                    elif collected:
                        break
                    continue
            collected.append(line.text)

        return "\n".join(item for item in collected if item.strip())

    def _geometry_bonus(
        self,
        line: LayoutLine,
        normalized_bounds: dict[str, Any],
        page_width: float,
        page_height: float,
    ) -> float:
        try:
            anchor_y = float(normalized_bounds.get("y", 0)) * page_height
            anchor_x = float(normalized_bounds.get("x", 0)) * page_width
        except Exception:
            return 0.0
        vertical_distance = abs(line.y0 - anchor_y)
        horizontal_distance = abs(line.x0 - anchor_x)
        return max(0.0, 35.0 - (vertical_distance / 18.0) - (horizontal_distance / 120.0))

    def _line_matches_part(self, line_text: str, label: str) -> bool:
        part_match = PART_HEADER.match(line_text)
        return bool(part_match and part_match.group(1).lower() == label.lower())

    def _segment_from_text(self, mark_scheme_text: str, question: dict[str, Any]) -> str:
        if not mark_scheme_text.strip():
            return ""
        question_number = int(question.get("question_number") or 0)
        part_labels = [str(item).lower() for item in (question.get("parts") or []) if str(item).strip()]
        lines = [line.rstrip() for line in mark_scheme_text.splitlines()]
        start_index = None
        for index, line in enumerate(lines):
            match = QUESTION_HEADER.match(line.strip())
            if match and int(match.group(1)) == question_number:
                start_index = index
                break
        if start_index is None:
            return ""

        collected: list[str] = []
        for index in range(start_index, len(lines)):
            line = lines[index].strip()
            if index > start_index:
                next_match = QUESTION_HEADER.match(line)
                if next_match and int(next_match.group(1)) != question_number:
                    break
            if part_labels:
                if QUESTION_HEADER.match(line):
                    collected.append(line)
                    continue
                if PART_HEADER.match(line):
                    part_match = PART_HEADER.match(line)
                    if part_match.group(1).lower() in part_labels:
                        collected.append(line)
                    elif collected:
                        break
                    continue
            collected.append(line)
        return "\n".join(item for item in collected if item)
