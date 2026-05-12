from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any


def _parse_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    raw = str(value).strip()
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


class PredictiveExamRiskService:
    def __init__(self, db) -> None:
        self._db = db

    def identify_high_yield_topics(
        self,
        syllabus_json: list[dict[str, Any]],
        *,
        limit: int = 12,
    ) -> list[dict[str, Any]]:
        ranked: list[dict[str, Any]] = []
        for node in syllabus_json:
            objective_id = str(
                node.get("objective_id")
                or node.get("code")
                or node.get("id")
                or ""
            ).strip()
            if not objective_id:
                continue
            explicit_hits = float(
                node.get("past_paper_frequency")
                or node.get("past_paper_hits")
                or node.get("historical_frequency")
                or 0.0
            )
            paper_weight = float(node.get("paper_weight", 0.0) or 0.0)
            command_word_bonus = min(
                0.15,
                len(node.get("command_words", []) or []) * 0.03,
            )
            yield_score = explicit_hits + (paper_weight * 100.0) + (command_word_bonus * 100.0)
            ranked.append(
                {
                    "objective_id": objective_id,
                    "topic": str(node.get("topic", "")).strip(),
                    "sub_topic": str(node.get("sub_topic", "")).strip(),
                    "paper": str(node.get("paper", "")).strip(),
                    "paper_weight": round(paper_weight, 4),
                    "past_paper_frequency": explicit_hits,
                    "yield_score": round(yield_score, 2),
                }
            )

        ranked.sort(key=lambda item: item["yield_score"], reverse=True)
        return ranked[:limit]

    def analyze_user(self, user_id: str, *, subject: str | None = None) -> dict[str, Any]:
        user_ref = self._db.collection("users_private").document(user_id)
        deadlines = list(user_ref.collection("deadlines").stream())
        now = datetime.now(timezone.utc)

        ranked_deadlines: list[tuple[datetime, dict[str, Any]]] = []
        for snapshot in deadlines:
            data = snapshot.to_dict() or {}
            exam_at = _parse_datetime(data.get("exam_date"))
            if exam_at is None or exam_at.date() < now.date():
                continue
            if subject and str(data.get("subject", "")).strip().lower() != subject.strip().lower():
                continue
            ranked_deadlines.append((exam_at, data))
        ranked_deadlines.sort(key=lambda item: item[0])
        if not ranked_deadlines:
            return {
                "readiness_score": 0.0,
                "projected_grade": "Insufficient Data",
                "coverage": 0.0,
                "weighted_mock_accuracy": 0.0,
                "days_remaining": 0,
                "time_proximity_score": 0.0,
                "crisis_mode": False,
                "high_yield_topics": [],
            }

        exam_at, deadline = ranked_deadlines[0]
        target_subject = str(deadline.get("subject", "")).strip()
        board = str(deadline.get("board", "")).strip()
        days_remaining = max(0, (exam_at.date() - now.date()).days)

        syllabus_query = self._db.collection("syllabus_maps").where("subject", "==", target_subject)
        if board:
            syllabus_query = syllabus_query.where("board", "==", board)
        syllabus_docs = list(syllabus_query.stream())
        syllabus_json = [doc.to_dict() or {} for doc in syllabus_docs]

        objective_ids = {
            str(
                item.get("objective_id")
                or item.get("code")
                or item.get("id")
                or ""
            ).strip()
            for item in syllabus_json
        }
        objective_ids.discard("")
        total_objectives = max(1, len(objective_ids))

        study_events = list(user_ref.collection("study_events").stream())
        mastery_docs = list(user_ref.collection("mastery").stream())
        mock_results = list(user_ref.collection("mock_results").stream())

        touched_ids: set[str] = set()
        for snapshot in study_events:
            data = snapshot.to_dict() or {}
            objective_id = str(
                data.get("objective_id")
                or data.get("learning_objective_id")
                or data.get("topic_id")
                or ""
            ).strip()
            if objective_id:
                touched_ids.add(objective_id)
        for snapshot in mastery_docs:
            data = snapshot.to_dict() or {}
            objective_id = str(
                data.get("learning_objective_id")
                or data.get("objective_id")
                or snapshot.id
            ).strip()
            if objective_id:
                touched_ids.add(objective_id)
        for snapshot in mock_results:
            data = snapshot.to_dict() or {}
            objective_id = str(
                data.get("objective_id")
                or data.get("learning_objective_id")
                or data.get("topic_id")
                or ""
            ).strip()
            if objective_id:
                touched_ids.add(objective_id)

        coverage = min(1.0, len(touched_ids & objective_ids) / total_objectives)

        weighted_numerator = 0.0
        weighted_denominator = 0.0
        for snapshot in mock_results:
            data = snapshot.to_dict() or {}
            error_type = str(data.get("error_type", "")).strip().lower()
            if error_type == "awaiting_grading":
                continue
            recorded_at = _parse_datetime(data.get("recorded_at"))
            age_days = 45 if recorded_at is None else max(0, (now - recorded_at).days)
            recency_weight = math.exp(-age_days / 21.0)
            available_marks = float(data.get("available_marks", 0.0) or 0.0)
            if available_marks <= 0:
                continue
            accuracy = float(data.get("awarded_marks", 0.0) or 0.0) / available_marks
            weighted_numerator += accuracy * recency_weight * available_marks
            weighted_denominator += recency_weight * available_marks
        weighted_mock_accuracy = (
            weighted_numerator / weighted_denominator if weighted_denominator > 0 else 0.0
        )

        # 90 days is treated as a healthy revision runway. Nearer than that increases risk.
        time_proximity_score = max(0.0, min(1.0, days_remaining / 90.0))

        readiness_score = round(
            (
                (coverage * 0.35)
                + (weighted_mock_accuracy * 0.45)
                + (time_proximity_score * 0.20)
            )
            * 100.0,
            2,
        )
        projected_grade = self._project_grade(readiness_score)
        crisis_mode = readiness_score < 50.0
        high_yield_topics = self.identify_high_yield_topics(syllabus_json)

        return {
            "subject": target_subject,
            "board": board,
            "anchor_date": exam_at.isoformat(),
            "days_remaining": days_remaining,
            "coverage": round(coverage, 4),
            "weighted_mock_accuracy": round(weighted_mock_accuracy, 4),
            "time_proximity_score": round(time_proximity_score, 4),
            "readiness_score": readiness_score,
            "projected_grade": projected_grade,
            "crisis_mode": crisis_mode,
            "high_yield_topics": high_yield_topics,
            "high_yield_objective_ids": [
                item["objective_id"] for item in high_yield_topics if item.get("objective_id")
            ],
        }

    def _project_grade(self, readiness_score: float) -> str:
        if readiness_score >= 85:
            return "A*/7"
        if readiness_score >= 72:
            return "A/6"
        if readiness_score >= 60:
            return "B/5"
        if readiness_score >= 50:
            return "C/4"
        return "D or below"
