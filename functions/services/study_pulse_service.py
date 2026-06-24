from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from typing import Any

import pandas as pd


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


class StudyPulseService:
    def __init__(self, db, advisor_model=None) -> None:
        self._db = db
        self._advisor_model = advisor_model

    def analyze_user(self, user_id: str, *, session_id: str | None = None) -> dict[str, Any]:
        user_ref = self._db.collection("users_private").document(user_id)
        from services.syllabus_cache import get_all_syllabus_docs
        syllabus_docs = get_all_syllabus_docs(self._db)
        study_event_docs = list(user_ref.collection("study_events").stream())
        mock_result_docs = list(user_ref.collection("mock_results").stream())
        mastery_docs = list(user_ref.collection("mastery").stream())

        now = datetime.now(timezone.utc)

        syllabus_df = pd.DataFrame(syllabus_docs)
        event_df = pd.DataFrame([doc.to_dict() or {} for doc in study_event_docs])
        mock_df = pd.DataFrame([doc.to_dict() or {} for doc in mock_result_docs])
        mastery_df = pd.DataFrame([doc.to_dict() or {} for doc in mastery_docs])

        if not event_df.empty:
            if "objective_id" not in event_df and "topic_id" in event_df:
                event_df["objective_id"] = event_df["topic_id"]
            event_df["occurred_at_dt"] = event_df["occurred_at"].apply(_parse_datetime)
            event_df["days_since"] = event_df["occurred_at_dt"].apply(
                lambda value: 0 if value is None else max(0, (now - value).days)
            )
            event_df["accuracy_score"] = event_df.get("accuracy_score", 0.0).fillna(0.0)
        else:
            event_df = pd.DataFrame(columns=["objective_id", "accuracy_score", "days_since"])

        if not mock_df.empty:
            if "objective_id" not in mock_df and "topic_id" in mock_df:
                mock_df["objective_id"] = mock_df["topic_id"]
            if "error_type" in mock_df:
                mock_df["error_type"] = mock_df["error_type"].fillna("").astype(str)
            graded_mock_df = (
                mock_df[mock_df["error_type"] != "awaiting_grading"].copy()
                if "error_type" in mock_df
                else mock_df.copy()
            )
            if graded_mock_df.empty:
                graded_mock_df = pd.DataFrame(
                    columns=[
                        "objective_id",
                        "awarded_marks",
                        "available_marks",
                        "command_word",
                        "error_type",
                        "duration_seconds",
                        "command_word_depth",
                        "semantic_match",
                        "time_per_mark",
                    ]
                )
            graded_mock_df["awarded_marks"] = graded_mock_df.get("awarded_marks", 0.0).fillna(0.0)
            graded_mock_df["available_marks"] = graded_mock_df.get("available_marks", 0.0).fillna(0.0)
            graded_mock_df["command_word_depth_score"] = graded_mock_df.apply(
                lambda row: self._extract_depth_score(row.get("command_word_depth")),
                axis=1,
            )
            graded_mock_df["semantic_match_score"] = graded_mock_df.apply(
                lambda row: self._extract_semantic_score(row.get("semantic_match")),
                axis=1,
            )
            graded_mock_df["time_per_mark"] = graded_mock_df.apply(
                lambda row: float(row.get("duration_seconds", 0) or 0)
                / max(float(row.get("available_marks", 0) or 0), 1.0),
                axis=1,
            )
        else:
            mock_df = pd.DataFrame(
                columns=[
                    "objective_id",
                    "awarded_marks",
                    "available_marks",
                    "command_word",
                    "error_type",
                    "duration_seconds",
                    "command_word_depth",
                    "semantic_match",
                    "command_word_depth_score",
                    "semantic_match_score",
                    "time_per_mark",
                ]
            )
            graded_mock_df = mock_df.copy()

        total_objectives = int(len(syllabus_df.index)) if not syllabus_df.empty else 0
        attempted_objectives = (
            int(event_df["objective_id"].dropna().astype(str).nunique())
            if not event_df.empty and "objective_id" in event_df
            else 0
        )
        coverage = attempted_objectives / total_objectives if total_objectives else 0.0

        total_marks_awarded = (
            float(graded_mock_df["awarded_marks"].sum()) if not graded_mock_df.empty else 0.0
        )
        total_marks_available = (
            float(graded_mock_df["available_marks"].sum()) if not graded_mock_df.empty else 0.0
        )
        accuracy = (
            total_marks_awarded / total_marks_available if total_marks_available > 0 else 0.0
        )

        topic_difficulty = {}
        if not syllabus_df.empty:
            for _, row in syllabus_df.iterrows():
                topic_key = str(
                    row.get("objective_id")
                    or row.get("code", "")
                    or row.get("topic", "")
                )
                topic_difficulty[topic_key] = float(row.get("paper_weight", 0.2) or 0.2) + 0.5

        readiness_by_topic: list[dict[str, Any]] = []
        grouped_topics = set()
        if "objective_id" in event_df:
            grouped_topics.update(
                value for value in event_df["objective_id"].dropna().astype(str).tolist() if value
            )
        if "objective_id" in mock_df:
            grouped_topics.update(
                value for value in mock_df["objective_id"].dropna().astype(str).tolist() if value
            )

        for topic_id in sorted(grouped_topics):
            topic_events = event_df[event_df["objective_id"].astype(str) == topic_id] if not event_df.empty else pd.DataFrame()
            topic_mocks = (
                graded_mock_df[graded_mock_df["objective_id"].astype(str) == topic_id]
                if not graded_mock_df.empty
                else pd.DataFrame()
            )

            topic_accuracy = 0.0
            if not topic_mocks.empty and float(topic_mocks["available_marks"].sum()) > 0:
                topic_accuracy = float(topic_mocks["awarded_marks"].sum()) / float(
                    topic_mocks["available_marks"].sum()
                )
            elif not topic_events.empty:
                topic_accuracy = float(topic_events["accuracy_score"].mean())

            days_since = (
                int(topic_events["days_since"].min())
                if not topic_events.empty and not topic_events["days_since"].empty
                else 30
            )
            difficulty = topic_difficulty.get(topic_id, 0.8)
            decay = math.exp(-0.005 * difficulty * (days_since * 24))
            readiness = max(0.0, min(1.0, topic_accuracy * decay))

            readiness_by_topic.append(
                {
                    "objective_id": topic_id,
                    "accuracy": round(topic_accuracy, 4),
                    "decay": round(decay, 4),
                    "readiness": round(readiness, 4),
                    "risk": "red" if readiness < 0.45 else ("amber" if readiness < 0.7 else "green"),
                }
            )

        average_readiness = (
            sum(item["readiness"] for item in readiness_by_topic) / len(readiness_by_topic)
            if readiness_by_topic
            else 0.0
        )
        exam_risk = max(0.0, min(1.0, 1.0 - ((coverage * 0.45) + (accuracy * 0.35) + (average_readiness * 0.20))))

        command_word_breakdown = []
        weak_command_words = []
        if not graded_mock_df.empty and "command_word" in graded_mock_df:
            for command_word, frame in graded_mock_df.groupby(
                graded_mock_df["command_word"].fillna("Unknown")
            ):
                available = float(frame["available_marks"].sum())
                earned = float(frame["awarded_marks"].sum())
                entry = {
                    "command_word": str(command_word),
                    "accuracy": round((earned / available) if available > 0 else 0.0, 4),
                    "depth_score": round(
                        float(frame.get("command_word_depth_score", pd.Series(dtype=float)).mean() or 0.0),
                        4,
                    ),
                    "semantic_score": round(
                        float(frame.get("semantic_match_score", pd.Series(dtype=float)).mean() or 0.0),
                        4,
                    ),
                    "question_count": int(len(frame.index)),
                }
                command_word_breakdown.append(entry)
                if entry["accuracy"] < 0.65 or entry["depth_score"] < 0.65:
                    weakest_objective = (
                        frame.groupby("objective_id")
                        .apply(
                            lambda objective_frame: float(objective_frame["awarded_marks"].sum())
                            / max(float(objective_frame["available_marks"].sum()), 1.0)
                        )
                        .sort_values()
                    )
                    weak_command_words.append(
                        {
                            "command_word": entry["command_word"],
                            "accuracy": entry["accuracy"],
                            "depth_score": entry["depth_score"],
                            "semantic_score": entry["semantic_score"],
                            "recommended_objective_id": (
                                str(weakest_objective.index[0]) if not weakest_objective.empty else ""
                            ),
                            "recommended_duration_minutes": 10,
                            "reason": (
                                f"{entry['command_word'].title()} responses are dropping marks "
                                "on depth or execution."
                            ),
                        }
                    )

        error_taxonomy = []
        if not graded_mock_df.empty and "error_type" in graded_mock_df:
            counts = (
                graded_mock_df["error_type"]
                .fillna("unknown")
                .astype(str)
                .value_counts()
                .head(5)
                .to_dict()
            )
            error_taxonomy = [
                {"error_type": key, "count": int(value)} for key, value in counts.items()
            ]

        time_leak = None
        if not graded_mock_df.empty and "time_per_mark" in graded_mock_df:
            worst = graded_mock_df.sort_values("time_per_mark", ascending=False).head(1)
            if not worst.empty:
                row = worst.iloc[0]
                time_leak = {
                    "objective_id": str(row.get("objective_id", "")),
                    "command_word": str(row.get("command_word", "")),
                    "time_per_mark": round(float(row.get("time_per_mark", 0.0) or 0.0), 2),
                }

        advisor_payload = {
            "coverage": round(coverage, 4),
            "accuracy": round(accuracy, 4),
            "average_readiness": round(average_readiness, 4),
            "exam_risk": round(exam_risk, 4),
            "readiness_by_topic": readiness_by_topic[:8],
            "command_word_breakdown": command_word_breakdown[:6],
            "weak_command_words": weak_command_words[:3],
            "error_taxonomy": error_taxonomy[:5],
            "time_leak": time_leak,
        }

        advisor = self._build_advisor_output(advisor_payload)
        analytics_doc = {
            "updated_at": now.isoformat(),
            "source_session_id": session_id or "",
            "coverage": round(coverage, 4),
            "accuracy": round(accuracy, 4),
            "average_readiness": round(average_readiness, 4),
            "exam_risk": round(exam_risk, 4),
            "predicted_grade_band": self._predict_grade_band(exam_risk),
            "readiness_heatmap": readiness_by_topic,
            "command_word_breakdown": command_word_breakdown,
            "weak_command_words": weak_command_words,
            "error_taxonomy": error_taxonomy,
            "time_leak": time_leak,
            "advisor": advisor,
        }

        user_ref.collection("analytics").document("current").set(analytics_doc, merge=True)
        self._queue_advisor_action(user_ref, advisor, now)
        return analytics_doc

    def _predict_grade_band(self, exam_risk: float) -> str:
        if exam_risk < 0.18:
            return "A*/7"
        if exam_risk < 0.34:
            return "A/6"
        if exam_risk < 0.52:
            return "B/5"
        if exam_risk < 0.7:
            return "C/4"
        return "D or below"

    def _build_advisor_output(self, payload: dict[str, Any]) -> dict[str, Any]:
        fallback = {
            "headline": "Exam Risk Updated",
            "insight": "Your weakest readiness topics are driving the current risk score.",
            "action": "A targeted drill has been queued for tomorrow.",
        }

        if self._advisor_model is None:
            return fallback

        prompt = f"""
Role: Axon strategist.
Task: Interpret this analytics payload and return strict JSON with keys headline, insight, action.
Payload: {json.dumps(payload)}
Rules:
1. Focus on one critical gap.
2. Mention one time-management or command-word issue if present.
3. Action must be a single specific next step.
"""
        try:
            response = self._advisor_model.generate_content(prompt)
            raw = getattr(response, "text", "").replace("```json", "").replace("```", "").strip()
            parsed = json.loads(raw)
            return {
                "headline": str(parsed.get("headline", fallback["headline"])),
                "insight": str(parsed.get("insight", fallback["insight"])),
                "action": str(parsed.get("action", fallback["action"])),
            }
        except Exception:
            return fallback

    def _extract_depth_score(self, payload: Any) -> float:
        if isinstance(payload, dict):
            raw_score = payload.get("depth_score")
            if isinstance(raw_score, (int, float)):
                return float(raw_score)
            depth_satisfied = payload.get("depth_satisfied")
            if isinstance(depth_satisfied, bool):
                return 1.0 if depth_satisfied else 0.0
        return 0.0

    def _extract_semantic_score(self, payload: Any) -> float:
        if isinstance(payload, dict):
            raw_score = payload.get("overall_score")
            if isinstance(raw_score, (int, float)):
                return float(raw_score)
        return 0.0

    def _queue_advisor_action(self, user_ref, advisor: dict[str, Any], now: datetime) -> None:
        title = str(advisor.get("action", "")).strip()
        if not title:
            return
        tomorrow = now + pd.Timedelta(days=1)
        start = datetime(
            tomorrow.year,
            tomorrow.month,
            tomorrow.day,
            18,
            0,
            tzinfo=timezone.utc,
        )
        end = start + pd.Timedelta(minutes=30)
        user_ref.collection("daily_plan").add(
            {
                "title": title,
                "subject": "Advisor Drill",
                "paper": "Strategy",
                "objective_id": "advisor_generated",
                "start_time": start.isoformat(),
                "end_time": end.isoformat(),
                "status": "pending",
                "date": start.date().isoformat(),
                "reason": advisor.get("insight", ""),
                "is_sync_to_google": False,
                "intensity_score": 1.8,
            }
        )
