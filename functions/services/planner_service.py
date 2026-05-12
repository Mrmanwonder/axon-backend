from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
import hashlib
import json
import math
from typing import Any

try:
    import networkx as nx
except Exception:  # pragma: no cover - optional dependency fallback
    nx = None


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


@dataclass(frozen=True)
class PlannerTask:
    title: str
    subject: str
    paper: str
    objective_id: str
    scheduled_start: datetime
    scheduled_end: datetime
    intensity_score: float
    intensity_label: str
    phase: str
    anchor_date: str
    task_type: str
    scheduled_window: str
    reason: str

    def to_firestore(self) -> dict[str, Any]:
        return {
            "title": " ".join(self.title.strip().split()),
            "subject": self.subject,
            "paper": self.paper,
            "objective_id": self.objective_id,
            "start_time": self.scheduled_start.isoformat(),
            "end_time": self.scheduled_end.isoformat(),
            "status": "pending",
            "date": self.scheduled_start.date().isoformat(),
            "reason": self.reason,
            "is_sync_to_google": True,
            "intensity_score": self.intensity_score,
            "intensity_label": self.intensity_label,
            "phase": self.phase,
            "anchor_date": self.anchor_date,
            "task_type": self.task_type,
            "scheduled_window": self.scheduled_window,
        }


class DailyPlannerService:
    def __init__(self, db, planner_model=None) -> None:
        self._db = db
        self._planner_model = planner_model

    def generate_and_persist_daily_plan(
        self,
        user_id: str,
        prioritized_objective_ids: list[str] | None = None,
        crisis_mode: bool = False,
        focus_areas: str | None = None,
        plan_date: str | None = None,
        force: bool = False,
    ) -> list[dict[str, Any]]:
        target_date = self._resolve_plan_date(plan_date)
        plan_ref = self._db.collection("users_private").document(user_id).collection("daily_plan")
        existing_docs = list(plan_ref.where("date", "==", target_date.isoformat()).stream())
        if existing_docs and not force:
            return [self._with_id(snapshot.id, snapshot.to_dict() or {}) for snapshot in existing_docs]

        tasks = self.calculate_daily_load(
            user_id,
            prioritized_objective_ids=prioritized_objective_ids,
            crisis_mode=crisis_mode,
            focus_areas=focus_areas,
            plan_date=target_date,
        )
        tasks = self._refine_tasks_with_model(
            user_id=user_id,
            tasks=tasks,
            focus_areas=focus_areas,
            plan_date=target_date,
        )

        existing_by_id = {
            self._task_document_id(snapshot.to_dict() or {}): snapshot.to_dict() or {}
            for snapshot in existing_docs
        }
        generated_ids: set[str] = set()

        for task in tasks:
            doc_id = self._task_document_id(task)
            generated_ids.add(doc_id)
            preserved = existing_by_id.get(doc_id, {})
            if preserved.get("is_completed") is True or str(preserved.get("status", "")).lower() in {
                "completed",
                "done",
            }:
                task = {
                    **task,
                    "is_completed": preserved.get("is_completed", False),
                    "status": preserved.get("status", "completed"),
                    "completed_at": preserved.get("completed_at"),
                }
            plan_ref.document(doc_id).set(
                {
                    **task,
                    "generated_by": "daily_planner_model" if self._planner_model else "deterministic_planner",
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                },
                merge=True,
            )

        for snapshot in existing_docs:
            data = snapshot.to_dict() or {}
            doc_id = self._task_document_id(data)
            if doc_id in generated_ids:
                continue
            if data.get("is_completed") is True:
                continue
            if str(data.get("status", "pending")).lower() not in {"pending", "generated"}:
                continue
            try:
                snapshot.reference.delete()
            except AttributeError:
                plan_ref.document(snapshot.id).delete()

        return [{**task, "id": self._task_document_id(task)} for task in tasks]

    def run_daily_build(self, user_id: str) -> dict[str, Any]:
        tasks = self.generate_and_persist_daily_plan(user_id, force=True)
        return {
            "tasks": tasks,
            "task_count": len(tasks),
            "built_at": datetime.now(timezone.utc).isoformat(),
            "source_type": "DAILY_BUILD",
        }

    def reschedule_missed_block(self, user_id: str, task_id: str) -> dict[str, Any]:
        plan_ref = self._db.collection("users_private").document(user_id).collection("daily_plan")
        task_ref = plan_ref.document(task_id)
        snapshot = task_ref.get()
        if not snapshot.exists:
            return {"status": "not_found", "task_id": task_id}

        task = snapshot.to_dict() or {}
        now = datetime.now(timezone.utc)
        start = datetime.combine(now.date(), time(hour=20), tzinfo=timezone.utc)
        end = start + timedelta(minutes=45)
        task_ref.set(
            {
                "status": "rescheduled",
                "start_time": start.isoformat(),
                "end_time": end.isoformat(),
                "reason": (
                    f"{task.get('reason', '')} Life happened: shifted this block to 8 PM tonight "
                    "and trimmed review work to compensate."
                ).strip(),
                "scheduled_window": "late_evening_recovery",
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
            merge=True,
        )

        trimmed_id = None
        for other in plan_ref.where("date", "==", now.date().isoformat()).stream():
            if other.id == task_id:
                continue
            payload = other.to_dict() or {}
            if str(payload.get("task_type", "")).lower() == "revision":
                other.reference.set(
                    {
                        "status": "trimmed",
                        "reason": (
                            f"{payload.get('reason', '')} Review trimmed to make room for "
                            f"the rescheduled {task.get('title', 'study block')}."
                        ).strip(),
                    },
                    merge=True,
                )
                trimmed_id = other.id
                break

        return {
            "status": "rescheduled",
            "task_id": task_id,
            "trimmed_task_id": trimmed_id,
            "new_start_time": start.isoformat(),
            "new_end_time": end.isoformat(),
        }

    def calculate_daily_load(
        self,
        user_id: str,
        prioritized_objective_ids: list[str] | None = None,
        crisis_mode: bool = False,
        focus_areas: str | None = None,
        plan_date: date | None = None,
    ) -> list[dict[str, Any]]:
        user_doc = self._db.collection("users_private").document(user_id).get()
        if not user_doc.exists:
            return []

        user_data = user_doc.to_dict() or {}
        target_hours = float(user_data.get("target_hours", 4) or 4)
        available_hours = max(1.0, target_hours)
        block_minutes = 45
        block_count = max(1, int((available_hours * 60) // block_minutes))

        deadlines = list(
            self._db.collection("users_private").document(user_id).collection("deadlines").stream()
        )
        analytics_snapshot = (
            self._db.collection("users_private")
            .document(user_id)
            .collection("analytics")
            .document("current")
            .get()
        )
        mastery_docs = list(
            self._db.collection("users_private").document(user_id).collection("mastery").stream()
        )
        event_docs = list(
            self._db.collection("users_private").document(user_id).collection("events").stream()
        )

        now = datetime.now(timezone.utc)
        target_date = plan_date or now.date()
        ranked_deadlines: list[tuple[datetime, dict[str, Any]]] = []
        for snapshot in deadlines:
            data = snapshot.to_dict() or {}
            exam_at = _parse_datetime(data.get("exam_date"))
            if exam_at is None or exam_at.date() < target_date:
                continue
            ranked_deadlines.append((exam_at, data))
        ranked_deadlines.sort(key=lambda item: item[0])
        if not ranked_deadlines:
            print(f"[PlannerService] No deadlines for user {user_id}, skipping plan generation")
            return []

        exam_at, deadline = ranked_deadlines[0]
        analytics_data = analytics_snapshot.to_dict() if analytics_snapshot.exists else {}
        subject = str(deadline.get("subject", "")).strip()
        paper = str(deadline.get("paper", "")).strip() or "Core Paper"
        days_to_exam = max(1, (exam_at.date() - now.date()).days)
        no_study_zones = self._extract_no_study_zones(event_docs, now, exam_at)
        completion_anchor = max(target_date + timedelta(days=1), exam_at.date() - timedelta(days=14))
        active_days_remaining = self._count_active_days(
            target_date,
            completion_anchor,
            no_study_zones,
        )

        syllabus = list(self._db.collection("syllabus_maps").where("subject", "==", subject).stream())
        if not syllabus:
            return []

        mastery_by_objective = {}
        for snapshot in mastery_docs:
            data = snapshot.to_dict() or {}
            objective_key = str(
                data.get("learning_objective_id")
                or data.get("objective_id")
                or snapshot.id
            )
            mastery_by_objective[objective_key] = data
        recent_failures = {
            str(
                (snapshot.to_dict() or {}).get("learning_objective_id")
                or (snapshot.to_dict() or {}).get("objective_id")
                or (snapshot.to_dict() or {}).get("topic_id")
                or ""
            )
            for snapshot in event_docs
            if str((snapshot.to_dict() or {}).get("type", "")).lower() in {"mock_failed", "question_failed"}
        }
        phase = self._phase_for_days_remaining(days_to_exam)
        completion_days_left = max(1, (completion_anchor - now.date()).days)
        baseline_velocity = len(syllabus) / max(completion_days_left, 1)
        adjusted_velocity = len(syllabus) / max(active_days_remaining, 1)
        redistributed_load = max(0.0, adjusted_velocity - baseline_velocity)
        scored: list[tuple[float, dict[str, Any], str, str, list[str], float]] = []
        mastered_objectives: set[str] = set()
        for snapshot in syllabus:
            data = snapshot.to_dict() or {}
            objective_id = str(
                data.get("objective_id")
                or data.get("code")
                or snapshot.id
            )
            mastery = mastery_by_objective.get(objective_id, {})
            confidence = float(mastery.get("confidence_score", 0.35) or 0.35)
            decay_factor = float(mastery.get("decay_factor", 0.012) or 0.012)
            last_tested = _parse_datetime(mastery.get("last_tested"))
            days_since_review = 30 if last_tested is None else max(0, (now - last_tested).days)
            stored_mastery = float(mastery.get("mastery_score", confidence) or confidence)
            decayed_mastery = stored_mastery * math.exp(-decay_factor * days_since_review)
            if decayed_mastery >= 0.85:
                mastered_objectives.add(objective_id)
            priority = (1.0 - decayed_mastery) * 50
            priority += min(days_since_review, 45) * 0.9
            priority += max(decay_factor, 0.005) * 120
            priority += float(data.get("paper_weight", 0.0) or 0.0) * 30
            if objective_id in recent_failures:
                priority += 40
            high_yield = float(
                data.get("past_paper_frequency")
                or data.get("past_paper_hits")
                or data.get("historical_frequency")
                or 0.0
            )
            priority += high_yield * 6
            if phase == "timed_mock_sprint":
                priority += high_yield * 10
                priority += decayed_mastery * 18
            elif phase == "hard_topic_deep_dive":
                priority += (1.0 - decayed_mastery) * 22
            elif phase == "first_pass_completion":
                priority += (0.65 - decayed_mastery) * 10

            prerequisite_ids = [
                str(item).strip()
                for item in (
                    data.get("prerequisite_ids")
                    or data.get("prerequisites")
                    or []
                )
                if str(item).strip()
            ]
            unmet_prerequisites = [
                item for item in prerequisite_ids if item not in mastered_objectives
            ]
            if phase == "timed_mock_sprint" and unmet_prerequisites:
                priority -= 60
            elif unmet_prerequisites:
                priority -= min(15, len(unmet_prerequisites) * 5)

            reason = (
                f"Anchor {exam_at.date().isoformat()} | velocity {adjusted_velocity:.2f}/day. "
                f"Mastery {decayed_mastery:.2f}, decay {decay_factor:.3f}, "
                f"{days_since_review} day(s) since review."
            )
            if redistributed_load > 0:
                reason += f" Buffer recovery adds {redistributed_load:.2f} objective/day."
            scored.append((priority, data, objective_id, reason, prerequisite_ids, decayed_mastery))

        scored = self._order_by_prerequisites(scored)
        selected = scored[:block_count]
        prioritized = [
            objective_id
            for objective_id in (prioritized_objective_ids or [])
            if objective_id.strip()
        ]
        prioritized_set = set(prioritized)

        if prioritized_set:
            prioritized_candidates = [
                item for item in scored if item[2] in prioritized_set
            ]
            non_prioritized_candidates = [
                item for item in scored if item[2] not in prioritized_set
            ]
            selected = prioritized_candidates[:block_count]
            if len(selected) < block_count:
                selected.extend(non_prioritized_candidates[: block_count - len(selected)])

        if crisis_mode and prioritized_set:
            crisis_candidates = [
                item for item in scored if item[2] in prioritized_set
            ]
            if crisis_candidates:
                selected = crisis_candidates[:block_count]
            if len(selected) < block_count:
                for item in scored:
                    if item in selected:
                        continue
                    selected.append(item)
                    if len(selected) >= block_count:
                        break

        tasks: list[dict[str, Any]] = []
        phase_label = self._phase_label(phase)
        intensity_baseline = max(1.0, adjusted_velocity * 12)
        focus_slots = self._build_slots(user_data, block_count)
        for index, (priority, data, objective_id, reason, prerequisite_ids, decayed_mastery) in enumerate(selected):
            slot = focus_slots[min(index, len(focus_slots) - 1)]
            start = datetime.combine(target_date, slot["start"], tzinfo=timezone.utc)
            end = start + timedelta(minutes=block_minutes)
            crisis_reason = "Crisis mode: low-priority topics pruned to high-yield objectives. "
            unmet_prerequisites = [
                item for item in prerequisite_ids if item not in mastered_objectives
            ]
            raw_title = str(data.get("title") or data.get("topic") or objective_id).strip()
            safe_title = raw_title or objective_id or "Study Objective"
            title = self._task_title_for_phase(
                phase=phase,
                title=safe_title,
                unmet_prerequisites=unmet_prerequisites,
            )
            if unmet_prerequisites:
                reason = (
                    f"{reason} Prerequisite hold: complete {', '.join(unmet_prerequisites[:3])} "
                    "before advanced mock work."
                )
            if phase == "hard_topic_deep_dive":
                reason = f"{reason} Hard-topic focus from mastery engine."
            elif phase == "timed_mock_sprint":
                reason = f"{reason} Full-length timed mock window."
            elif phase == "first_pass_completion":
                reason = f"{reason} First-pass syllabus completion window."
            task_type = "deep_work" if decayed_mastery < 0.55 else "revision"
            if phase == "timed_mock_sprint":
                task_type = "mock_exam"
            if self._has_recent_missed_block(event_docs):
                reason = (
                    f"{reason} Yesterday's missed block increased velocity to "
                    f"{adjusted_velocity:.2f}/day, so the schedule is tighter today."
                )
            if focus_areas:
                reason = f"{reason} Student focus request: {focus_areas.strip()[:160]}."
            tasks.append(
                PlannerTask(
                    title=title,
                    subject=subject,
                    paper=str(data.get("paper", paper)),
                    objective_id=objective_id,
                    scheduled_start=start,
                    scheduled_end=end,
                    intensity_score=round(min(3.0, max(1.0, priority / intensity_baseline)), 2),
                    intensity_label=self._intensity_label(priority / intensity_baseline),
                    phase=phase_label,
                    anchor_date=exam_at.date().isoformat(),
                    task_type=task_type,
                    scheduled_window=slot["label"],
                    reason=(crisis_reason + reason) if crisis_mode else reason,
                ).to_firestore()
            )
        self._inject_command_word_drills(
            tasks=tasks,
            analytics_data=analytics_data,
            subject=subject,
            paper=paper,
            anchor_date=exam_at.date().isoformat(),
            now=datetime.combine(target_date, time(hour=19), tzinfo=timezone.utc),
        )
        return tasks

    def _resolve_plan_date(self, value: str | None) -> date:
        if value:
            try:
                return date.fromisoformat(str(value).split("T", 1)[0])
            except ValueError:
                pass
        return datetime.now(timezone.utc).date()

    def _task_document_id(self, task: dict[str, Any]) -> str:
        seed = "|".join(
            [
                str(task.get("date", "")),
                str(task.get("objective_id", "")),
                str(task.get("task_type", "")),
                str(task.get("scheduled_window", "")),
            ]
        )
        digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:16]
        return f"plan_{digest}"

    def _with_id(self, doc_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return {"id": doc_id, **payload}

    def _refine_tasks_with_model(
        self,
        *,
        user_id: str,
        tasks: list[dict[str, Any]],
        focus_areas: str | None,
        plan_date: date,
    ) -> list[dict[str, Any]]:
        if not tasks or self._planner_model is None:
            return tasks

        prompt = {
            "role": "daily_plan_refiner",
            "instruction": (
                "Refine this student's daily study plan. Keep the same number of tasks, "
                "same objective_id values, same date, and valid non-overlapping time windows. "
                "Return strict JSON only: {\"tasks\": [...]} with title, description, reason, "
                "start_time, end_time, task_type, intensity_label. Do not add markdown."
            ),
            "student_focus": focus_areas or "",
            "user_id": user_id,
            "plan_date": plan_date.isoformat(),
            "tasks": tasks,
        }
        try:
            response = self._planner_model.generate_content(json.dumps(prompt, ensure_ascii=False))
            raw = getattr(response, "text", "") or ""
            payload = self._extract_json_object(raw)
            refined = payload.get("tasks") if isinstance(payload, dict) else None
            if not isinstance(refined, list) or len(refined) != len(tasks):
                return tasks
            return self._merge_model_tasks(tasks, refined)
        except Exception:
            return tasks

    def _extract_json_object(self, raw: str) -> dict[str, Any]:
        text = raw.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:].strip()
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {}
        return json.loads(text[start : end + 1])

    def _merge_model_tasks(
        self,
        base_tasks: list[dict[str, Any]],
        model_tasks: list[Any],
    ) -> list[dict[str, Any]]:
        allowed_labels = {"blue", "orange", "red"}
        merged: list[dict[str, Any]] = []
        for base, candidate in zip(base_tasks, model_tasks):
            if not isinstance(candidate, dict):
                merged.append(base)
                continue
            task = dict(base)
            for field in ("title", "description", "reason", "task_type"):
                value = str(candidate.get(field, "")).strip()
                if value:
                    task[field] = value[:600] if field == "reason" else value[:160]
            label = str(candidate.get("intensity_label", "")).strip().lower()
            if label in allowed_labels:
                task["intensity_label"] = label
            start = _parse_datetime(candidate.get("start_time"))
            end = _parse_datetime(candidate.get("end_time"))
            base_date = str(base.get("date", ""))
            if start and end and end > start and start.date().isoformat() == base_date:
                task["start_time"] = start.isoformat()
                task["end_time"] = end.isoformat()
            merged.append(task)
        return merged

    def _extract_no_study_zones(
        self,
        event_docs: list[Any],
        now: datetime,
        exam_at: datetime,
    ) -> list[tuple[datetime, datetime, str]]:
        zones: list[tuple[datetime, datetime, str]] = []
        for snapshot in event_docs:
            data = snapshot.to_dict() or {}
            event_type = str(data.get("type", "")).lower().strip()
            if event_type not in {"travel", "trip", "blocked", "no_study_zone", "no_study"}:
                continue
            start = _parse_datetime(data.get("start_at") or data.get("occurred_at"))
            if start is None:
                continue
            end = _parse_datetime(data.get("end_at"))
            if end is None:
                duration_minutes = int(data.get("duration_minutes", 0) or 0)
                if duration_minutes > 0:
                    end = start + timedelta(minutes=duration_minutes)
                else:
                    end = start
            if end < now or start.date() > exam_at.date():
                continue
            zones.append((start, end, str(data.get("label", "")).strip()))
        return zones

    def _count_active_days(
        self,
        start_date,
        exam_date,
        no_study_zones: list[tuple[datetime, datetime, str]],
    ) -> int:
        active_days = 0
        cursor = start_date
        while cursor < exam_date:
            if not self._is_blocked(cursor, no_study_zones):
                active_days += 1
            cursor += timedelta(days=1)
        return max(active_days, 1)

    def _is_blocked(
        self,
        date_value,
        no_study_zones: list[tuple[datetime, datetime, str]],
    ) -> bool:
        for start, end, _ in no_study_zones:
            if start.date() <= date_value <= end.date():
                return True
        return False

    def _phase_for_days_remaining(self, days_to_exam: int) -> str:
        if days_to_exam <= 7:
            return "timed_mock_sprint"
        if days_to_exam <= 14:
            return "hard_topic_deep_dive"
        if days_to_exam <= 30:
            return "first_pass_completion"
        return "foundation_build"

    def _phase_label(self, phase: str) -> str:
        return {
            "timed_mock_sprint": "T-7 Mock Sprint",
            "hard_topic_deep_dive": "T-14 Deep Dive",
            "first_pass_completion": "T-30 Completion",
            "foundation_build": "Foundation Build",
        }.get(phase, "Study Block")

    def _build_slots(self, user_data: dict[str, Any], block_count: int) -> list[dict[str, Any]]:
        deep_work_hour = int(user_data.get("deep_work_start_hour", 7) or 7)
        review_hour = int(user_data.get("review_start_hour", 18) or 18)
        slots: list[dict[str, Any]] = []
        morning_blocks = max(1, math.ceil(block_count * 0.6))
        for index in range(morning_blocks):
            slots.append(
                {
                    "start": time(hour=min(22, deep_work_hour + index)),
                    "label": "peak_focus_morning",
                }
            )
        for index in range(block_count - morning_blocks):
            slots.append(
                {
                    "start": time(hour=min(22, review_hour + index)),
                    "label": "review_evening",
                }
            )
        return slots or [{"start": time(hour=7), "label": "peak_focus_morning"}]

    def _task_title_for_phase(
        self,
        phase: str,
        title: str,
        unmet_prerequisites: list[str],
    ) -> str:
        if unmet_prerequisites:
            return f"Prerequisite Recovery: {title}"
        prefix = {
            "timed_mock_sprint": "Timed Mock",
            "hard_topic_deep_dive": "Hard-Topic Deep Dive",
            "first_pass_completion": "First Pass",
            "foundation_build": "Foundation Build",
        }.get(phase, "Study")
        return f"{prefix}: {title}"

    def _intensity_label(self, intensity_score: float) -> str:
        if intensity_score >= 2.2:
            return "red"
        if intensity_score >= 1.15:
            return "orange"
        return "blue"

    def _has_recent_missed_block(self, event_docs: list[Any]) -> bool:
        for snapshot in event_docs:
            data = snapshot.to_dict() or {}
            event_type = str(data.get("type", "")).lower().strip()
            if event_type in {"missed_block", "skipped_session", "missed_session"}:
                return True
        return False

    def _inject_command_word_drills(
        self,
        *,
        tasks: list[dict[str, Any]],
        analytics_data: dict[str, Any],
        subject: str,
        paper: str,
        anchor_date: str,
        now: datetime,
    ) -> None:
        weak_words = analytics_data.get("weak_command_words") or []
        if not isinstance(weak_words, list) or not weak_words:
            return

        prioritized = sorted(
            [
                item
                for item in weak_words
                if isinstance(item, dict) and str(item.get("command_word", "")).strip()
            ],
            key=lambda item: (
                0 if str(item.get("command_word", "")).strip().lower() == "explain" else 1,
                float(item.get("accuracy", 1.0) or 1.0),
                float(item.get("depth_score", 1.0) or 1.0),
            ),
        )
        if not prioritized:
            return

        latest_end = max(
            (
                _parse_datetime(task.get("end_time"))
                for task in tasks
                if isinstance(task, dict) and _parse_datetime(task.get("end_time")) is not None
            ),
            default=datetime.combine(now.date(), time(hour=19), tzinfo=timezone.utc),
        )
        drill_start = latest_end + timedelta(minutes=15)
        for item in prioritized[:2]:
            command_word = str(item.get("command_word", "")).strip()
            if not command_word:
                continue
            duration_minutes = int(item.get("recommended_duration_minutes", 10) or 10)
            drill_end = drill_start + timedelta(minutes=duration_minutes)
            objective_id = str(item.get("recommended_objective_id", "")).strip() or "command_word_focus"
            reason = str(item.get("reason", "")).strip() or (
                f"{command_word.title()} responses need depth repair."
            )
            tasks.append(
                PlannerTask(
                    title=f"{command_word.title()} Drill",
                    subject=subject,
                    paper=paper,
                    objective_id=objective_id,
                    scheduled_start=drill_start,
                    scheduled_end=drill_end,
                    intensity_score=1.15,
                    intensity_label="orange",
                    phase="Skill Repair",
                    anchor_date=anchor_date,
                    task_type="command_word_drill",
                    scheduled_window="skill_repair_evening",
                    reason=reason,
                ).to_firestore()
            )
            drill_start = drill_end + timedelta(minutes=10)

    def _order_by_prerequisites(
        self,
        scored: list[tuple[float, dict[str, Any], str, str, list[str], float]],
    ) -> list[tuple[float, dict[str, Any], str, str, list[str], float]]:
        if nx is None:
            return sorted(scored, key=lambda item: item[0], reverse=True)

        graph = nx.DiGraph()
        by_id = {item[2]: item for item in scored}
        for _, _, objective_id, _, prerequisite_ids, _ in scored:
            graph.add_node(objective_id)
            for prereq in prerequisite_ids:
                if prereq in by_id:
                    graph.add_edge(prereq, objective_id)
        try:
            order = list(nx.topological_sort(graph))
        except Exception:
            return sorted(scored, key=lambda item: item[0], reverse=True)
        position = {objective_id: index for index, objective_id in enumerate(order)}
        return sorted(
            scored,
            key=lambda item: (
                position.get(item[2], len(position)),
                -item[0],
            ),
        )
