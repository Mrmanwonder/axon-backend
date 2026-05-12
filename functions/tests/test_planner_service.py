from __future__ import annotations

from datetime import datetime, timedelta, timezone

from services.planner_service import DailyPlannerService


class _FakeDocSnapshot:
    def __init__(self, doc_id: str, data: dict, exists: bool = True):
        self.id = doc_id
        self._data = data
        self.exists = exists

    def to_dict(self):
        return self._data


class _FakeDocumentRef:
    def __init__(self, collection, doc_id: str):
        self._collection = collection
        self.id = doc_id

    def get(self):
        data = self._collection._docs.get(self.id)
        if data is None:
            return _FakeDocSnapshot(self.id, {}, exists=False)
        return _FakeDocSnapshot(self.id, data, exists=True)

    def set(self, payload, merge: bool = False):
        current = self._collection._docs.get(self.id, {})
        if merge:
            current = {**current, **payload}
            self._collection._docs[self.id] = current
        else:
            self._collection._docs[self.id] = dict(payload)

    def delete(self):
        self._collection._docs.pop(self.id, None)

    def collection(self, name: str):
        path = f"{self._collection._path}/{self.id}/{name}"
        return self._collection._db.collection(path)


class _FakeQuery:
    def __init__(self, docs):
        self._docs = docs

    def stream(self):
        return self._docs

    def limit(self, count: int):
        return _FakeQuery(self._docs[:count])

    def get(self):
        return self._docs


class _FakeCollectionRef:
    def __init__(self, db, path: str):
        self._db = db
        self._path = path
        self._docs = db._store.setdefault(path, {})

    def document(self, doc_id: str):
        return _FakeDocumentRef(self, doc_id)

    def stream(self):
        return [_FakeDocSnapshot(doc_id, data) for doc_id, data in self._docs.items()]

    def where(self, field: str, op: str, value):
        matched = []
        for doc_id, data in self._docs.items():
            if op == "==" and data.get(field) == value:
                matched.append(_FakeDocSnapshot(doc_id, data))
        return _FakeQuery(matched)


class _FakeDB:
    def __init__(self):
        self._store = {}

    def collection(self, path: str):
        return _FakeCollectionRef(self, path)


def _seed_common_data(db: _FakeDB, user_id: str):
    now = datetime.now(timezone.utc)
    users_private = db.collection("users_private")
    users_private.document(user_id).set({"target_hours": 1})

    deadlines = db.collection(f"users_private/{user_id}/deadlines")
    deadlines.document("deadline-1").set(
        {
            "subject": "Physics",
            "paper": "Paper 1",
            "exam_date": (now + timedelta(days=10)).isoformat(),
        }
    )

    db.collection(f"users_private/{user_id}/analytics").document("current").set({})
    mastery = db.collection(f"users_private/{user_id}/mastery")
    mastery.document("obj-a").set({"objective_id": "obj-a", "mastery_score": 0.2})
    mastery.document("obj-b").set({"objective_id": "obj-b", "mastery_score": 0.8})

    syllabus = db.collection("syllabus_maps")
    syllabus.document("obj-a").set(
        {
            "subject": "Physics",
            "paper": "Paper 1",
            "objective_id": "obj-a",
            "title": "Kinematics",
            "past_paper_frequency": 0.9,
        }
    )
    syllabus.document("obj-b").set(
        {
            "subject": "Physics",
            "paper": "Paper 1",
            "objective_id": "obj-b",
            "title": "Dynamics",
            "past_paper_frequency": 0.1,
        }
    )


def test_prioritized_objective_is_respected_without_crisis_mode():
    db = _FakeDB()
    user_id = "user-1"
    _seed_common_data(db, user_id)

    service = DailyPlannerService(db)
    tasks = service.calculate_daily_load(
        user_id,
        prioritized_objective_ids=["obj-b"],
        crisis_mode=False,
    )

    assert tasks, "Expected at least one task"
    assert tasks[0]["objective_id"] == "obj-b"


def test_generated_tasks_have_non_empty_title_and_time_bounds():
    db = _FakeDB()
    user_id = "user-2"
    _seed_common_data(db, user_id)

    syllabus = db.collection("syllabus_maps")
    syllabus.document("obj-empty").set(
        {
            "subject": "Physics",
            "paper": "Paper 1",
            "objective_id": "obj-empty",
            "title": "   ",
            "topic": "",
            "past_paper_frequency": 0.5,
        }
    )
    db.collection(f"users_private/{user_id}/mastery").document("obj-empty").set(
        {"objective_id": "obj-empty", "mastery_score": 0.1}
    )

    service = DailyPlannerService(db)
    tasks = service.calculate_daily_load(user_id)

    assert tasks, "Expected planner to generate tasks"
    for task in tasks:
        assert isinstance(task.get("title"), str) and task["title"].strip()
        start = datetime.fromisoformat(task["start_time"])
        end = datetime.fromisoformat(task["end_time"])
        assert end > start
