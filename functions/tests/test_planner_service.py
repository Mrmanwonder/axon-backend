from __future__ import annotations

from datetime import datetime, timedelta, timezone

from functions.services.planner_service import DailyPlannerServiceV2


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

    def update(self, payload):
        current = self._collection._docs.get(self.id, {})
        current.update(payload)
        self._collection._docs[self.id] = current


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
            elif op == "in" and isinstance(value, list) and data.get(field) in value:
                matched.append(_FakeDocSnapshot(doc_id, data))
        return _FakeQuery(matched)


class _FakeDB:
    def __init__(self):
        self._store = {}

    def collection(self, path: str):
        return _FakeCollectionRef(self, path)

    def batch(self):
        class _Batch:
            def __init__(self, store):
                self._ops = []
                self._store = store

            def set(self, ref, data):
                self._ops.append(("set", ref, data))

            def delete(self, ref):
                self._ops.append(("delete", ref))

            def commit(self):
                for op, ref, *args in self._ops:
                    if op == "set":
                        ref.set(args[0])
                    elif op == "delete":
                        ref.delete()

        return _Batch(self._store)


def test_v2_generates_task_list():
    db = _FakeDB()
    uid = "test-user"
    now = datetime.now(timezone.utc)

    # Seed: settings
    ref = db.collection("users_private").document(uid)
    ref.set({})
    ref.collection("settings").document("prefs").set({
        "target_hours_per_day": 2.0,
        "day_start_hour": 8,
    })

    # Seed: subject with exam 20 days away
    subj_id = "physics-9702"
    subj_ref = db.collection("users_private").document(uid).collection("subjects").document(subj_id)
    subj_ref.set({
        "id": subj_id,
        "name": "Physics",
        "code": "9702",
        "level": "A_LEVEL",
        "exam_date": (now + timedelta(days=20)).isoformat(),
        "target_grade": "A",
        "papers": [{"number": 1, "type": "structured", "duration_minutes": 90, "total_marks": 100, "weight_pct": 50.0}],
        "weak_command_words": [],
    })

    # Seed: objective
    subj_ref.collection("objectives").document("obj-1").set({
        "id": "obj-1",
        "topic": "Kinematics",
        "subtopic": "Motion",
        "paper_numbers": [1],
        "prerequisites": [],
        "mastery_score": 0.3,
        "stability": 1.0,
        "difficulty": 0.3,
    })

    # Seed: analytics summary
    db.collection("users_private").document(uid).collection("analytics").document("summary").set({})

    service = DailyPlannerServiceV2(db)
    import asyncio
    tasks = asyncio.run(service.generate_and_persist_daily_plan(uid, force=True))

    assert isinstance(tasks, list)
    if tasks:
        assert hasattr(tasks[0], "title")
        assert hasattr(tasks[0], "start_time")
        assert hasattr(tasks[0], "end_time")
        assert tasks[0].end_time > tasks[0].start_time
