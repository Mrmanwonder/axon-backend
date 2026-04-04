from __future__ import annotations

import asyncio
import base64
import os
import tempfile
import uuid
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator

from middleware.auth import FirebaseAuthMiddleware, current_user, initialize_firebase
from services.exam_dates_service import OfficialExamDatesService
from services.grading_service import HandwritingGradingGateway
from services.planner_service import DailyPlannerService
from services.study_pulse_service import StudyPulseService


initialize_firebase()

app = FastAPI(
    title="Axon Backend",
    version="2.1.0",
    description="ASGI backend for Axon document analysis, grading, and trust-safe sync.",
)
app.add_middleware(FirebaseAuthMiddleware)

model = None
model_path = None
_firestore_client = None
_job_cache: dict[str, dict[str, Any]] = {}
_firestore_database_id = os.environ.get("FIRESTORE_DATABASE_ID", "axon")
_gemini_api_key = os.environ.get("GEMINI_API_KEY", "")
_grading_gateway = None
_planner_service = None
_study_pulse_service = None
_exam_dates_service = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_firestore():
    global _firestore_client
    if _firestore_client is not None:
        return _firestore_client

    from firebase_admin import firestore

    try:
        _firestore_client = firestore.client(database_id=_firestore_database_id)
    except TypeError:
        _firestore_client = firestore.client()
    return _firestore_client


def jobs_collection():
    return get_firestore().collection("jobs")


def private_users_collection():
    return get_firestore().collection("users_private")


def public_users_collection():
    return get_firestore().collection("users_public")


def syllabus_maps_collection():
    return get_firestore().collection("syllabus_maps")


def save_job(job_id: str, payload: dict[str, Any]) -> None:
    _job_cache[job_id] = payload
    jobs_collection().document(job_id).set(payload, merge=True)


def update_job(job_id: str, **fields: Any) -> None:
    payload = {**_job_cache.get(job_id, {}), **fields, "updated_at": utc_now()}
    save_job(job_id, payload)


def validate_https_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme == "https" and bool(parsed.netloc)


def get_model():
    global model, model_path
    if model is None:
        try:
            from ultralytics import YOLO

            possible_paths = [
                "exam_layout.pt",
                "./exam_layout.pt",
                os.path.join(os.getcwd(), "exam_layout.pt"),
                "/opt/render/project/src/functions/exam_layout.pt",
            ]

            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    break

            if not model_path:
                raise RuntimeError("Model file not found")

            model = YOLO(model_path)
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Model unavailable: {exc}") from exc
    return model


def get_grading_gateway() -> HandwritingGradingGateway:
    global _grading_gateway
    if _grading_gateway is None:
        _grading_gateway = HandwritingGradingGateway(_gemini_api_key)
    return _grading_gateway


def get_planner_service() -> DailyPlannerService:
    global _planner_service
    if _planner_service is None:
        _planner_service = DailyPlannerService(get_firestore())
    return _planner_service


def get_study_pulse_service() -> StudyPulseService:
    global _study_pulse_service
    if _study_pulse_service is None:
        advisor_model = None
        if _gemini_api_key:
            try:
                import google.generativeai as genai

                genai.configure(api_key=_gemini_api_key)
                advisor_model = genai.GenerativeModel("gemini-1.5-flash")
            except Exception:
                advisor_model = None
        _study_pulse_service = StudyPulseService(get_firestore(), advisor_model=advisor_model)
    return _study_pulse_service


def get_exam_dates_service() -> OfficialExamDatesService:
    global _exam_dates_service
    if _exam_dates_service is None:
        _exam_dates_service = OfficialExamDatesService(get_firestore())
    return _exam_dates_service


class DetectLayoutRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image_data: str | None = None
    image_url: str | None = None

    @model_validator(mode="after")
    def ensure_one_source(self):
        if not self.image_data and not self.image_url:
            raise ValueError("image_data or image_url is required")
        if self.image_url and not validate_https_url(self.image_url):
            raise ValueError("image_url must be a public HTTPS URL")
        return self


class DetectLayoutBatchItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image_data: str | None = None
    image_url: str | None = None
    item_id: str | None = None

    @model_validator(mode="after")
    def ensure_one_source(self):
        if not self.image_data and not self.image_url:
            raise ValueError("image_data or image_url is required")
        if self.image_url and not validate_https_url(self.image_url):
            raise ValueError("image_url must be a public HTTPS URL")
        return self


class DetectLayoutBatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[DetectLayoutBatchItem] = Field(min_length=1, max_length=24)


class AnalyzePdfRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pdf_base64: str | None = None
    pdf_url: str | None = None
    filename: str = "upload.pdf"

    @model_validator(mode="after")
    def ensure_pdf_source(self):
        if not self.pdf_base64 and not self.pdf_url:
            raise ValueError("pdf_base64 or pdf_url is required")
        if self.pdf_url and not validate_https_url(self.pdf_url):
            raise ValueError("pdf_url must be a public HTTPS URL")
        return self


class SyncLeaderboardProfileRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GradeHandwritingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image_url: str
    marking_scheme: dict[str, Any]
    objective: str
    question_prompt: str
    learning_objective_ids: list[str] = Field(default_factory=list)
    command_word: str

    @model_validator(mode="after")
    def validate_https_image(self):
        if not validate_https_url(self.image_url):
            raise ValueError("image_url must be a public HTTPS URL")
        return self


class ExtractTextRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image_data: str | None = None
    image_url: str | None = None

    @model_validator(mode="after")
    def ensure_one_source(self):
        if not self.image_data and not self.image_url:
            raise ValueError("image_data or image_url is required")
        if self.image_url and not validate_https_url(self.image_url):
            raise ValueError("image_url must be a public HTTPS URL")
        return self


class GenerateDailyPlanRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: str | None = None


class StudyPulseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: str | None = None
    session_id: str | None = None


class SyncExamDatesRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    board: str
    subjects: list[str] = Field(min_length=1, max_length=16)
    year: int | None = None
    series: str | None = None
    administrative_zone: str | None = None
    persist: bool = True


def parse_detections(results):
    detections = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            detections.append(
                {
                    "class": int(box.cls[0]),
                    "confidence": float(box.conf[0]),
                    "x": float(box.xywh[0][0]),
                    "y": float(box.xywh[0][1]),
                    "width": float(box.xywh[0][2]),
                    "height": float(box.xywh[0][3]),
                }
            )
    return detections


async def download_or_decode_image(payload: DetectLayoutRequest) -> bytes:
    if payload.image_data:
        if payload.image_data.startswith("data:"):
            return base64.b64decode(payload.image_data.split(",", 1)[1])
        raise HTTPException(status_code=400, detail="image_data must be a data URL")

    import requests

    def fetch() -> bytes:
        response = requests.get(payload.image_url, timeout=20)
        response.raise_for_status()
        return response.content

    return await asyncio.to_thread(fetch)


async def download_or_decode_batch_item(payload: DetectLayoutBatchItem) -> bytes:
    request = DetectLayoutRequest(
        image_data=payload.image_data,
        image_url=payload.image_url,
    )
    return await download_or_decode_image(request)


async def load_pdf_bytes(payload: AnalyzePdfRequest) -> tuple[bytes, str]:
    if payload.pdf_base64:
        return base64.b64decode(payload.pdf_base64), payload.filename

    import requests

    def fetch() -> tuple[bytes, str]:
        response = requests.get(payload.pdf_url, timeout=30)
        response.raise_for_status()
        filename = os.path.basename(urlparse(payload.pdf_url).path) or payload.filename
        return response.content, filename

    return await asyncio.to_thread(fetch)


def build_syllabus_context(learning_objective_ids: list[str]) -> str:
    if not learning_objective_ids:
        return ""

    contexts: list[str] = []
    for objective_id in learning_objective_ids:
        snapshot = syllabus_maps_collection().where("code", is_equal_to=objective_id).limit(1).get()
        for doc in snapshot:
            data = doc.to_dict() or {}
            contexts.append(
                f"{data.get('board', '')} / {data.get('subject', '')} / "
                f"{data.get('paper', '')} / {data.get('topic', '')} / "
                f"{data.get('code', objective_id)}: {data.get('description', '')}"
            )
    return " | ".join(contexts)


def publish_public_user(uid: str) -> dict[str, Any]:
    private_snapshot = private_users_collection().document(uid).get()
    if not private_snapshot.exists:
        raise HTTPException(status_code=404, detail="Private user document not found")

    private_data = private_snapshot.to_dict() or {}
    public_payload = {
        "display_name": private_data.get("display_name", "Student"),
        "photo_url": private_data.get("photo_url"),
        "total_sessions": int(private_data.get("total_sessions", 0) or 0),
        "total_minutes": int(private_data.get("total_minutes", 0) or 0),
        "current_streak": int(private_data.get("current_streak", 0) or 0),
        "predicted_performance": float(
            private_data.get("predicted_performance", 0.0) or 0.0
        ),
        "last_synced_at": utc_now(),
    }
    public_users_collection().document(uid).set(public_payload, merge=True)
    return public_payload


async def process_pdf_job(job_id: str, pdf_bytes: bytes, owner_uid: str, filename: str) -> None:
    update_job(job_id, status="processing", filename=filename)
    tmp_path = None
    try:
        from PyPDF2 import PdfReader

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name

        def extract():
            reader = PdfReader(tmp_path)
            pages_text = []
            for index, page in enumerate(reader.pages):
                pages_text.append({"page": index + 1, "text": page.extract_text()})
            return pages_text

        pages_text = await asyncio.to_thread(extract)
        combined_text = "\n\n".join(
            f"[Page {page['page']}]\n{page['text'] or ''}" for page in pages_text
        ).strip()
        update_job(
            job_id,
            owner_uid=owner_uid,
            status="completed",
            result={"pages": pages_text, "combined_text": combined_text},
            source_type="PDF_TEXT_EXTRACTION",
            completed_at=utc_now(),
        )
    except Exception as exc:
        update_job(
            job_id,
            owner_uid=owner_uid,
            status="failed",
            error=str(exc),
            completed_at=utc_now(),
        )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


async def run_layout_detection_on_bytes(image_bytes: bytes) -> list[dict[str, Any]]:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name

    try:
        yolo_model = get_model()
        results = await asyncio.to_thread(yolo_model, tmp_path, verbose=False)
        return parse_detections(results)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "firebase_admin_ready": True,
        "firestore_database_id": _firestore_database_id,
        "model_loaded": model is not None,
        "transport": "fastapi",
    }


@app.post("/detectLayout")
async def detect_layout(
    payload: DetectLayoutRequest,
    user: dict[str, Any] = Depends(current_user),
):
    image_bytes = await download_or_decode_image(payload)
    detections = await run_layout_detection_on_bytes(image_bytes)
    return {
        "detections": detections,
        "owner_uid": user["uid"],
        "source_type": "MODEL_INFERENCE",
    }


@app.post("/detectLayoutBatch")
async def detect_layout_batch(
    payload: DetectLayoutBatchRequest,
    user: dict[str, Any] = Depends(current_user),
):
    semaphore = asyncio.Semaphore(4)

    async def process_item(index: int, item: DetectLayoutBatchItem) -> dict[str, Any]:
        async with semaphore:
            try:
                image_bytes = await download_or_decode_batch_item(item)
                detections = await run_layout_detection_on_bytes(image_bytes)
                return {
                    "index": index,
                    "item_id": item.item_id,
                    "detections": detections,
                    "status": "completed",
                }
            except Exception as exc:
                return {
                    "index": index,
                    "item_id": item.item_id,
                    "detections": [],
                    "status": "failed",
                    "error": str(exc),
                }

    results = await asyncio.gather(
        *(process_item(index, item) for index, item in enumerate(payload.items))
    )
    return {
        "owner_uid": user["uid"],
        "source_type": "MODEL_INFERENCE_BATCH",
        "item_count": len(results),
        "results": results,
    }


@app.post("/extractText")
async def extract_text(
    payload: ExtractTextRequest,
    user: dict[str, Any] = Depends(current_user),
):
    image_bytes = await download_or_decode_image(payload)

    from google.cloud import vision

    def detect():
        client = vision.ImageAnnotatorClient()
        response = client.text_detection(image=vision.Image(content=image_bytes))
        output = []
        for annotation in response.text_annotations:
            bounds = [
                {"x": vertex.x, "y": vertex.y}
                for vertex in annotation.bounding_poly.vertices
            ]
            output.append({"text": annotation.description, "bounds": bounds})
        return output

    texts = await asyncio.to_thread(detect)
    return {
        "owner_uid": user["uid"],
        "source_type": "GOOGLE_VISION_OCR",
        "texts": texts,
    }


@app.post("/analyzePdf", status_code=202)
async def analyze_pdf(
    payload: AnalyzePdfRequest,
    background_tasks: BackgroundTasks,
    user: dict[str, Any] = Depends(current_user),
):
    pdf_bytes, filename = await load_pdf_bytes(payload)
    job_id = f"pdf_{uuid.uuid4().hex}"
    save_job(
        job_id,
        {
            "job_id": job_id,
            "owner_uid": user["uid"],
            "status": "queued",
            "filename": filename,
            "created_at": utc_now(),
            "updated_at": utc_now(),
        },
    )
    background_tasks.add_task(process_pdf_job, job_id, pdf_bytes, user["uid"], filename)
    return {"job_id": job_id, "status": "queued"}


@app.get("/jobs/{job_id}")
async def get_job(job_id: str, user: dict[str, Any] = Depends(current_user)):
    job = _job_cache.get(job_id)
    if job is None:
        snapshot = jobs_collection().document(job_id).get()
        if snapshot.exists:
            job = snapshot.to_dict()
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.get("owner_uid") != user["uid"]:
        raise HTTPException(status_code=403, detail="Forbidden")
    return job


@app.post("/syncLeaderboardProfile")
async def sync_leaderboard_profile(
    _: SyncLeaderboardProfileRequest,
    user: dict[str, Any] = Depends(current_user),
):
    return {"status": "synced", "public": publish_public_user(user["uid"])}


@app.post("/generate/grade-handwriting")
async def grade_handwriting(
    payload: GradeHandwritingRequest,
    user: dict[str, Any] = Depends(current_user),
):
    gateway = get_grading_gateway()
    syllabus_context = build_syllabus_context(payload.learning_objective_ids)
    result = await gateway.grade_answer(
        image_url=payload.image_url,
        marking_scheme=payload.marking_scheme,
        objective=payload.objective,
        question_prompt=payload.question_prompt,
        learning_objective_ids=payload.learning_objective_ids,
        command_word=payload.command_word,
        syllabus_context=syllabus_context,
    )
    return {"owner_uid": user["uid"], "result": result}


@app.post("/generate-daily-plan")
async def generate_daily_plan(
    payload: GenerateDailyPlanRequest,
    user: dict[str, Any] = Depends(current_user),
):
    owner_uid = payload.user_id or user["uid"]
    if owner_uid != user["uid"]:
        raise HTTPException(status_code=403, detail="Forbidden")

    planner = get_planner_service()
    tasks = await asyncio.to_thread(planner.generate_and_persist_daily_plan, owner_uid)
    return {
        "owner_uid": owner_uid,
        "task_count": len(tasks),
        "tasks": tasks,
        "source_type": "DETERMINISTIC_DAILY_PLAN",
    }


@app.post("/analyze-study-pulse")
async def analyze_study_pulse(
    payload: StudyPulseRequest,
    user: dict[str, Any] = Depends(current_user),
):
    owner_uid = payload.user_id or user["uid"]
    if owner_uid != user["uid"]:
        raise HTTPException(status_code=403, detail="Forbidden")

    service = get_study_pulse_service()
    analytics = await asyncio.to_thread(
        service.analyze_user,
        owner_uid,
        session_id=payload.session_id,
    )
    return {
        "owner_uid": owner_uid,
        "analytics": analytics,
        "source_type": "STUDY_PULSE",
    }


@app.post("/sync-exam-dates")
async def sync_exam_dates(
    payload: SyncExamDatesRequest,
    user: dict[str, Any] = Depends(current_user),
):
    service = get_exam_dates_service()
    result = await asyncio.to_thread(
        service.sync_user_deadlines,
        user_id=user["uid"],
        board=payload.board,
        subjects=payload.subjects,
        year=payload.year,
        series=payload.series,
        administrative_zone=payload.administrative_zone,
        persist=payload.persist,
    )
    return {
        "owner_uid": user["uid"],
        "source_type": "OFFICIAL_DATESHEET_SCRAPER",
        **result,
    }
