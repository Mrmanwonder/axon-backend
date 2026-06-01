from __future__ import annotations

import asyncio
import base64
import json
import os
import tempfile
import uuid
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

import uvicorn

import time
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from middleware.auth import current_user, initialize_firebase
from services.exam_dates_service import OfficialExamDatesService
from services.grading_service import HandwritingGradingGateway
from services.planner_service import DailyPlannerServiceV2
from services.study_pulse_service import StudyPulseService
from services.university_catalog_service import UniversityCatalogService
from services.university_program_crawler import UniversityProgramCrawler
from services.ai_proxy_service import AiProxyService


app = FastAPI(
    title="Axon Backend",
    version="2.1.0",
    description="ASGI backend for Axon document analysis, grading, and trust-safe sync.",
)

# Security: CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://axon.edu", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Content-Type", "Authorization", "Accept", "X-Client-Version", "X-Client-Platform"],
)

# Security: rate limiting
@app.middleware("http")
async def rate_limit_middleware(request, call_next):
    import collections

    client_ip = request.client.host if request.client else "unknown"
    key = f"rl:{client_ip}:{int(time.time() // 60)}"
    now = time.time()

    # Simple in-memory rate limit: 60 requests per minute per IP
    if not hasattr(rate_limit_middleware, "_counts"):
        rate_limit_middleware._counts = collections.defaultdict(list)

    rate_limit_middleware._counts[key] = [
        t for t in rate_limit_middleware._counts[key] if now - t < 60
    ]
    if len(rate_limit_middleware._counts[key]) >= 60:
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded"},
        )
    rate_limit_middleware._counts[key].append(now)

    return await call_next(request)


initialize_firebase()

_firestore_client = None

# Removed bounded in-memory job cache; now purely stateless via Firestore





_firestore_database_id = os.environ.get("FIRESTORE_DATABASE_ID", "axon")
_gemini_api_key = os.environ.get("GEMINI_API_KEY", "")
_grading_gateway = None
_planner_service: "DailyPlannerServiceV2 | None" = None
_study_pulse_service = None
_exam_dates_service = None
_uni_catalog_service = None
_ai_proxy_service = None


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
    try:
        jobs_collection().document(job_id).set(payload, merge=True)
    except Exception as e:
        print(f"Error saving job {job_id}: {e}")

def update_job(job_id: str, **fields: Any) -> None:
    try:
        fields["updated_at"] = utc_now()
        jobs_collection().document(job_id).update(fields)
    except Exception as e:
        print(f"Error updating job {job_id}: {e}")


def validate_https_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme == "https" and bool(parsed.netloc)


def get_grading_gateway() -> HandwritingGradingGateway:
    global _grading_gateway
    if _grading_gateway is None:
        _grading_gateway = HandwritingGradingGateway(_gemini_api_key)
    return _grading_gateway


def get_planner_service() -> DailyPlannerServiceV2:
    global _planner_service
    if _planner_service is None:
        _planner_service = DailyPlannerServiceV2(get_firestore(), gemini_api_key=_gemini_api_key)
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


def get_uni_catalog_service() -> UniversityCatalogService:
    global _uni_catalog_service
    if _uni_catalog_service is None:
        planner_model = None
        if _gemini_api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=_gemini_api_key)
                planner_model = genai.GenerativeModel("gemini-1.5-flash")
            except Exception:
                planner_model = None
        _uni_catalog_service = UniversityCatalogService(
            get_firestore(), model=planner_model
        )
    return _uni_catalog_service


def get_program_crawler() -> UniversityProgramCrawler:
    return UniversityProgramCrawler(get_firestore())


def get_ai_proxy_service() -> AiProxyService:
    global _ai_proxy_service
    if _ai_proxy_service is None:
        _ai_proxy_service = AiProxyService()
    return _ai_proxy_service


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
    focus_areas: str | None = Field(default=None, max_length=800)
    client_date: str | None = Field(default=None, max_length=32)
    force: bool = False


class StudyPulseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: str | None = None
    session_id: str | None = None


class SupabaseQueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    table: str = Field(min_length=1, max_length=64)
    method: str = Field(default="select", pattern=r"^(select|insert|update|delete|upsert)$")
    params: dict[str, Any] = {}


ALLOWED_SUPABASE_TABLES = {
    "boards", "subjects", "chapters", "Datesheet", "PYQs",
    "curriculum_subjects", "curriculum_chapters", "curriculum_subchapters", "curriculum_syllabi",
    "user_notes", "user_subjects", "global_notes",
    "subchapter_notes", "user_subchapter_progress",
    "user_pyqs", "user_mocks", "study_progress",
}


@app.post("/supabase/query")
async def supabase_query(
    payload: SupabaseQueryRequest,
    user: dict[str, Any] = Depends(current_user),
):
    if payload.table not in ALLOWED_SUPABASE_TABLES:
        raise HTTPException(status_code=403, detail=f"Table '{payload.table}' not allowed")

    supabase_url = os.environ.get("SUPABASE_URL", "").rstrip("/")
    service_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")

    if not supabase_url or not service_key:
        raise HTTPException(status_code=500, detail="Supabase not configured on server")

    url = f"{supabase_url}/rest/v1/{payload.table}"
    headers = {
        "apikey": service_key,
        "Authorization": f"Bearer {service_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    import httpx

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            if payload.method == "select":
                resp = await client.get(url, headers=headers, params=payload.params)
            elif payload.method == "insert":
                body = payload.params.pop("body", [])
                resp = await client.post(url, headers=headers, json=body, params=payload.params)
            elif payload.method == "upsert":
                body = payload.params.pop("body", [])
                headers["Prefer"] = "resolution=merge-duplicates"
                resp = await client.post(url, headers=headers, json=body, params=payload.params)
            elif payload.method == "update":
                body = payload.params.pop("body", {})
                resp = await client.patch(url, headers=headers, json=body, params=payload.params)
            elif payload.method == "delete":
                resp = await client.delete(url, headers=headers, params=payload.params)
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported method: {payload.method}")

        if resp.status_code >= 400:
            detail = resp.text[:500]
            raise HTTPException(status_code=502, detail=f"Supabase error ({resp.status_code}): {detail}")

        try:
            return resp.json()
        except Exception:
            return {"result": resp.text}

    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Supabase query timed out")
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Supabase request failed: {e}")


class SyncExamDatesRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    board: str
    subjects: list[str] = Field(min_length=1, max_length=16)
    year: int | None = None
    series: str | None = None
    administrative_zone: str | None = None
    persist: bool = True


class UniversitySearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = ""
    country: str = ""
    name: str = ""
    limit: int = Field(default=50, le=200)


class UniversityProgramsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    university_name: str
    country: str = ""
    domain: str = ""


class NormalizeDegreeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    degree_name: str
    country: str = ""


class AiChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    messages: list[dict[str, str]] = Field(min_length=1, max_length=100)
    stream: bool = False
    max_tokens: int | None = Field(default=None, ge=1, le=16384)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)


async def download_or_decode_image(payload: BaseModel) -> bytes:
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


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "firebase_admin_ready": True,
        "firestore_database_id": _firestore_database_id,
        "transport": "fastapi",
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
    try:
        snapshot = jobs_collection().document(job_id).get()
        if snapshot.exists:
            job = snapshot.to_dict()
        else:
            raise HTTPException(status_code=404, detail="Job not found")
            
        if job.get("owner_uid") != user["uid"]:
            raise HTTPException(status_code=403, detail="Forbidden")
        return job
    except Exception as e:
        if isinstance(e, HTTPException): raise e
        raise HTTPException(status_code=500, detail=str(e))


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
    from datetime import date
    target = date.fromisoformat(payload.client_date) if payload.client_date else None
    tasks = await planner.generate_and_persist_daily_plan(
        owner_uid,
        force=payload.force,
        target_date=target,
    )
    serialized = [t.__dict__ if hasattr(t, '__dict__') else t for t in tasks]
    return {
        "owner_uid": owner_uid,
        "task_count": len(serialized),
        "tasks": serialized,
        "source_type": "MODEL_BACKED_DAILY_PLAN",
    }


@app.post("/planner/daily-build")
async def run_daily_build(
    payload: GenerateDailyPlanRequest,
    user: dict[str, Any] = Depends(current_user),
):
    owner_uid = payload.user_id or user["uid"]
    if owner_uid != user["uid"]:
        raise HTTPException(status_code=403, detail="Forbidden")

    planner = get_planner_service()
    tasks = await planner.generate_and_persist_daily_plan(
        owner_uid,
        force=True,
        target_date=None,
    )
    return {
        "owner_uid": owner_uid,
        "task_count": len(tasks),
        "tasks": [t.__dict__ if hasattr(t, '__dict__') else t for t in tasks],
    }


class RescheduleMissedBlockRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: str | None = None
    task_id: str


@app.post("/planner/reschedule-missed-block")
async def reschedule_missed_block(
    payload: RescheduleMissedBlockRequest,
    user: dict[str, Any] = Depends(current_user),
):
    owner_uid = payload.user_id or user["uid"]
    if owner_uid != user["uid"]:
        raise HTTPException(status_code=403, detail="Forbidden")

    planner = get_planner_service()
    success = await planner.reschedule_missed_block(owner_uid, payload.task_id)
    if not success:
        raise HTTPException(status_code=404, detail="Task not found or could not be rescheduled")
    return {"owner_uid": owner_uid, "success": True}


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


@app.post("/universities/search")
async def search_universities(
    payload: UniversitySearchRequest,
    user: dict[str, Any] = Depends(current_user),
):
    service = get_uni_catalog_service()
    results = await asyncio.to_thread(
        service.search_universities,
        query=payload.query,
        country=payload.country,
        name=payload.name,
        limit=payload.limit,
    )
    return {
        "results": results,
        "total": len(results),
        "source_type": "UNIVERSITY_CATALOG",
    }


@app.post("/universities/programs")
async def get_university_programs(
    payload: UniversityProgramsRequest,
    user: dict[str, Any] = Depends(current_user),
):
    service = get_program_crawler()
    programs = await asyncio.to_thread(
        service.get_programs,
        university_name=payload.university_name,
        country=payload.country,
        domain=payload.domain,
    )
    return {
        "university": payload.university_name,
        "programs": programs,
        "total": len(programs),
        "source_type": "UNIVERSITY_PROGRAMS",
    }


@app.post("/universities/normalize-degree")
async def normalize_degree(
    payload: NormalizeDegreeRequest,
    user: dict[str, Any] = Depends(current_user),
):
    service = get_uni_catalog_service()
    result = await asyncio.to_thread(
        service.normalize_degree,
        degree_name=payload.degree_name,
        country=payload.country,
    )
    return {
        "original": payload.degree_name,
        "normalized": result,
        "source_type": "DEGREE_NORMALIZER",
    }


@app.post("/api/ai/chat")
async def ai_chat(
    payload: AiChatRequest,
    user: dict[str, Any] = Depends(current_user),
):
    service = get_ai_proxy_service()

    if payload.stream:
        async def event_stream():
            async for chunk in service.chat_stream(
                messages=payload.messages,
                user_id=user["uid"],
                max_tokens=payload.max_tokens,
                temperature=payload.temperature,
            ):
                yield chunk

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    result = await service.chat(
        messages=payload.messages,
        user_id=user["uid"],
        stream=False,
        max_tokens=payload.max_tokens,
        temperature=payload.temperature,
    )

    if result and "error" in result:
        status_code = 429 if result["error"] == "rate_limited" else 503
        raise HTTPException(status_code=status_code, detail=result["message"])

    return result


@app.get("/api/ai/status")
async def ai_status(user: dict[str, Any] = Depends(current_user)):
    service = get_ai_proxy_service()
    return service.get_status()


_CREDENTIAL_KEYS = [
    "SERPER_API_KEY",
    "CLOUDINARY_CLOUD_NAME",
    "CLOUDINARY_UPLOAD_PRESET",
    "GROK_API_KEY",
    "VERCEL_API_KEY",
    "OPENROUTER_API_KEY",
    "DEEPSEEK_API_KEY",
    "DEEPGRAM_API_KEY",
    "GOOGLE_PRIVATE_KEY",
]


@app.post("/api/credentials")
async def get_credentials(user: dict[str, Any] = Depends(current_user)):
    result = {}
    for key in _CREDENTIAL_KEYS:
        val = os.environ.get(key, "")
        if key == "GOOGLE_PRIVATE_KEY":
            val = val.replace("\\n", "\n")
        result[key.lower()] = val
    return result


@app.post("/api/search")
async def proxy_search(
    payload: dict[str, Any],
    user: dict[str, Any] = Depends(current_user),
):
    api_key = os.environ.get("SERPER_API_KEY")
    if not api_key:
        raise HTTPException(status_code=502, detail="Serper API key not configured")
    import httpx
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            json=payload,
        )
    if resp.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"Serper error: {resp.text[:300]}")
    return resp.json()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
