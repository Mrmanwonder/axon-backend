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
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from middleware.auth import current_user, optional_current_user, initialize_firebase
from middleware.supabase_scope import apply_user_scope, assert_table_access
from middleware.url_safety import assert_safe_https_url
from services.exam_dates_service import OfficialExamDatesService
from services.grading_service import HandwritingGradingGateway
from services.planner_service import DailyPlannerServiceV2
from services.study_pulse_service import StudyPulseService
from services.university_catalog_service import UniversityCatalogService
from services.university_program_crawler import UniversityProgramCrawler
from services.ai_proxy_service import AiProxyService
from services.deepgram_auth_service import DeepgramAuthService
from middleware.rate_limit import cors_allowed_origins, is_rate_limited


from fastapi import Depends
from typing import Annotated

# Global dependency to make auth optional
async def optional_user():
    return {"uid": "anonymous", "email": "anonymous@example.com"}

app = FastAPI(
    title="Axon Backend",
    version="2.1.0",
    description="ASGI backend for Axon document analysis, grading, and trust-safe sync.",
)


def custom_openapi() -> dict[str, Any]:
    """Return OpenAPI schema; fall back to route-only schema if Pydantic schema generation fails."""
    if app.openapi_schema:
        return app.openapi_schema
    try:
        app.openapi_schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
        )
    except Exception as exc:
        paths: dict[str, Any] = {}
        for route in app.routes:
            path = getattr(route, "path", None)
            methods = getattr(route, "methods", None)
            if not path or not methods:
                continue
            paths[path] = {
                method.lower(): {
                    "summary": getattr(route, "name", path),
                    "responses": {"200": {"description": "Successful Response"}},
                }
                for method in sorted(methods)
                if method not in {"HEAD", "OPTIONS"}
            }
        app.openapi_schema = {
            "openapi": "3.1.0",
            "info": {"title": app.title, "version": app.version, "description": app.description},
            "paths": paths,
            "x-openapi-fallback": True,
            "x-openapi-error": str(exc)[:500],
        }
    return app.openapi_schema


app.openapi = custom_openapi

# Security: CORS (configure via CORS_ALLOWED_ORIGINS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_allowed_origins(),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Content-Type", "Authorization", "Accept", "X-Client-Version", "X-Client-Platform"],
)


@app.middleware("http")
async def rate_limit_middleware(request, call_next):
    # DEBUG: Log all incoming requests with auth
    auth_header = request.headers.get("Authorization", "None")[:50]
    print(f">>> {request.method} {request.url.path} Auth:{auth_header}...")

    if request.url.path == "/health":
        return await call_next(request)

    client_ip = request.client.host if request.client else "unknown"
    auth_header = request.headers.get("Authorization", "")
    identifier = (
        auth_header.split(" ", 1)[1][:32]
        if auth_header.startswith("Bearer ")
        else client_ip
    )

    if is_rate_limited(identifier, request.url.path):
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded"},
        )

    return await call_next(request)


@app.middleware("http")
async def security_headers_middleware(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    if request.url.scheme == "https":
        response.headers["Strict-Transport-Security"] = (
            "max-age=31536000; includeSubDomains"
        )
    return response


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
_deepgram_auth_service: DeepgramAuthService | None = None
_drive_service = None

MAX_DRIVE_DOWNLOAD_BYTES = 600 * 1024 * 1024
_DRIVE_FILE_ID_PATTERN = r"^[a-zA-Z0-9_-]{10,}$"


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
    try:
        assert_safe_https_url(url)
        return True
    except ValueError:
        return False


MAX_PDF_BASE64_CHARS = 20 * 1024 * 1024  # ~15 MB decoded
MAX_IMAGE_BASE64_CHARS = 8 * 1024 * 1024


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
                advisor_model = genai.GenerativeModel(
                    "gemini-1.5-flash",
                    generation_config=genai.GenerationConfig(
                        response_mime_type="application/json"
                    )
                )
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


def get_deepgram_auth_service() -> DeepgramAuthService:
    global _deepgram_auth_service
    if _deepgram_auth_service is None:
        _deepgram_auth_service = DeepgramAuthService()
    return _deepgram_auth_service


def get_drive_service():
    global _drive_service
    if _drive_service is None:
        from services.google_drive_service import GoogleDriveService

        _drive_service = GoogleDriveService()
    return _drive_service


def assert_drive_file_id(file_id: str) -> str:
    import re

    if not re.fullmatch(_DRIVE_FILE_ID_PATTERN, file_id or ""):
        raise HTTPException(status_code=400, detail="Invalid file id")
    return file_id


class AnalyzePdfRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pdf_base64: str | None = None
    pdf_url: str | None = None
    filename: str = Field(default="upload.pdf", max_length=255)

    @model_validator(mode="after")
    def ensure_pdf_source(self):
        if not self.pdf_base64 and not self.pdf_url:
            raise ValueError("pdf_base64 or pdf_url is required")
        if self.pdf_url and not validate_https_url(self.pdf_url):
            raise ValueError("pdf_url must be a public HTTPS URL")
        if self.pdf_base64 and len(self.pdf_base64) > MAX_PDF_BASE64_CHARS:
            raise ValueError("pdf_base64 exceeds maximum allowed size")
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
        if self.image_data and len(self.image_data) > MAX_IMAGE_BASE64_CHARS:
            raise ValueError("image_data exceeds maximum allowed size")
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


# Public tables - accessible without auth
PUBLIC_SUPABASE_TABLES = {
    "boards", "subjects", "chapters", "Datesheet", "PYQs",
    "curriculum",  # Unified table
    "global_notes", "subchapter_notes", "papers", "topics",
    "sme_questions",  # SaveMyExams scraped questions
}

# User-scoped tables - require auth
USER_SCOPED_TABLES = {
    "user_notes", "user_subjects", "user_subchapter_progress",
    "user_pyqs", "user_mocks", "study_progress",
    "user_bookmarks", "user_recent_papers",
    "user_personal_index",
}

ALLOWED_SUPABASE_TABLES = PUBLIC_SUPABASE_TABLES | USER_SCOPED_TABLES


# Public endpoint - no auth required for public tables
@app.post("/supabase/query")
async def supabase_query(
    payload: SupabaseQueryRequest,
    user: dict[str, Any] | None = Depends(optional_current_user),
):
    if payload.table not in ALLOWED_SUPABASE_TABLES:
        raise HTTPException(status_code=403, detail=f"Table '{payload.table}' not allowed")

    # Check if public table - no auth needed
    is_public = payload.table in PUBLIC_SUPABASE_TABLES

    if not is_public and user is None:
        raise HTTPException(status_code=401, detail="Authentication required for user-scoped tables")

    # Get user_id safely (None for anonymous on public tables)
    user_id = user.get("uid") if user else None

    try:
        assert_table_access(payload.table, payload.method)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc

    scoped_params = apply_user_scope(
        payload.table,
        payload.method,
        payload.params,
        user_id,
    )

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
                resp = await client.get(url, headers=headers, params=scoped_params)
            elif payload.method == "insert":
                body = scoped_params.pop("body", [])
                resp = await client.post(url, headers=headers, json=body, params=scoped_params)
            elif payload.method == "upsert":
                body = scoped_params.pop("body", [])
                headers["Prefer"] = "resolution=merge-duplicates"
                resp = await client.post(url, headers=headers, json=body, params=scoped_params)
            elif payload.method == "update":
                body = scoped_params.pop("body", {})
                resp = await client.patch(url, headers=headers, json=body, params=scoped_params)
            elif payload.method == "delete":
                resp = await client.delete(url, headers=headers, params=scoped_params)
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

    messages: list[dict[str, str]] = Field(min_length=1, max_length=50)
    stream: bool = False
    max_tokens: int | None = Field(default=None, ge=1, le=16384)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)

    @model_validator(mode="after")
    def validate_messages(self):
        allowed_roles = {"system", "user", "assistant"}
        for entry in self.messages:
            role = entry.get("role", "")
            if role not in allowed_roles:
                raise ValueError(f"Invalid message role: {role}")
            content = entry.get("content", "")
            if not isinstance(content, str) or not content.strip():
                raise ValueError("Message content must be a non-empty string")
            if len(content) > 12000:
                raise ValueError("Message content exceeds maximum length")
        return self


async def download_or_decode_image(payload: BaseModel) -> bytes:
    if payload.image_data:
        if payload.image_data.startswith("data:"):
            return base64.b64decode(payload.image_data.split(",", 1)[1])
        raise HTTPException(status_code=400, detail="image_data must be a data URL")

    import requests

    def fetch() -> bytes:
        assert_safe_https_url(payload.image_url)
        response = requests.get(
            payload.image_url,
            timeout=20,
            allow_redirects=False,
        )
        if response.status_code in {301, 302, 303, 307, 308}:
            location = response.headers.get("Location")
            if not location or not validate_https_url(location):
                raise requests.HTTPError("Unsafe redirect target")
            response = requests.get(location, timeout=20, allow_redirects=False)
        response.raise_for_status()
        content = response.content
        if len(content) > MAX_IMAGE_BASE64_CHARS:
            raise ValueError("Downloaded image exceeds maximum allowed size")
        return content

    try:
        return await asyncio.to_thread(fetch)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


async def load_pdf_bytes(payload: AnalyzePdfRequest) -> tuple[bytes, str]:
    if payload.pdf_base64:
        return base64.b64decode(payload.pdf_base64), payload.filename

    import requests

    def fetch() -> tuple[bytes, str]:
        assert_safe_https_url(payload.pdf_url)
        response = requests.get(
            payload.pdf_url,
            timeout=30,
            allow_redirects=False,
        )
        if response.status_code in {301, 302, 303, 307, 308}:
            location = response.headers.get("Location")
            if not location or not validate_https_url(location):
                raise requests.HTTPError("Unsafe redirect target")
            response = requests.get(location, timeout=30, allow_redirects=False)
        response.raise_for_status()
        content = response.content
        if len(content) > MAX_PDF_BASE64_CHARS:
            raise ValueError("Downloaded PDF exceeds maximum allowed size")
        filename = os.path.basename(urlparse(payload.pdf_url).path) or payload.filename
        return content, filename

    try:
        return await asyncio.to_thread(fetch)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


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
        "version": "2.1.1",  # DEBUG: v2 endpoint added in e82b52e
        "firebase_admin_ready": True,
        "firestore_database_id": _firestore_database_id,
        "transport": "fastapi",
    }


# Public debug endpoint - no auth required
@app.get("/ping")
async def ping():
    """Lightweight ping to check backend is awake"""
    ai_service = get_ai_proxy_service()
    return {
        "status": "awake",
        "ai_configured": len(ai_service._provider_configs) > 0,
        "model": ai_service._provider_configs.get("openrouter", {}).get("model", "none"),
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
    request: Request,
):
    # Bypass auth check for testing - any request works
    user_id = payload.user_id or request.headers.get("X-User-Id", "test_user")
    owner_uid = user_id

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


class SerperSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    q: str = Field(min_length=1, max_length=500)
    num: int = Field(default=10, ge=1, le=20)
    gl: str | None = Field(default=None, max_length=8)
    hl: str | None = Field(default=None, max_length=8)


@app.get("/api/service-capabilities")
async def service_capabilities(user: dict[str, Any] = Depends(current_user)):
    del user  # authenticated access only
    ai_status = get_ai_proxy_service().get_status()
    return {
        "ai_chat": ai_status.get("provider_count", 0) > 0,
        "search": bool(os.environ.get("SERPER_API_KEY")),
        "cloudinary": bool(
            os.environ.get("CLOUDINARY_CLOUD_NAME")
            and os.environ.get("CLOUDINARY_UPLOAD_PRESET")
        ),
        "cloudinary_cloud_name": os.environ.get("CLOUDINARY_CLOUD_NAME", ""),
        "cloudinary_upload_preset": os.environ.get("CLOUDINARY_UPLOAD_PRESET", ""),
        "speech": bool(os.environ.get("DEEPGRAM_API_KEY")),
        "drive_sync": bool(os.environ.get("GOOGLE_PRIVATE_KEY")),
    }


@app.post("/api/credentials")
async def get_credentials_deprecated(user: dict[str, Any] = Depends(current_user)):
    """Deprecated: never expose server secrets to clients."""
    return await service_capabilities(user)


@app.post("/api/search")
async def proxy_search(
    payload: SerperSearchRequest,
    user: dict[str, Any] = Depends(current_user),
):
    del user
    api_key = os.environ.get("SERPER_API_KEY")
    if not api_key:
        raise HTTPException(status_code=502, detail="Serper API key not configured")
    import httpx

    serper_body: dict[str, Any] = {"q": payload.q, "num": payload.num}
    if payload.gl:
        serper_body["gl"] = payload.gl
    if payload.hl:
        serper_body["hl"] = payload.hl

    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            json=serper_body,
        )
    if resp.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"Serper error: {resp.text[:300]}")
    return resp.json()


class DeepgramTokenRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ttl_seconds: int | None = Field(default=None, ge=30, le=300)


@app.get("/api/drive/status")
async def drive_status(user: dict[str, Any] = Depends(current_user)):
    del user
    configured = bool(os.environ.get("GOOGLE_PRIVATE_KEY", "").strip())
    return {"configured": configured, "source_type": "GOOGLE_DRIVE_PROXY"}


@app.get("/api/drive/folders/find")
async def drive_find_folder(
    name: str,
    user: dict[str, Any] = Depends(current_user),
):
    del user
    clean_name = name.strip()
    if not clean_name or len(clean_name) > 128:
        raise HTTPException(status_code=400, detail="Invalid folder name")

    try:
        service = get_drive_service()
        folder_id = await asyncio.to_thread(service.find_folder, clean_name)
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return {"name": clean_name, "folder_id": folder_id}


@app.get("/api/drive/folders/{folder_id}/items")
async def drive_list_items(
    folder_id: str,
    user: dict[str, Any] = Depends(current_user),
):
    del user
    assert_drive_file_id(folder_id)

    try:
        service = get_drive_service()
        items = await asyncio.to_thread(service.list_items, folder_id)
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return {"items": items, "total": len(items)}


@app.get("/api/drive/files/{file_id}/metadata")
async def drive_file_metadata(
    file_id: str,
    user: dict[str, Any] = Depends(current_user),
):
    del user
    assert_drive_file_id(file_id)

    try:
        service = get_drive_service()
        metadata = await asyncio.to_thread(service.get_file_info, file_id)
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return metadata


@app.get("/api/drive/files/{file_id}/content")
async def drive_file_content(
    file_id: str,
    user: dict[str, Any] = Depends(current_user),
):
    del user
    assert_drive_file_id(file_id)

    try:
        service = get_drive_service()
        metadata = await asyncio.to_thread(service.get_file_info, file_id)
        file_size = int(metadata.get("size", 0) or 0)
        if file_size > MAX_DRIVE_DOWNLOAD_BYTES:
            raise HTTPException(status_code=413, detail="File exceeds download limit")

        payload = await asyncio.to_thread(service.download_file, file_id)
        content = payload.getvalue()
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    mime_type = metadata.get("mimeType") or "application/octet-stream"
    filename = metadata.get("name") or "download"
    return Response(
        content=content,
        media_type=mime_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/api/speech/deepgram/token")
async def deepgram_token(
    user: dict[str, Any] = Depends(current_user),
    payload: DeepgramTokenRequest | None = None,
):
    del user
    service = get_deepgram_auth_service()
    if not service.is_configured:
        raise HTTPException(status_code=503, detail="Speech service not configured")

    try:
        return await service.grant_token(
            ttl_seconds=payload.ttl_seconds if payload else None
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/api/speech/deepgram/transcribe")
async def deepgram_transcribe(
    user: dict[str, Any] = Depends(current_user),
    file: UploadFile = File(...),
):
    del user
    service = get_deepgram_auth_service()
    if not service.is_configured:
        raise HTTPException(status_code=503, detail="Speech service not configured")

    audio = await file.read()
    if not audio:
        raise HTTPException(status_code=400, detail="Empty audio payload")
    if len(audio) > 25 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Audio file too large")

    content_type = file.content_type or "audio/wav"
    try:
        transcript = await service.transcribe_bytes(audio, content_type=content_type)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    if not transcript:
        raise HTTPException(status_code=422, detail="No speech detected")

    return {"transcript": transcript, "source_type": "DEEPGRAM_PROXY"}


# ══════════════════════════════════════════════════════════════
# V2 DAILY PLAN - No Firestore dependency, works offline
# ══════════════════════════════════════════════════════════════

class V2DailyPlanRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    user_id: str | None = None
    subjects: list[str] | None = Field(default=None, max_length=20)
    target_hours: float | None = Field(default=4.0, ge=1.0, le=16.0)
    client_date: str | None = Field(default=None, max_length=32)
    focus_areas: str | None = Field(default=None, max_length=800)
    force: bool = False
    exam_dates: dict[str, str] | None = Field(default=None)  # {"Maths": "2026-06-15"}


@app.post("/api/v2/daily-plan/generate")
async def v2_generate_daily_plan(payload: V2DailyPlanRequest):
    """
    Firestore-free daily plan generator.
    Returns a complete daily plan using the same algorithm as planner_service.py
    but without any Firestore dependency.
    """
    from datetime import date, datetime, timedelta
    import hashlib
    import math
    import random

    today = (date.fromisoformat(payload.client_date) if payload.client_date else date.today()).isoformat()
    subjects = payload.subjects or ["Mathematics", "Physics", "Chemistry"]
    target_hours = payload.target_hours or 4.0
    exam_dates = payload.exam_dates or {}
    focus_areas = payload.focus_areas or ""

    # ── Phase detection from nearest exam ──
    days_to_exam = 999
    for subj in subjects:
        ed_str = exam_dates.get(subj)
        if ed_str:
            try:
                d = date.fromisoformat(ed_str)
                dt = (d - date.today()).days
                if 0 <= dt < days_to_exam:
                    days_to_exam = dt
            except ValueError:
                pass

    if days_to_exam <= 7:
        phase = "T-7 Mock Sprint"
        phase_key = "t7"
    elif days_to_exam <= 14:
        phase = "T-14 Deep Dive"
        phase_key = "t14"
    elif days_to_exam <= 30:
        phase = "T-30 Completion"
        phase_key = "t30"
    else:
        phase = "Foundation Build"
        phase_key = "foundation"

    # ── Phase-adaptive task mix (from planner_service.py) ──
    phase_mix = {
        "foundation": [("deep_work", 0.55), ("practice", 0.20), ("review", 0.15), ("flashcards", 0.10)],
        "t30": [("practice", 0.35), ("past_paper", 0.20), ("deep_work", 0.20), ("review", 0.15), ("flashcards", 0.10)],
        "t14": [("past_paper", 0.40), ("command_word_drill", 0.20), ("review", 0.20), ("examiner_report", 0.10), ("flashcards", 0.10)],
        "t7": [("mock_exam", 0.60), ("examiner_report", 0.15), ("command_word_drill", 0.15), ("flashcards", 0.10)],
    }

    # ── Cognitive Load Units ──
    clu_map = {
        "mock_exam": 10.0, "past_paper": 7.0, "deep_work": 6.0,
        "command_word_drill": 5.0, "practice": 5.0,
        "examiner_report": 3.0, "review": 3.0, "flashcards": 2.0,
    }
    daily_clu_budget = 32.0

    # ── Task durations (minutes) ──
    dur_map = {
        "mock_exam": 120, "past_paper": 60, "deep_work": 50,
        "practice": 40, "command_word_drill": 30,
        "review": 30, "examiner_report": 25, "flashcards": 20,
    }

    # ── Time windows ──
    total_minutes = int(target_hours * 60)
    windows = [
        {"name": "peak_focus_morning", "pct": 0.40},
        {"name": "structured_morning", "pct": 0.20},
        # 20-min break
        {"name": "afternoon", "pct": 0.20},
        # 30-min break
        {"name": "review_evening", "pct": 0.15},
        {"name": "light_evening", "pct": 0.05},
    ]

    # ── Build slots ──
    from datetime import time as dt_time
    now_dt = datetime.now()
    start_hour = max(now_dt.hour + 1, 8)
    if start_hour >= 24:
        start_hour = 8
    cursor = datetime.combine(date.today(), dt_time(hour=start_hour))
    slots = []
    for w in windows:
        dur = int(total_minutes * w["pct"])
        end = cursor + timedelta(minutes=dur)
        slots.append({"window": w["name"], "start": cursor, "end": end, "clu_remaining": daily_clu_budget * w["pct"]})
        cursor = end + (timedelta(minutes=20) if w["name"] == "structured_morning" else timedelta(minutes=(30 if w["name"] == "afternoon" else 5)))

    # ── Sample tasks per subject based on phase ──
    mix = phase_mix.get(phase_key, phase_mix["foundation"])
    tasks = []
    slot_idx = 0
    clu_used = 0.0
    subject_counts = {}
    last_intensity = None
    task_num = 0

    # Round-robin through subjects
    subject_queue = list(subjects) * 3  # repeat for enough coverage
    random.seed(hash(today))
    random.shuffle(subject_queue)

    for si, subj in enumerate(subject_queue):
        if slot_idx >= len(slots): break
        if clu_used >= daily_clu_budget: break

        # Pick task type from phase mix (cycle through)
        tt_name, _ = mix[task_num % len(mix)]
        clu = clu_map.get(tt_name, 5.0)

        # Cognitive load guard
        if clu_used + clu > daily_clu_budget: continue

        # Max 3 tasks per subject
        subject_counts[subj] = subject_counts.get(subj, 0) + 1
        if subject_counts[subj] > 3: continue

        # Intensity
        if days_to_exam <= 3 or (days_to_exam <= 14 and task_num <= 1):
            intensity = "red"
            score = round(0.75 + random.random() * 0.25, 3)
        elif days_to_exam <= 30:
            intensity = "orange"
            score = round(0.45 + random.random() * 0.30, 3)
        else:
            intensity = "blue"
            score = round(0.2 + random.random() * 0.25, 3)

        # No back-to-back red
        if last_intensity == "red" and intensity == "red":
            intensity = "orange"
            score = round(0.5 + random.random() * 0.25, 3)

        # Find slot
        slot = None
        for j in range(slot_idx, len(slots)):
            if slots[j]["clu_remaining"] >= clu and slots[j]["start"] < slots[j]["end"]:
                slot = slots[j]
                slot_idx = j + 1
                break
        if not slot: break

        # Consume CLU
        slot["clu_remaining"] -= clu
        clu_used += clu
        last_intensity = intensity

        # Task title prefixes
        prefix_map = {
            "deep_work": "Master", "practice": "Practice",
            "review": "Review", "past_paper": "Past Paper -",
            "flashcards": "Flashcards -", "mock_exam": "Mock Exam -",
            "command_word_drill": "Command Drill -",
            "examiner_report": "Examiner Notes -",
        }
        prefix = prefix_map.get(tt_name, "Study")

        # Generate topic-like name
        topics_by_subject = {
            "Mathematics": ["Algebra & Functions", "Calculus", "Trigonometry", "Probability & Statistics", "Vectors & Mechanics"],
            "Physics": ["Mechanics", "Waves & Optics", "Electricity & Magnetism", "Thermal Physics", "Modern Physics"],
            "Chemistry": ["Organic Chemistry", "Physical Chemistry", "Inorganic Chemistry", "Electrochemistry", "Kinetics"],
            "Biology": ["Cell Biology", "Genetics", "Ecology", "Human Physiology", "Biochemistry"],
            "Economics": ["Microeconomics", "Macroeconomics", "International Trade", "Market Failure"],
            "Computer Science": ["Algorithms & Data Structures", "Databases", "Networking", "Programming Paradigms"],
        }
        topics = topics_by_subject.get(subj, ["Core Concepts", "Advanced Topics", "Problem Solving", "Theory & Application"])
        topic = topics[task_num % len(topics)]

        duration = dur_map.get(tt_name, 40)
        task_start = slot["start"]
        task_end = task_start + timedelta(minutes=duration)

        stable_id = hashlib.sha1(f"{today}|{subj}|{tt_name}|{task_start.isoformat()}".encode()).hexdigest()[:24]

        priority = 3 if intensity == "red" else (2 if intensity == "orange" else 1)

        tasks.append({
            "id": f"plan_{stable_id}",
            "title": f"{prefix} {topic}",
            "subject": subj,
            "description": f"Phase: {phase} | Focus on key concepts. {'Use mark-scheme after completing.' if tt_name in ('past_paper', 'mock_exam') else 'Take notes and attempt practice problems.'}",
            "paper": "Paper 1" if tt_name in ("past_paper", "mock_exam") else "",
            "objective_id": "",
            "start_time": task_start.isoformat(),
            "end_time": task_end.isoformat(),
            "status": "pending",
            "date": today,
            "reason": f"Score {score:.3f} | {phase} | {max(0, days_to_exam)}d to exam | target A*",
            "intensity_score": score,
            "intensity_label": intensity,
            "phase": phase,
            "anchor_date": today,
            "task_type": tt_name,
            "scheduled_window": slot["window"],
            "priority": priority,
            "is_completed": False,
            "is_sync_to_google": False,
        })

        slot["start"] = task_end + timedelta(minutes=(10 if tt_name in ("deep_work", "past_paper", "mock_exam") else 5))
        task_num += 1

    return {
        "owner_uid": payload.user_id or "v2_user",
        "task_count": len(tasks),
        "tasks": tasks,
        "source_type": "V2_OFFLINE_DAILY_PLAN",
        "phase": phase,
        "days_to_exam": max(0, days_to_exam),
        "target_hours": target_hours,
    }


class SMEImportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    csv_content: str = Field(min_length=1)


@app.post("/sme/questions/import")
async def import_sme_questions(
    payload: SMEImportRequest,
    user: dict[str, Any] = Depends(current_user),
):
    """Import questions from SaveMyExams CSV data into Supabase."""
    import io

    # Parse CSV
    reader = csv.DictReader(io.StringIO(payload.csv_content))
    records = []
    for row in reader:
        records.append({
            "subject": row.get("subject", ""),
            "topic": row.get("topic", ""),
            "question": row.get("question", ""),
            "source": "savemyexams",
        })

    if not records:
        raise HTTPException(status_code=400, detail="No records in CSV")

    # Insert into Supabase
    from services.supabase_client import get_supabase_client
    supabase = get_supabase_client()

    try:
        result = supabase.table("sme_questions").upsert(records).execute()
    except Exception as e:
        # Try creating table first
        try:
            supabase.rpc("create_sme_questions_table").execute()
        except:
            pass
        try:
            result = supabase.table("sme_questions").upsert(records).execute()
        except Exception as e2:
            raise HTTPException(status_code=500, detail=f"Import failed: {str(e2)}")

    return {
        "imported": len(records),
        "source_type": "SAVEMYEXAMS_IMPORTER",
    }


@app.get("/sme/questions")
async def get_sme_questions(
    subject: str | None = None,
    topic: str | None = None,
    user: dict[str, Any] | None = Depends(optional_current_user),
):
    """Get saved questions from SaveMyExams."""
    import requests
    SUPABASE_URL = "https://anmfwzxyvqxyxxeobxti.supabase.co"
    SUPABASE_KEY = "sb_publishable_fIbfGtT5yyFaogq4DQAuxw_tZ54kolM"
    
    params = {}
    if subject:
        params["subject"] = f"eq.{subject}"
    if topic:
        params["topic"] = f"eq.{topic}"
    
    resp = requests.get(
        f"{SUPABASE_URL}/rest/v1/sme_questions",
        headers={"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"},
        params=params,
        timeout=15
    )
    
    questions = resp.json() if resp.status_code == 200 else []
    return {
        "questions": questions,
        "total": len(questions),
        "source_type": "SAVEMYEXAMS",
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
