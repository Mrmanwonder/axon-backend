from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import math
import os
import re
import time
from typing import Any

from fastapi import HTTPException


class HandwritingGradingGateway:
    def __init__(self, api_key: str):
        if not api_key.strip():
            raise ValueError("GEMINI_API_KEY is required for handwriting grading")

        import google.generativeai as genai

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-1.5-flash")
        self._cloudinary_cloud_name = os.environ.get("CLOUDINARY_CLOUD_NAME", "").strip()
        self._cloudinary_api_key = os.environ.get("CLOUDINARY_API_KEY", "").strip()
        self._cloudinary_api_secret = os.environ.get("CLOUDINARY_API_SECRET", "").strip()

    async def grade_answer(
        self,
        *,
        image_url: str,
        marking_scheme: dict[str, Any],
        objective: str,
        question_prompt: str,
        learning_objective_ids: list[str],
        command_word: str,
        syllabus_context: str = "",
        spatial_layout: dict[str, Any] | None = None,
        cloudinary_public_id: str | None = None,
        archive_after_grading: bool = False,
    ) -> dict[str, Any]:
        image_part = await self._fetch_image_part(image_url)
        spatial_json = json.dumps(spatial_layout or {}, ensure_ascii=True)
        prompt = f"""
Role: Senior Examiner for {objective}.
Task: Grade the student's handwritten response from the supplied image.
Question: {question_prompt}
Learning Objectives: {", ".join(learning_objective_ids)}
Command Word: {command_word}
Syllabus Context: {syllabus_context or objective}
Spatial Layout Metadata: {spatial_json}
Marking Scheme JSON: {json.dumps(marking_scheme, ensure_ascii=True)}

Multimodal Rules:
1. Evaluate both text content and spatial structure. Diagrams, labelled parts, force arrows,
   derivations, and equation layout all count when they satisfy a marking point.
2. If the answer contains a circuit, graph, biological sketch, or multi-line derivation,
   judge whether the spatial arrangement supports the marking point.
3. Map every awarded mark to a concrete marking point.
4. Map every missed mark to a concrete missing statement, label, or spatial feature.

Output strict JSON with keys:
score, available_marks, feedback, marks_awarded, marks_missed,
learning_objective_gaps, marking_point_extract, command_word_depth, semantic_match, source_type

Each marks_awarded / marks_missed item must include:
point, evidence

Each learning_objective_gap must include:
learning_objective_id, reason

Feedback style:
"You got 2/3. You correctly identified the formula, but you failed to state that
the resultant force is zero, which is a required marking point."
"""

        parsed = await self._generate_json([prompt, image_part], "Gemini grading failed")
        parsed.setdefault("source_type", "MULTIMODAL_SPATIAL_GRADING")
        parsed.setdefault("available_marks", parsed.get("score", 0))
        parsed.setdefault("marking_point_extract", [])
        parsed.setdefault("command_word_depth", {})
        parsed.setdefault("semantic_match", {})

        if archive_after_grading and cloudinary_public_id:
            parsed["archive_status"] = await self.archive_asset(cloudinary_public_id)

        return parsed

    async def grade_typed_answer(
        self,
        *,
        question_prompt: str,
        student_answer: str,
        marking_scheme_text: str,
        objective: str,
        command_word: str,
        learning_objective_ids: list[str],
        available_marks: int | float | None = None,
        syllabus_context: str = "",
    ) -> dict[str, Any]:
        if not student_answer.strip():
            return {
                "awarded_marks": 0,
                "available_marks": int(available_marks or 0),
                "feedback": "No answer submitted.",
                "marks_awarded": [],
                "marks_missed": [],
                "learning_objective_gaps": [
                    {
                        "learning_objective_id": (
                            learning_objective_ids[0] if learning_objective_ids else objective
                        ),
                        "reason": "No answer was provided.",
                    }
                ],
                "error_type": "unanswered",
                "marking_point_extract": [],
                "command_word_depth": self._analyze_command_word_depth(
                    command_word=command_word,
                    student_answer=student_answer,
                ),
                "semantic_match": {
                    "overall_score": 0.0,
                    "matched_points": [],
                    "top_matches": [],
                    "source_type": "EMBEDDING_HEURISTIC",
                },
            }

        command_word_depth = self._analyze_command_word_depth(
            command_word=command_word,
            student_answer=student_answer,
        )
        semantic_match = await self._build_semantic_match(
            student_answer=student_answer,
            marking_scheme_text=marking_scheme_text,
        )

        prompt = f"""
Role: Senior Examiner for {objective}.
Task: Grade the student's typed answer against the supplied marking scheme.
Question: {question_prompt}
Student Answer: {student_answer}
Command Word: {command_word}
Available Marks: {available_marks if available_marks is not None else 'unknown'}
Learning Objectives: {", ".join(learning_objective_ids)}
Syllabus Context: {syllabus_context or objective}
Command Word Depth Heuristic JSON:
{json.dumps(command_word_depth, ensure_ascii=True)}
Semantic Match JSON:
{json.dumps(semantic_match, ensure_ascii=True)}
Marking Scheme Text:
{marking_scheme_text}

Strict Rules:
1. Use ONLY the supplied marking scheme text.
2. Distinguish command-word misses from content misses.
3. Award marks only for explicitly present marking points.
4. Return strict JSON with keys:
   awarded_marks, available_marks, feedback, marks_awarded, marks_missed,
   learning_objective_gaps, error_type, marking_point_extract,
   command_word_depth, semantic_match
5. error_type must be one of:
   none, command_word_miss, learning_objective_gap, marking_point_miss
"""

        parsed = await self._generate_json(prompt, "Gemini typed grading failed")
        parsed["awarded_marks"] = float(parsed.get("awarded_marks", 0) or 0)
        parsed["available_marks"] = float(
            parsed.get("available_marks", available_marks or 0) or 0
        )
        parsed["feedback"] = str(parsed.get("feedback", "")).strip()
        parsed["marks_awarded"] = list(parsed.get("marks_awarded", []))
        parsed["marks_missed"] = list(parsed.get("marks_missed", []))
        parsed["learning_objective_gaps"] = list(parsed.get("learning_objective_gaps", []))
        parsed["error_type"] = str(parsed.get("error_type", "none")).strip() or "none"
        parsed["marking_point_extract"] = list(parsed.get("marking_point_extract", []))
        parsed["command_word_depth"] = self._merge_command_word_depth(
            command_word_depth,
            parsed.get("command_word_depth"),
        )
        parsed["semantic_match"] = self._merge_semantic_match(
            semantic_match,
            parsed.get("semantic_match"),
        )
        if (
            not parsed["command_word_depth"].get("depth_satisfied", False)
            and parsed["error_type"] in {"", "none", "marking_point_miss"}
            and parsed["awarded_marks"] < parsed["available_marks"]
        ):
            parsed["error_type"] = "command_word_miss"
        parsed["feedback"] = self._build_feedback_with_depth(
            feedback=parsed["feedback"],
            command_word_depth=parsed["command_word_depth"],
        )
        return parsed

    async def archive_asset(self, public_id: str) -> dict[str, Any]:
        if not public_id.strip():
            return {"status": "skipped", "reason": "missing_public_id"}
        if not (
            self._cloudinary_cloud_name
            and self._cloudinary_api_key
            and self._cloudinary_api_secret
        ):
            return {"status": "skipped", "reason": "cloudinary_credentials_unavailable"}

        import requests

        timestamp = str(int(time.time()))
        signature_payload = f"public_id={public_id}&tags=axon_archived&timestamp={timestamp}{self._cloudinary_api_secret}"
        signature = hashlib.sha1(signature_payload.encode("utf-8")).hexdigest()
        endpoint = (
            f"https://api.cloudinary.com/v1_1/{self._cloudinary_cloud_name}/image/explicit"
        )

        def post_update() -> dict[str, Any]:
            response = requests.post(
                endpoint,
                data={
                    "public_id": public_id,
                    "tags": "axon_archived",
                    "type": "upload",
                    "timestamp": timestamp,
                    "api_key": self._cloudinary_api_key,
                    "signature": signature,
                },
                timeout=20,
            )
            response.raise_for_status()
            payload = response.json()
            return {
                "status": "archived",
                "public_id": payload.get("public_id", public_id),
                "version": payload.get("version"),
            }

        try:
            return await asyncio.to_thread(post_update)
        except Exception as exc:
            return {"status": "failed", "reason": str(exc)}

    async def _fetch_image_part(self, image_url: str) -> dict[str, Any]:
        if not image_url.startswith("https://"):
            raise HTTPException(status_code=400, detail="image_url must be an HTTPS URL")

        import requests

        def fetch() -> tuple[bytes, str]:
            response = requests.get(image_url, timeout=20)
            response.raise_for_status()
            content_type = response.headers.get("content-type", "image/png").split(";")[0].strip()
            return response.content, content_type or "image/png"

        try:
            image_bytes, mime_type = await asyncio.to_thread(fetch)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Unable to fetch answer image: {exc}") from exc

        return {"mime_type": mime_type, "data": image_bytes}

    async def _generate_json(self, prompt: Any, error_prefix: str) -> dict[str, Any]:
        try:
            response = await self.model.generate_content_async(prompt)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"{error_prefix}: {exc}") from exc

        raw_text = getattr(response, "text", "").strip()
        if not raw_text:
            raise HTTPException(status_code=502, detail="Gemini returned an empty grading response")

        candidate = raw_text.replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=502,
                detail="Gemini grading response was not valid JSON",
            ) from exc

    def _normalize_command_word(self, command_word: str) -> str:
        return re.sub(r"[^a-z]", "", (command_word or "").strip().lower())

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"[A-Za-z0-9']+", text.lower())

    def _extract_marking_points(self, marking_scheme_text: str) -> list[str]:
        chunks: list[str] = []
        for raw_line in re.split(r"[\r\n]+", marking_scheme_text):
            line = " ".join(raw_line.strip().split())
            if not line:
                continue
            normalized = re.sub(r"^[\-\u2022*\d().\s]+", "", line).strip()
            if len(normalized) < 4:
                continue
            if len(normalized) > 220 and ";" in normalized:
                parts = [part.strip() for part in normalized.split(";") if part.strip()]
                chunks.extend(parts)
                continue
            chunks.append(normalized)
        deduped: list[str] = []
        seen: set[str] = set()
        for chunk in chunks:
            key = chunk.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(chunk)
        return deduped[:10]

    def _analyze_command_word_depth(
        self,
        *,
        command_word: str,
        student_answer: str,
    ) -> dict[str, Any]:
        normalized_word = self._normalize_command_word(command_word)
        normalized_answer = " ".join(student_answer.strip().split())
        words = self._tokenize(normalized_answer)
        word_count = len(words)
        lower_answer = normalized_answer.lower()
        sequence_markers = [
            "first",
            "then",
            "next",
            "after",
            "before",
            "finally",
            "initially",
            "subsequently",
        ]
        causal_markers = [
            "because",
            "therefore",
            "therefor",
            "thus",
            "hence",
            "so that",
            "so ",
            "since",
            "due to",
            "as a result",
            "results in",
            "leads to",
            "causes",
        ]
        observed_sequence = [marker for marker in sequence_markers if marker in lower_answer]
        observed_causal = [marker.strip() for marker in causal_markers if marker in lower_answer]
        missing_depth: list[str] = []
        depth_score = 0.0
        depth_satisfied = False
        expected_pattern = "Respond with the depth expected by the command word."

        if normalized_word == "state":
            expected_pattern = "Needs 1 to 5 words with no extra explanation."
            if word_count == 0:
                missing_depth.append("No answer supplied.")
            if word_count > 5:
                missing_depth.append("Trim the answer to a concise fact or definition.")
            if observed_causal:
                missing_depth.append("Do not add causal explanation to a state answer.")
            depth_satisfied = 1 <= word_count <= 5 and not observed_causal
            depth_score = 1.0 if depth_satisfied else (0.5 if word_count and word_count <= 8 else 0.0)
        elif normalized_word == "describe":
            expected_pattern = "Needs an ordered description of features or events."
            if word_count < 6:
                missing_depth.append("Add enough detail to show the sequence or observable features.")
            if not observed_sequence:
                missing_depth.append("Show order using sequence language such as first, then, or finally.")
            depth_satisfied = word_count >= 6 and bool(observed_sequence)
            depth_score = min(
                1.0,
                (0.45 if word_count >= 6 else word_count / 12.0)
                + (0.55 if observed_sequence else 0.0),
            )
        elif normalized_word == "explain":
            expected_pattern = "Needs a causal chain using because, therefore, so, or equivalent logic."
            if word_count < 8:
                missing_depth.append("Add a fuller why/how chain.")
            if not observed_causal:
                missing_depth.append("Make the cause-and-effect link explicit.")
            depth_satisfied = word_count >= 8 and bool(observed_causal)
            depth_score = min(
                1.0,
                (0.35 if word_count >= 8 else word_count / 16.0)
                + (0.65 if observed_causal else 0.0),
            )
        else:
            depth_satisfied = word_count > 0
            depth_score = 1.0 if depth_satisfied else 0.0
            if not depth_satisfied:
                missing_depth.append("No answer supplied.")

        return {
            "command_word": command_word,
            "word_count": word_count,
            "depth_satisfied": depth_satisfied,
            "depth_score": round(max(0.0, min(depth_score, 1.0)), 4),
            "expected_pattern": expected_pattern,
            "observed_features": {
                "sequence_markers": observed_sequence,
                "causal_markers": observed_causal,
            },
            "missing_depth": missing_depth,
            "source_type": "COMMAND_WORD_HEURISTIC",
        }

    async def _build_semantic_match(
        self,
        *,
        student_answer: str,
        marking_scheme_text: str,
    ) -> dict[str, Any]:
        points = self._extract_marking_points(marking_scheme_text)
        if not points:
            return {
                "overall_score": 0.0,
                "matched_points": [],
                "top_matches": [],
                "source_type": "EMBEDDING_HEURISTIC",
            }

        answer_embedding = await self._embed_text(student_answer)
        if answer_embedding is None:
            return self._lexical_semantic_match(student_answer=student_answer, points=points)

        matches: list[dict[str, Any]] = []
        for point in points:
            point_embedding = await self._embed_text(point)
            if point_embedding is None:
                return self._lexical_semantic_match(student_answer=student_answer, points=points)
            similarity = self._cosine_similarity(answer_embedding, point_embedding)
            matches.append(
                {
                    "point": point,
                    "similarity": round(similarity, 4),
                }
            )

        matches.sort(key=lambda item: item["similarity"], reverse=True)
        top_matches = matches[:3]
        matched_points = [
            item["point"] for item in matches if float(item["similarity"]) >= 0.72
        ]
        overall = (
            sum(float(item["similarity"]) for item in top_matches) / len(top_matches)
            if top_matches
            else 0.0
        )
        return {
            "overall_score": round(overall, 4),
            "matched_points": matched_points,
            "top_matches": top_matches,
            "source_type": "GEMINI_EMBEDDINGS",
        }

    async def _embed_text(self, text: str) -> list[float] | None:
        cleaned = " ".join(text.strip().split())
        if not cleaned:
            return None
        try:
            import google.generativeai as genai

            def run_embedding() -> Any:
                return genai.embed_content(
                    model="models/text-embedding-004",
                    content=cleaned,
                    task_type="semantic_similarity",
                )

            result = await asyncio.to_thread(run_embedding)
            embedding = result.get("embedding") if isinstance(result, dict) else None
            if isinstance(embedding, list) and embedding:
                return [float(value) for value in embedding]
        except Exception as exc:
            exc_name = type(exc).__name__
            if "quota" in str(exc).lower() or "rate_limit" in str(exc).lower() or "429" in str(exc):
                print(f"[GradingService] Gemini quota exceeded for embedding: {exc}")
            else:
                print(f"[GradingService] Embedding failed ({exc_name}): {exc}")
            return None
        return None

    def _lexical_semantic_match(
        self,
        *,
        student_answer: str,
        points: list[str],
    ) -> dict[str, Any]:
        answer_tokens = set(self._tokenize(student_answer))
        if not answer_tokens:
            return {
                "overall_score": 0.0,
                "matched_points": [],
                "top_matches": [],
                "source_type": "TOKEN_OVERLAP_FALLBACK",
            }

        matches: list[dict[str, Any]] = []
        for point in points:
            point_tokens = set(self._tokenize(point))
            if not point_tokens:
                continue
            overlap = len(answer_tokens & point_tokens)
            score = overlap / math.sqrt(len(answer_tokens) * len(point_tokens))
            matches.append({"point": point, "similarity": round(score, 4)})

        matches.sort(key=lambda item: item["similarity"], reverse=True)
        top_matches = matches[:3]
        matched_points = [
            item["point"] for item in matches if float(item["similarity"]) >= 0.45
        ]
        overall = (
            sum(float(item["similarity"]) for item in top_matches) / len(top_matches)
            if top_matches
            else 0.0
        )
        return {
            "overall_score": round(overall, 4),
            "matched_points": matched_points,
            "top_matches": top_matches,
            "source_type": "TOKEN_OVERLAP_FALLBACK",
        }

    def _cosine_similarity(self, left: list[float], right: list[float]) -> float:
        if not left or not right or len(left) != len(right):
            return 0.0
        numerator = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(a * a for a in left))
        right_norm = math.sqrt(sum(b * b for b in right))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return max(0.0, min(1.0, numerator / (left_norm * right_norm)))

    def _merge_command_word_depth(
        self,
        heuristic: dict[str, Any],
        model_value: Any,
    ) -> dict[str, Any]:
        merged = dict(heuristic)
        if isinstance(model_value, dict):
            merged.update(model_value)
        merged.setdefault("command_word", heuristic.get("command_word", ""))
        merged.setdefault("word_count", heuristic.get("word_count", 0))
        merged.setdefault("source_type", "COMMAND_WORD_HEURISTIC")
        return merged

    def _merge_semantic_match(
        self,
        heuristic: dict[str, Any],
        model_value: Any,
    ) -> dict[str, Any]:
        merged = dict(heuristic)
        if isinstance(model_value, dict):
            merged.update(model_value)
        merged.setdefault("overall_score", heuristic.get("overall_score", 0.0))
        merged.setdefault("matched_points", heuristic.get("matched_points", []))
        merged.setdefault("top_matches", heuristic.get("top_matches", []))
        merged.setdefault("source_type", heuristic.get("source_type", "EMBEDDING_HEURISTIC"))
        return merged

    def _build_feedback_with_depth(
        self,
        *,
        feedback: str,
        command_word_depth: dict[str, Any],
    ) -> str:
        base = feedback.strip()
        if command_word_depth.get("depth_satisfied", False):
            return base
        missing = command_word_depth.get("missing_depth") or []
        if not missing:
            return base
        depth_note = " ".join(str(item).strip() for item in missing if str(item).strip())
        if not depth_note:
            return base
        if not base:
            return depth_note
        if depth_note.lower() in base.lower():
            return base
        return f"{base} {depth_note}".strip()
