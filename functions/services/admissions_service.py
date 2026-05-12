from __future__ import annotations

import json
from typing import Any

from fastapi import HTTPException


class UniversityAdmissionsService:
    def __init__(self, db, model) -> None:
        self._db = db
        self._model = model

    async def build_fit_recommendations(
        self,
        *,
        user_id: str,
        predicted_grades: dict[str, str],
        target_course: str,
        countries: list[str],
        portfolio_links: list[str],
        portfolio_evidence: list[dict[str, str]],
        readiness_score: float,
    ) -> dict[str, Any]:
        if self._model is None:
            raise HTTPException(
                status_code=503,
                detail="Gemini is required for admissions fit recommendations",
            )

        profile = self._db.collection("users_private").document(user_id).get()
        profile_data = profile.to_dict() if profile.exists else {}

        prompt = f"""
Role: Global university admissions strategist for UK and Singapore applicants.

Student Board: {profile_data.get("board", "")}
Predicted Grades: {json.dumps(predicted_grades, ensure_ascii=True)}
Target Course: {target_course}
Countries: {json.dumps(countries, ensure_ascii=True)}
Readiness Score: {readiness_score}
Portfolio Links: {json.dumps(portfolio_links, ensure_ascii=True)}
Portfolio Evidence: {json.dumps(portfolio_evidence, ensure_ascii=True)}

Constraints:
- Use realistic public-facing admissions reasoning for current Oxford, Cambridge, Imperial, NUS, and similar top-tier patterns.
- Do not claim guaranteed admission.
- If exact requirements are uncertain, say they should be verified on the official university page and keep the recommendation conservative.
- Use readiness score as a risk signal, not as an official admissions criterion.

Return strict JSON with keys:
- safety
- match
- reach
- strategy_notes

Each university entry must include:
- name
- country
- rationale
- grade_fit
- readiness_fit
- portfolio_signal
- entry_requirements
- next_step
- source_type

Source type should be one of:
- latest_public_pattern
- conservative_inference

You must explicitly separate grade strength from application-evidence strength.
"""

        try:
            response = await self._model.generate_content_async(prompt)
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Gemini admissions fit failed: {exc}",
            ) from exc

        raw_text = getattr(response, "text", "").strip()
        if not raw_text:
            raise HTTPException(
                status_code=502,
                detail="Gemini returned an empty admissions fit response",
            )

        candidate = raw_text.replace("```json", "").replace("```", "").strip()
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=502,
                detail="Admissions fit response was not valid JSON",
            ) from exc

        parsed.setdefault("safety", [])
        parsed.setdefault("match", [])
        parsed.setdefault("reach", [])
        parsed.setdefault("strategy_notes", "")

        for bucket in ("safety", "match", "reach"):
            cleaned_entries = []
            for item in parsed.get(bucket, []):
                if not isinstance(item, dict):
                    continue
                cleaned_entries.append(
                    {
                        "name": str(item.get("name", "")),
                        "country": str(item.get("country", "")),
                        "rationale": str(item.get("rationale", "")),
                        "grade_fit": str(item.get("grade_fit", "")),
                        "readiness_fit": str(item.get("readiness_fit", "")),
                        "portfolio_signal": str(item.get("portfolio_signal", "")),
                        "entry_requirements": str(item.get("entry_requirements", "")),
                        "next_step": str(item.get("next_step", "")),
                        "source_type": str(
                            item.get("source_type", "conservative_inference")
                        ),
                    }
                )
            parsed[bucket] = cleaned_entries

        return parsed
