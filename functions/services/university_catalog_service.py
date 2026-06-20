from __future__ import annotations

import hashlib
import json
from typing import Any

import requests


class UniversityCatalogService:
    HIPO_API = "http://universities.hipolabs.com/search"

    def __init__(self, db, model=None):
        self._db = db
        self._model = model

    def search_universities(
        self,
        query: str = "",
        country: str = "",
        name: str = "",
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        params: dict[str, str] = {}
        if query:
            params["name"] = query
        if country:
            params["country"] = country
        if name:
            params["name"] = name

        try:
            resp = requests.get(self.HIPO_API, params=params, timeout=15)
            resp.raise_for_status()
            results = resp.json()
        except Exception as exc:
            print(f"[UniversityCatalog] Hipo API error: {exc}")
            results = self._search_local_cache(query, country)
            return results[:limit]

        enriched = []
        for uni in results[:limit]:
            enriched.append(self._enrich_university(uni))

        self._cache_results(enriched)
        return enriched

    def get_university_programs(
        self,
        university_name: str,
        country: str = "",
        domain: str = "",
    ) -> list[dict[str, Any]]:
        cached = self._get_cached_programs(university_name)
        if cached:
            return cached

        programs = self._generate_programs_with_llm(
            university_name=university_name,
            country=country,
            domain=domain,
        )
        if programs:
            self._cache_programs(university_name, programs)
        return programs

    def normalize_degree(
        self,
        degree_name: str,
        country: str = "",
    ) -> dict[str, Any]:
        if self._model is None:
            return self._heuristic_normalize(degree_name)

        prompt = {
            "role": "degree_normalizer",
            "instruction": (
                "Normalize this university degree/program name into a standard format. "
                "Return strict JSON only: {\"normalized_name\": \"...\", \"degree_type\": \"Bachelor/Master/PhD/Diploma/Certificate\", "
                "\"field\": \"...\", \"duration_years\": N, \"country_variant\": \"...\"}. "
                "Do not add markdown."
            ),
            "degree_name": degree_name,
            "country": country or "unknown",
        }
        try:
            resp = self._model.generate_content(json.dumps(prompt, ensure_ascii=False))
            raw = getattr(resp, "text", "") or ""
            return self._extract_json(raw) or self._heuristic_normalize(degree_name)
        except Exception as exc:
            print(f"[UniversityCatalog] LLM normalization error: {exc}")
            return self._heuristic_normalize(degree_name)

    def _enrich_university(self, hipo_data: dict) -> dict[str, Any]:
        name = hipo_data.get("name", "Unknown University")
        domain_list = hipo_data.get("domains", [])
        domain = domain_list[0] if domain_list else ""
        return {
            "id": self._university_id(name, hipo_data.get("country", "")),
            "name": name,
            "country": hipo_data.get("country", ""),
            "alpha_two_code": hipo_data.get("alpha_two_code", ""),
            "domains": domain_list,
            "web_pages": hipo_data.get("web_pages", []),
            "domain": domain,
            "logo_url": f"https://logo.clearbit.com/{domain}" if domain else "",
            "state_province": hipo_data.get("state-province") or "",
        }

    def _university_id(self, name: str, country: str) -> str:
        seed = f"{name.strip().lower()}|{country.strip().lower()}"
        return f"uni_{hashlib.sha1(seed.encode()).hexdigest()[:12]}"

    def _search_local_cache(self, query: str, country: str) -> list[dict]:
        try:
            cache_ref = self._db.collection("university_cache").limit(500)
            if country:
                cache_ref = cache_ref.where("country", "==", country)
            docs = list(cache_ref.stream())
            q = query.lower().strip()
            results = []
            for doc in docs:
                data = doc.to_dict() or {}
                if not q or q in data.get("name", "").lower():
                    results.append(data)
            return results
        except Exception as exc:
            print(f"[UniversityCatalog] Cache search error: {exc}")
            return []

    def _cache_results(self, universities: list[dict]) -> None:
        try:
            batch = self._db.batch()
            cache_ref = self._db.collection("university_cache")
            for uni in universities:
                doc_ref = cache_ref.document(uni["id"])
                batch.set(doc_ref, {**uni, "_cached_at": __import__("datetime").datetime.now(
                    __import__("datetime").timezone.utc
                ).isoformat()}, merge=True)
            batch.commit()
        except Exception as exc:
            print(f"[UniversityCatalog] Cache write error: {exc}")

    def _get_cached_programs(self, university_name: str) -> list[dict]:
        try:
            doc_id = f"prog_{hashlib.sha1(university_name.lower().encode()).hexdigest()[:12]}"
            doc = self._db.collection("university_programs_cache").document(doc_id).get()
            if doc.exists:
                data = doc.to_dict() or {}
                return data.get("programs", [])
        except Exception:
            pass
        return []

    def _cache_programs(self, university_name: str, programs: list[dict]) -> None:
        try:
            doc_id = f"prog_{hashlib.sha1(university_name.lower().encode()).hexdigest()[:12]}"
            self._db.collection("university_programs_cache").document(doc_id).set({
                "university_name": university_name,
                "programs": programs,
                "_cached_at": __import__("datetime").datetime.now(
                    __import__("datetime").timezone.utc
                ).isoformat(),
            })
        except Exception as exc:
            print(f"[UniversityCatalog] Programs cache write error: {exc}")

    def _generate_programs_with_llm(
        self,
        university_name: str,
        country: str,
        domain: str,
    ) -> list[dict]:
        if self._model is None:
            return self._default_programs(university_name)

        prompt = {
            "role": "university_program_generator",
            "instruction": (
                f"You are an admissions expert for {university_name} in {country}. "
                "Generate realistic undergraduate degree programs this university likely offers. "
                "For each program include: name, degree_type (Bachelor/Master), field, duration_years, "
                "core_modules (list of 3-6 subjects), grade_requirements by board (IGCSE, A-Level, IB, CBSE, etc.), "
                "and minimum_threshold (e.g. 'AAA', '85%', '38/45'). "
                "Return strict JSON only: {\"programs\": [...]}. Do not add markdown."
            ),
            "university": university_name,
            "country": country,
        }
        try:
            resp = self._model.generate_content(json.dumps(prompt, ensure_ascii=False))
            raw = getattr(resp, "text", "") or ""
            payload = self._extract_json(raw)
            if payload and isinstance(payload.get("programs"), list):
                return payload["programs"]
        except Exception as exc:
            print(f"[UniversityCatalog] LLM program generation error: {exc}")
        return self._default_programs(university_name)

    def _default_programs(self, university_name: str) -> list[dict]:
        return [
            {
                "id": f"prog_{hashlib.sha1(f'{university_name}cs'.encode()).hexdigest()[:8]}",
                "name": "Computer Science",
                "degree_type": "Bachelor",
                "field": "Computer Science",
                "duration_years": 4,
                "core_modules": [
                    "Algorithms & Data Structures",
                    "Programming Fundamentals",
                    "Mathematics for Computing",
                    "Computer Systems",
                    "Software Engineering",
                ],
                "grade_requirements": {
                    "IB": "36/45",
                    "A-Level": "AAB",
                    "IGCSE": "5 A's including Math & English",
                    "CBSE": "85% in PCM",
                },
                "minimum_threshold": "BBB",
            },
            {
                "id": f"prog_{hashlib.sha1(f'{university_name}eng'.encode()).hexdigest()[:8]}",
                "name": "Engineering (General)",
                "degree_type": "Bachelor",
                "field": "Engineering",
                "duration_years": 4,
                "core_modules": [
                    "Mathematics",
                    "Physics",
                    "Engineering Fundamentals",
                    "Design & Drawing",
                    "Thermodynamics",
                ],
                "grade_requirements": {
                    "IB": "34/45",
                    "A-Level": "ABB",
                    "IGCSE": "5 A's including Math, Physics & English",
                    "CBSE": "80% in PCM",
                },
                "minimum_threshold": "BBC",
            },
            {
                "id": f"prog_{hashlib.sha1(f'{university_name}bus'.encode()).hexdigest()[:8]}",
                "name": "Business Administration",
                "degree_type": "Bachelor",
                "field": "Business",
                "duration_years": 3,
                "core_modules": [
                    "Microeconomics",
                    "Macroeconomics",
                    "Accounting",
                    "Marketing",
                    "Organizational Behavior",
                ],
                "grade_requirements": {
                    "IB": "32/45",
                    "A-Level": "ABB",
                    "IGCSE": "5 A's including Math & English",
                    "CBSE": "80%",
                },
                "minimum_threshold": "BBC",
            },
        ]

    def _heuristic_normalize(self, degree_name: str) -> dict[str, Any]:
        name_lower = degree_name.lower().strip()
        degree_type = "Bachelor"
        if any(kw in name_lower for kw in ["master", "msc", "ma ", "meng", "llm", "mba"]):
            degree_type = "Master"
        elif any(kw in name_lower for kw in ["phd", "doctorate", "dphil"]):
            degree_type = "PhD"
        elif any(kw in name_lower for kw in ["diploma", "certificate"]):
            degree_type = "Diploma"

        duration = 4
        if degree_type == "Master":
            duration = 2
        elif degree_type == "PhD":
            duration = 5
        elif degree_type == "Diploma":
            duration = 1

        field_keywords = {
            "computer": "Computer Science",
            "engineer": "Engineering",
            "business": "Business",
            "economic": "Economics",
            "mathematics": "Mathematics",
            "physics": "Physics",
            "chemistry": "Chemistry",
            "biology": "Biology",
            "psychology": "Psychology",
            "law": "Law",
            "medicine": "Medicine",
            "architecture": "Architecture",
            "art": "Arts",
            "history": "History",
            "literature": "Literature",
            "philosophy": "Philosophy",
        }
        field = "General Studies"
        for kw, f in field_keywords.items():
            if kw in name_lower:
                field = f
                break

        return {
            "normalized_name": degree_name.strip(),
            "degree_type": degree_type,
            "field": field,
            "duration_years": duration,
            "country_variant": "",
        }

    def _extract_json(self, raw: str) -> dict | None:
        text = raw.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:].strip()
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
