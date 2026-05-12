from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup


class UniversityProgramCrawler:
    """Multi-source crawler for real university degree programs.

    Data sources (in priority order):
    1. College Scorecard API (US .edu domains) — official US govt data
    2. Web scraping — parses course catalog pages
    3. Country-based templates — realistic defaults by country
    """

    COLLEGE_SCORECARD_API = "https://api.data.gov/ed/collegescorecard/v1/schools"
    _scorecard_api_key = os.environ.get("COLLEGE_SCORECARD_API_KEY", "")

    # Common course catalog URL paths to try when scraping
    CATALOG_PATHS = [
        "/academics/programs",
        "/academics/programs/",
        "/academics/courses",
        "/academics/courses/",
        "/programs",
        "/programs/",
        "/programs-study",
        "/programs-study/",
        "/study",
        "/study/",
        "/study/courses",
        "/study/courses/",
        "/degrees",
        "/degrees/",
        "/degrees-programs",
        "/degrees-programs/",
        "/academics/degrees",
        "/academics/degrees/",
        "/undergraduate",
        "/undergraduate/",
        "/undergraduate/courses",
        "/undergraduate/courses/",
        "/courses",
        "/courses/",
        "/course-list",
        "/course-list/",
        "/prospective-students/courses",
        "/prospective-students/courses/",
        "/future-students/courses",
        "/future-students/courses/",
        "/admissions/courses",
        "/admissions/courses/",
        "/academic-program",
        "/academic-program/",
        "/teaching/courses",
        "/teaching/courses/",
    ]

    def __init__(self, db):
        self._db = db

    # ── Main entry point ──────────────────────────────────────────────

    def get_programs(
        self,
        university_name: str,
        country: str = "",
        domain: str = "",
        force_refresh: bool = False,
    ) -> list[dict[str, Any]]:
        if not force_refresh:
            cached = self._get_cached(university_name)
            if cached:
                return cached

        programs: list[dict] = []

        # Tier 1: College Scorecard (US schools)
        us_programs = self._try_college_scorecard(university_name, domain)
        if us_programs:
            programs = us_programs

        # Tier 2: Web scraping
        if not programs and domain:
            scraped = self._try_web_scrape(domain, university_name)
            if scraped:
                programs = scraped

        # Tier 3: Country-based templates
        if not programs:
            programs = self._country_programs(university_name, country)

        if programs:
            self._cache(university_name, programs)
        return programs

    # ── Tier 1: College Scorecard API ─────────────────────────────────

    def _try_college_scorecard(
        self,
        university_name: str,
        domain: str,
    ) -> list[dict] | None:
        if not self._scorecard_api_key:
            return None
        if domain and not domain.endswith(".edu"):
            return None

        try:
            params = {
                "api_key": self._scorecard_api_key,
                "fields": "id,school.name,school.city,latest.academics.program_percentage",
                "per_page": 1,
            }
            if domain:
                base = domain.split(".")[0]
                params["school.name"] = base.replace("-", " ")
            elif university_name:
                params["school.name"] = university_name[:50]

            resp = requests.get(
                self.COLLEGE_SCORECARD_API,
                params=params,
                timeout=15,
            )
            if resp.status_code != 200:
                print(f"[Crawler] Scorecard API error {resp.status_code}")
                return None

            data = resp.json()
            results = data.get("results") or data.get("data") or []
            if not results:
                return None

            school = results[0]
            prog_pct = school.get("latest", {}).get("academics", {}).get("program_percentage", {}) or {}
            programs = []
            seen = set()
            for cip_code, pct in prog_pct.items():
                if not isinstance(pct, (int, float)) or pct < 0.01:
                    continue
                name = self._cip_to_name(cip_code)
                if not name or name in seen:
                    continue
                seen.add(name)
                programs.append({
                    "id": f"csc_{cip_code}",
                    "name": name,
                    "degree_type": "Bachelor",
                    "field": name,
                    "duration_years": 4,
                    "core_modules": [],
                    "grade_requirements": self._us_grade_requirements(),
                    "minimum_threshold": "3.0 GPA / 1200 SAT",
                    "source": "college_scorecard",
                })

            if programs:
                print(f"[Crawler] Scorecard: found {len(programs)} programs for {university_name}")
                return programs[:20]

        except Exception as exc:
            print(f"[Crawler] Scorecard error: {exc}")

        return None

    # ── Tier 2: Web scraping ──────────────────────────────────────────

    def _try_web_scrape(self, domain: str, university_name: str) -> list[dict] | None:
        if not domain:
            return None

        protocol = "https://"
        base_url = f"{protocol}{domain}"

        for path in self.CATALOG_PATHS:
            url = urljoin(base_url, path)
            print(f"[Crawler] Trying: {url}")
            try:
                resp = requests.get(url, timeout=8, headers=self._headers())
                if resp.status_code != 200:
                    continue

                programs = self._parse_catalog_page(resp.text, url, university_name)
                if programs:
                    print(f"[Crawler] Scrape OK: {len(programs)} programs from {url}")
                    return programs

            except requests.RequestException:
                continue
            except Exception as exc:
                print(f"[Crawler] Scrape error {url}: {exc}")
                continue

        return None

    def _parse_catalog_page(
        self,
        html: str,
        url: str,
        university_name: str,
    ) -> list[dict]:
        soup = BeautifulSoup(html, "html.parser")
        programs = []
        seen = set()

        # Strategy 1: Look for <a> tags with course/program keywords in href/text
        for a in soup.find_all("a", href=True):
            text = a.get_text(strip=True)
            href = a["href"]
            if not text or len(text) < 5:
                continue

            # Filter: must look like a degree program name
            if not self._is_program_name(text):
                continue

            key = text.lower().strip()
            if key in seen:
                continue
            seen.add(key)

            duration = self._guess_duration(text)
            degree_type = self._detect_degree_type(text)
            programs.append({
                "id": f"web_{hashlib.sha1(key.encode()).hexdigest()[:8]}",
                "name": text,
                "degree_type": degree_type,
                "field": self._classify_field(text),
                "duration_years": duration,
                "core_modules": [],
                "grade_requirements": {},
                "minimum_threshold": "",
                "source": "web_scrape",
                "url": urljoin(url, href),
            })

        # Strategy 2: Look for heading + list patterns (common in course catalogs)
        if len(programs) < 3:
            for heading in soup.find_all(["h2", "h3", "h4"]):
                heading_text = heading.get_text(strip=True)
                if not self._is_program_name(heading_text):
                    continue
                key = heading_text.lower().strip()
                if key in seen:
                    continue
                seen.add(key)

                programs.append({
                    "id": f"web_{hashlib.sha1(key.encode()).hexdigest()[:8]}",
                    "name": heading_text,
                    "degree_type": self._detect_degree_type(heading_text),
                    "field": self._classify_field(heading_text),
                    "duration_years": self._guess_duration(heading_text),
                    "core_modules": [],
                    "grade_requirements": {},
                    "minimum_threshold": "",
                    "source": "web_scrape",
                })

        return programs[:30]

    def _is_program_name(self, text: str) -> bool:
        """Heuristic: does this text look like an academic program name?"""
        text_lower = text.lower().strip()

        # Too short
        if len(text_lower) < 6:
            return False

        # Contains typical degree keywords
        degree_kw = [
            "bachelor", "master", "phd", "doctorate", "b.sc", "b.a.", "m.sc",
            "m.a.", "b.eng", "m.eng", "ll.b", "ll.m", "b.com", "m.com",
            "b.b.a", "m.b.a", "b.f.a", "m.f.a", "b.s.", "b.a", "m.s.",
            "diploma", "certificate", "foundation", "undergraduate",
            "postgraduate", "graduate", "honours", "hons",
        ]
        if any(kw in text_lower for kw in degree_kw):
            return True

        # Contains common program subject words with length check
        subject_words = [
            "computer", "engineer", "business", "economics", "mathematics",
            "physics", "chemistry", "biology", "psychology", "law",
            "medicine", "nursing", "pharmacy", "accounting", "finance",
            "marketing", "architecture", "education", "history", "arts",
            "design", "music", "literature", "philosophy", "sociology",
            "political", "international", "environmental", "biotechnology",
            "data science", "artificial intelligence", "mechanical",
            "electrical", "civil", "chemical", "biomedical",
        ]
        if len(text_lower) > 10 and any(kw in text_lower for kw in subject_words):
            return True

        return False

    def _detect_degree_type(self, text: str) -> str:
        t = text.lower()
        if any(kw in t for kw in ["phd", "doctorate", "doctoral", "dphil"]):
            return "PhD"
        if any(kw in t for kw in ["master", "msc", "m.sc", "m.a.", "meng",
                                    "m.eng", "llm", "ll.m", "mba", "m.b.a",
                                    "ma ", "postgraduate", "graduate"]):
            return "Master"
        if any(kw in t for kw in ["diploma", "certificate", "foundation"]):
            return "Diploma"
        if any(kw in t for kw in ["bachelor", "b.sc", "b.s.", "b.a.", "beng",
                                    "b.eng", "llb", "ll.b", "b.com", "b.b.a",
                                    "b.f.a", "undergraduate", "honours", "hons",
                                    "ba ", "bsc"]):
            return "Bachelor"
        return "Bachelor"

    def _classify_field(self, text: str) -> str:
        t = text.lower()
        fields = [
            ("computer", "Computer Science"),
            ("software", "Computer Science"),
            ("data science", "Data Science"),
            ("artificial intelligence", "AI & Machine Learning"),
            ("machine learning", "AI & Machine Learning"),
            ("engineer", "Engineering"),
            ("mechanical", "Mechanical Engineering"),
            ("electrical", "Electrical Engineering"),
            ("civil", "Civil Engineering"),
            ("chemical", "Chemical Engineering"),
            ("biomedical", "Biomedical Engineering"),
            ("business", "Business"),
            ("finance", "Finance"),
            ("accounting", "Accounting"),
            ("marketing", "Marketing"),
            ("economics", "Economics"),
            ("mathematics", "Mathematics"),
            ("physics", "Physics"),
            ("chemistry", "Chemistry"),
            ("biology", "Biology"),
            ("biotechnology", "Biotechnology"),
            ("psychology", "Psychology"),
            ("law", "Law"),
            ("medicine", "Medicine"),
            ("nursing", "Nursing"),
            ("pharmacy", "Pharmacy"),
            ("architecture", "Architecture"),
            ("education", "Education"),
            ("history", "History"),
            ("arts", "Arts"),
            ("design", "Design"),
            ("music", "Music"),
            ("literature", "Literature"),
            ("philosophy", "Philosophy"),
            ("sociology", "Sociology"),
            ("political science", "Political Science"),
            ("international relations", "International Relations"),
            ("environmental", "Environmental Science"),
            ("nursing", "Nursing"),
            ("public health", "Public Health"),
        ]
        for kw, field in fields:
            if kw in t:
                return field
        return "General Studies"

    def _guess_duration(self, text: str) -> int:
        t = text.lower()
        if self._detect_degree_type(text) == "Bachelor":
            if any(kw in t for kw in ["b.sc", "b.s.", "beng", "b.eng"]):
                return 4
            return 3
        if self._detect_degree_type(text) == "Master":
            return 1
        if self._detect_degree_type(text) == "PhD":
            return 3
        return 1

    # ── Tier 3: Country templates ────────────────────────────────────

    def _country_programs(self, name: str, country: str) -> list[dict]:
        c = country.lower().strip()
        if "india" in c:
            return self._india_programs(name)
        if "united kingdom" in c or "uk" in c:
            return self._uk_programs(name)
        if "united states" in c or "usa" in c or "america" in c:
            return self._us_programs(name)
        if "singapore" in c:
            return self._sg_programs(name)
        if "australia" in c:
            return self._au_programs(name)
        if "canada" in c:
            return self._ca_programs(name)
        if "germany" in c:
            return self._de_programs(name)
        if "france" in c:
            return self._fr_programs(name)
        return self._generic_programs(name)

    def _us_programs(self, name: str) -> list[dict]:
        return [
            self._prog(name, "Computer Science B.Sc", "Bachelor", "Computer Science", 4,
                       {"SAT": "1350+", "GPA": "3.5+", "TOEFL": "90+"}),
            self._prog(name, "Business Administration B.B.A", "Bachelor", "Business", 4,
                       {"SAT": "1200+", "GPA": "3.0+", "TOEFL": "85+"}),
            self._prog(name, "Mechanical Engineering B.Sc", "Bachelor", "Mechanical Engineering", 4,
                       {"SAT": "1300+", "GPA": "3.3+", "TOEFL": "90+"}),
            self._prog(name, "Economics B.A", "Bachelor", "Economics", 4,
                       {"SAT": "1250+", "GPA": "3.2+"}),
            self._prog(name, "Psychology B.A", "Bachelor", "Psychology", 4,
                       {"SAT": "1150+", "GPA": "3.0+"}),
            self._prog(name, "Data Science M.Sc", "Master", "Data Science", 2,
                       {"GRE": "320+", "TOEFL": "95+"}),
            self._prog(name, "Computer Science M.Sc", "Master", "Computer Science", 2,
                       {"GRE": "325+", "TOEFL": "95+"}),
        ]

    def _uk_programs(self, name: str) -> list[dict]:
        return [
            self._prog(name, "Computer Science B.Sc (Hons)", "Bachelor", "Computer Science", 3,
                       {"A-Level": "AAA", "IB": "36/45", "IELTS": "7.0"}),
            self._prog(name, "Engineering M.Eng", "Bachelor", "Engineering", 4,
                       {"A-Level": "A*AA", "IB": "38/45", "IELTS": "7.0"}),
            self._prog(name, "Economics B.Sc (Hons)", "Bachelor", "Economics", 3,
                       {"A-Level": "AAA", "IB": "36/45"}),
            self._prog(name, "Law LL.B (Hons)", "Bachelor", "Law", 3,
                       {"A-Level": "AAA", "IB": "36/45", "LNAT": "Competitive"}),
            self._prog(name, "Mathematics B.Sc (Hons)", "Bachelor", "Mathematics", 3,
                       {"A-Level": "AAA", "IB": "36/45"}),
            self._prog(name, "Business & Management B.Sc", "Bachelor", "Business", 3,
                       {"A-Level": "AAB", "IB": "34/45"}),
        ]

    def _india_programs(self, name: str) -> list[dict]:
        return [
            self._prog(name, "B.Tech Computer Science", "Bachelor", "Computer Science", 4,
                       {"JEE": "Advanced Rank", "CBSE": "75%+", "State Board": "75%+"}),
            self._prog(name, "B.Tech Mechanical Engineering", "Bachelor", "Mechanical Engineering", 4,
                       {"JEE": "Advanced Rank", "CBSE": "75%+", "State Board": "75%+"}),
            self._prog(name, "B.Sc Physics (Hons)", "Bachelor", "Physics", 3,
                       {"CBSE": "70%+", "State Board": "70%+"}),
            self._prog(name, "B.Com (Hons)", "Bachelor", "Commerce", 3,
                       {"CBSE": "75%+", "State Board": "70%+"}),
            self._prog(name, "B.A Economics (Hons)", "Bachelor", "Economics", 3,
                       {"CBSE": "75%+", "State Board": "70%+"}),
            self._prog(name, "M.Tech Computer Science", "Master", "Computer Science", 2,
                       {"GATE": "Qualified", "B.Tech": "60%+"}),
            self._prog(name, "MBA", "Master", "Business Administration", 2,
                       {"CAT": "95+ percentile", "Work Exp": "2+ years"}),
        ]

    def _sg_programs(self, name: str) -> list[dict]:
        return [
            self._prog(name, "B.Comp Computer Science", "Bachelor", "Computer Science", 4,
                       {"A-Level": "AAA/AAB", "IB": "38/45", "Poly": "3.6+ GPA"}),
            self._prog(name, "B.Eng Electrical Engineering", "Bachelor", "Electrical Engineering", 4,
                       {"A-Level": "AAB", "IB": "36/45"}),
            self._prog(name, "B.Sc Business Administration", "Bachelor", "Business", 3,
                       {"A-Level": "AAB/ABB", "IB": "36/45"}),
            self._prog(name, "B.Acc Accountancy", "Bachelor", "Accountancy", 3,
                       {"A-Level": "AAA", "IB": "38/45"}),
            self._prog(name, "B.Sc Economics", "Bachelor", "Economics", 3,
                       {"A-Level": "AAB", "IB": "36/45"}),
            self._prog(name, "B.Sc Data Science & Analytics", "Bachelor", "Data Science", 4,
                       {"A-Level": "AAB", "IB": "36/45"}),
            self._prog(name, "B.A Environmental Studies", "Bachelor", "Environmental Science", 4,
                       {"A-Level": "ABB", "IB": "34/45"}),
        ]

    def _au_programs(self, name: str) -> list[dict]:
        return [
            self._prog(name, "Bachelor of Computer Science", "Bachelor", "Computer Science", 3,
                       {"ATAR": "85+", "IELTS": "6.5"}),
            self._prog(name, "Bachelor of Engineering (Honours)", "Bachelor", "Engineering", 4,
                       {"ATAR": "85+", "IELTS": "6.5"}),
            self._prog(name, "Bachelor of Commerce", "Bachelor", "Commerce", 3,
                       {"ATAR": "80+", "IELTS": "6.5"}),
            self._prog(name, "Master of Data Science", "Master", "Data Science", 2,
                       {"Bachelor": "65%+", "IELTS": "6.5"}),
        ]

    def _ca_programs(self, name: str) -> list[dict]:
        return [
            self._prog(name, "Bachelor of Computer Science", "Bachelor", "Computer Science", 4,
                       {"Ontario": "75%+", "IELTS": "6.5"}),
            self._prog(name, "Bachelor of Engineering", "Bachelor", "Engineering", 4,
                       {"Ontario": "80%+", "IELTS": "6.5"}),
            self._prog(name, "Bachelor of Commerce", "Bachelor", "Business", 4,
                       {"Ontario": "75%+", "IELTS": "6.5"}),
            self._prog(name, "Bachelor of Science - Biology", "Bachelor", "Biology", 4,
                       {"Ontario": "75%+", "IELTS": "6.5"}),
            self._prog(name, "Master of Business Administration", "Master", "Business", 2,
                       {"GMAT": "600+", "IELTS": "7.0"}),
        ]

    def _de_programs(self, name: str) -> list[dict]:
        return [
            self._prog(name, "B.Sc Informatik (Computer Science)", "Bachelor", "Computer Science", 3,
                       {"Abitur": "1.5+", "TestDaF": "4+"}),
            self._prog(name, "B.Sc Maschinenbau (Mechanical Engineering)", "Bachelor", "Mechanical Engineering", 3,
                       {"Abitur": "1.5+", "TestDaF": "4+"}),
            self._prog(name, "B.Sc Wirtschaftswissenschaften (Economics)", "Bachelor", "Economics", 3,
                       {"Abitur": "2.0+"}),
            self._prog(name, "M.Sc Data Science", "Master", "Data Science", 2,
                       {"Bachelor": "2.0+", "IELTS": "6.5"}),
        ]

    def _fr_programs(self, name: str) -> list[dict]:
        return [
            self._prog(name, "Licence Informatique (Computer Science)", "Bachelor", "Computer Science", 3,
                       {"Bac": "14/20+", "TCF": "B2"}),
            self._prog(name, "Licence Économie (Economics)", "Bachelor", "Economics", 3,
                       {"Bac": "12/20+"}),
            self._prog(name, "Master Data Science", "Master", "Data Science", 2,
                       {"Licence": "14/20+", "IELTS": "6.5"}),
            self._prog(name, "Diplôme d'Ingénieur", "Bachelor", "Engineering", 5,
                       {"Classes Prépa": "Admissible", "Concours": "Rank"}),
        ]

    def _generic_programs(self, name: str) -> list[dict]:
        return [
            self._prog(name, "Computer Science", "Bachelor", "Computer Science", 4,
                       {"IB": "34/45", "A-Level": "AAB"}),
            self._prog(name, "Business Administration", "Bachelor", "Business", 3,
                       {"IB": "32/45", "A-Level": "ABB"}),
            self._prog(name, "Engineering (General)", "Bachelor", "Engineering", 4,
                       {"IB": "33/45", "A-Level": "AAB"}),
            self._prog(name, "Economics", "Bachelor", "Economics", 3,
                       {"IB": "32/45", "A-Level": "ABB"}),
            self._prog(name, "Mathematics", "Bachelor", "Mathematics", 3,
                       {"IB": "33/45", "A-Level": "AAB"}),
            self._prog(name, "Data Science", "Master", "Data Science", 2,
                       {"Bachelor": "70%+", "IELTS": "6.5"}),
        ]

    def _prog(
        self,
        uni: str,
        name: str,
        dtype: str,
        field: str,
        years: int,
        requirements: dict[str, str],
    ) -> dict[str, Any]:
        return {
            "id": f"tmpl_{hashlib.sha1(f'{uni}|{name}'.encode()).hexdigest()[:8]}",
            "name": name,
            "degree_type": dtype,
            "field": field,
            "duration_years": years,
            "core_modules": [],
            "grade_requirements": requirements,
            "minimum_threshold": next(iter(requirements.values()), ""),
            "source": "country_template",
        }

    # ── Helpers ──────────────────────────────────────────────────────

    def _cip_to_name(self, cip_code: str) -> str:
        """Map CIP codes or Scorecard category names to program names."""
        # Scorecard category name mapping (string keys)
        category_map = {
            "agriculture": "Agriculture",
            "architecture": "Architecture",
            "area_ethnic_cultural_gender": "Area & Cultural Studies",
            "biological": "Biology",
            "business": "Business",
            "communication": "Communications",
            "communications_technology": "Communications Technology",
            "computer": "Computer Science",
            "construction": "Construction Trades",
            "education": "Education",
            "engineering": "Engineering",
            "engineering_technology": "Engineering Technology",
            "english": "English",
            "ethnic_cultural_gender": "Cultural & Gender Studies",
            "family_consumer_science": "Family & Consumer Science",
            "foreign_language": "Foreign Languages",
            "health": "Health Sciences",
            "history": "History",
            "language": "Languages",
            "legal": "Legal Studies",
            "liberal_arts": "Liberal Arts",
            "library": "Library Science",
            "mathematics": "Mathematics",
            "mechanics": "Mechanics & Repair",
            "military": "Military Science",
            "multidiscipline": "Interdisciplinary Studies",
            "parks_recreation_fitness": "Sports & Fitness",
            "personal_services": "Personal Services",
            "philosophy_religious": "Philosophy & Religious Studies",
            "physical_science": "Physical Sciences",
            "precision_production": "Precision Production",
            "psychology": "Psychology",
            "public_administration_social_service": "Public Administration",
            "science_technology": "Science & Technology",
            "security_law_enforcement": "Security & Law Enforcement",
            "social_science": "Social Sciences",
            "theology_religious_vocation": "Theology",
            "transportation": "Transportation",
            "visual_performing_arts": "Visual & Performing Arts",
        }
        if cip_code in category_map:
            return category_map[cip_code]

        # Numeric CIP code mapping (original)
        mapping = {
            "11": "Computer Science",
            "11.01": "Computer Science",
            "11.02": "Computer Programming",
            "11.04": "Information Science",
            "11.07": "Computer Networking",
            "11.08": "Computer Software",
            "11.09": "Computer Systems",
            "11.10": "Computer/Information Tech",
            "14": "Engineering",
            "14.01": "Engineering General",
            "14.07": "Chemical Engineering",
            "14.08": "Civil Engineering",
            "14.09": "Computer Engineering",
            "14.10": "Electrical Engineering",
            "14.11": "Mechanical Engineering",
            "15": "Engineering Technology",
            "24": "Liberal Arts",
            "26": "Biology",
            "27": "Mathematics",
            "40": "Physical Sciences",
            "40.05": "Chemistry",
            "40.08": "Physics",
            "42": "Psychology",
            "45": "Social Sciences",
            "45.02": "Anthropology",
            "45.06": "Economics",
            "45.10": "Political Science",
            "45.11": "Sociology",
            "50": "Visual Arts",
            "50.09": "Music",
            "51": "Health Sciences",
            "51.38": "Nursing",
            "52": "Business",
            "52.02": "Business Administration",
            "52.03": "Accounting",
            "52.06": "Finance",
            "52.14": "Marketing",
            "54": "History",
        }
        # Try exact match first, then prefix match
        if cip_code in mapping:
            return mapping[cip_code]
        for prefix, name in sorted(mapping.items(), key=lambda x: -len(x[0])):
            if cip_code.startswith(prefix):
                return name
        return ""

    def _us_grade_requirements(self) -> dict[str, str]:
        return {
            "SAT": "1200-1450",
            "ACT": "26-32",
            "GPA": "3.3+",
            "TOEFL": "85+",
        }

    def _headers(self) -> dict[str, str]:
        return {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "en-US,en;q=0.9",
        }

    # ── Firestore cache ──────────────────────────────────────────────

    def _cache_key(self, university_name: str) -> str:
        seed = university_name.strip().lower()
        return f"prog_{hashlib.sha1(seed.encode()).hexdigest()[:12]}"

    def _get_cached(self, university_name: str) -> list[dict] | None:
        try:
            doc = self._db.collection("university_programs_cache").document(
                self._cache_key(university_name)
            ).get()
            if doc.exists:
                data = doc.to_dict() or {}
                ts = data.get("_cached_at", "")
                # Cache valid for 30 days
                if ts:
                    cached_time = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    age = datetime.now(timezone.utc) - cached_time
                    if age.days < 30:
                        return data.get("programs", [])
        except Exception:
            pass
        return None

    def _cache(self, university_name: str, programs: list[dict]) -> None:
        try:
            self._db.collection("university_programs_cache").document(
                self._cache_key(university_name)
            ).set({
                "university_name": university_name,
                "programs": programs,
                "_cached_at": datetime.now(timezone.utc).isoformat(),
            })
        except Exception as exc:
            print(f"[Crawler] Cache write error: {exc}")
