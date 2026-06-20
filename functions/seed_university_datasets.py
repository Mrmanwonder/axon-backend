#!/usr/bin/env python3
"""
Import AISHE (India) and IPEDS (USA) university datasets into Firestore.
These are supplemental datasets that provide detailed program/degree information
beyond what the Hipo API offers.

Usage:
    python seed_university_datasets.py --aishe data/aishe.csv
    python seed_university_datasets.py --ipeds data/ipeds.csv
    python seed_university_datasets.py --dry-run
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys
import re
from datetime import datetime, timezone


def _doc_id(seed: str) -> str:
    return f"prog_{hashlib.sha1(seed.lower().encode()).hexdigest()[:12]}"


def _detect_degree_type(course: str) -> str:
    cl = course.lower()
    if any(kw in cl for kw in ["phd", "doctorate", "doctoral"]):
        return "PhD"
    if any(kw in cl for kw in ["master", "msc", "ma ", "mba", "meng", "llm", "postgrad"]):
        return "Master"
    if any(kw in cl for kw in ["diploma", "certificate", "pgdip"]):
        return "Diploma"
    return "Bachelor"


def _field_from_course(course: str) -> str:
    cl = course.lower()
    field_map = {
        "computer": "Computer Science",
        "engineer": "Engineering",
        "technol": "Technology",
        "business": "Business",
        "manage": "Management",
        "economic": "Economics",
        "mathematics": "Mathematics",
        "physics": "Physics",
        "chemistry": "Chemistry",
        "biology": "Biology",
        "biotech": "Biotechnology",
        "psychology": "Psychology",
        "law ": "Law",
        "legal": "Law",
        "medicine": "Medicine",
        "pharmacy": "Pharmacy",
        "nursing": "Nursing",
        "architecture": "Architecture",
        "art": "Arts",
        "design": "Design",
        "history": "History",
        "literature": "Literature",
        "philosophy": "Philosophy",
        "education": "Education",
        "accounting": "Accounting",
        "finance": "Finance",
        "marketing": "Marketing",
        "environment": "Environmental Science",
        "agricultur": "Agriculture",
        "dental": "Dentistry",
        "veterinary": "Veterinary Science",
    }
    for kw, field in field_map.items():
        if kw in cl:
            return field
    return "General Studies"


def import_aishe(csv_path: str, dry_run: bool = False) -> list[dict]:
    """Import AISHE CSV (India Higher Education Survey format)."""
    programs = []
    if not os.path.exists(csv_path):
        print(f"AISHE file not found: {csv_path}")
        return programs

    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            university = row.get("University Name", row.get("institute_name", "")).strip()
            course = row.get("Course Name", row.get("program_name", "")).strip()
            level = row.get("Level", row.get("level", "")).strip()
            duration = row.get("Duration", row.get("duration_years", "3")).strip()

            if not university or not course:
                continue

            try:
                duration_years = int(re.sub(r"\D", "", duration) or "3")
            except ValueError:
                duration_years = 3

            degree_type = _detect_degree_type(f"{course} {level}")
            field = _field_from_course(course)

            programs.append({
                "id": _doc_id(f"{university}|{course}"),
                "university_name": university,
                "country": "India",
                "degree_type": degree_type,
                "field": field,
                "course_name": course,
                "duration_years": duration_years,
                "core_modules": [],
                "grade_requirements": {
                    "CBSE": "75%",
                    "ISC": "75%",
                    "State Board": "75%",
                },
                "minimum_threshold": "Pass",
                "source": "AISHE",
            })

            if dry_run and i < 10:
                print(f"  [{degree_type}] {university}: {course} ({duration_years}y)")

    return programs


def import_ipeds(csv_path: str, dry_run: bool = False) -> list[dict]:
    """Import IPEDS CSV (US Integrated Postsecondary Education Data System)."""
    programs = []
    if not os.path.exists(csv_path):
        print(f"IPEDS file not found: {csv_path}")
        return programs

    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            university = row.get("institution_name", row.get("INSTNM", "")).strip()
            course = row.get("cip_title", row.get("cip_title", "")).strip()
            cred_level = row.get("credential_level", "").strip()

            if not university or not course:
                continue

            degree_type = "Bachelor"
            if "master" in cred_level.lower() or "graduate" in cred_level.lower():
                degree_type = "Master"
            elif "doctor" in cred_level.lower():
                degree_type = "PhD"
            elif "associate" in cred_level.lower() or "certificate" in cred_level.lower():
                degree_type = "Diploma"

            field = _field_from_course(course)

            programs.append({
                "id": _doc_id(f"{university}|{course}"),
                "university_name": university,
                "country": "United States",
                "degree_type": degree_type,
                "field": field,
                "course_name": course,
                "duration_years": 4 if degree_type == "Bachelor" else (2 if degree_type in ("Master", "Diploma") else 5),
                "core_modules": [],
                "grade_requirements": {
                    "SAT": "1200",
                    "ACT": "25",
                    "GPA": "3.0",
                },
                "minimum_threshold": "2.5 GPA",
                "source": "IPEDS",
            })

            if dry_run and i < 10:
                print(f"  [{degree_type}] {university}: {course}")

    return programs


def main():
    parser = argparse.ArgumentParser(description="Seed university datasets into Firestore")
    parser.add_argument("--aishe", help="Path to AISHE CSV file")
    parser.add_argument("--ipeds", help="Path to IPEDS CSV file")
    parser.add_argument("--dry-run", action="store_true", help="Only print counts")
    args = parser.parse_args()

    all_programs = []

    if args.aishe:
        print("Importing AISHE (India)...")
        progs = import_aishe(args.aishe, args.dry_run)
        print(f"  Found {len(progs)} programs")
        all_programs.extend(progs)

    if args.ipeds:
        print("Importing IPEDS (USA)...")
        progs = import_ipeds(args.ipeds, args.dry_run)
        print(f"  Found {len(progs)} programs")
        all_programs.extend(progs)

    if not all_programs:
        print("No data imported. Use --aishe path/to/aishe.csv or --ipeds path/to/ipeds.csv")
        return

    print(f"\nTotal: {len(all_programs)} programs")

    if args.dry_run:
        return

    try:
        import firebase_admin
        from firebase_admin import credentials, firestore

        cred_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
        if cred_path and os.path.exists(cred_path):
            cred = credentials.Certificate(cred_path)
            firebase_admin.initialize_app(cred)
        else:
            firebase_admin.initialize_app()

        db = firestore.client(database_id=os.environ.get("FIRESTORE_DATABASE_ID", "axon"))

        batch = db.batch()
        prog_ref = db.collection("university_programs_cache")
        ops = 0
        for prog in all_programs:
            doc_ref = prog_ref.document(prog["id"])
            batch.set(doc_ref, {
                **prog,
                "_cached_at": datetime.now(timezone.utc).isoformat(),
            }, merge=True)
            ops += 1
            if ops >= 400:
                batch.commit()
                print(f"  Committed {ops} programs...")
                batch = db.batch()
                ops = 0

        if ops > 0:
            batch.commit()
            print(f"  Committed final {ops} programs")

        print(f"\nDone! Seeded {len(all_programs)} programs into Firestore university_programs_cache")
    except Exception as e:
        print(f"\nError writing to Firestore: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
