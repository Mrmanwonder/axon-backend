#!/usr/bin/env python3
"""
Seed universities from the Hipo API into Firestore university_cache.
Usage:
    python seed_universities.py                          # All countries
    python seed_universities.py --countries India,USA    # Specific countries
    python seed_universities.py --dry-run                # Show counts only
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone

import requests

# Countries with the most universities in the Hipo API
DEFAULT_COUNTRIES = [
    "United States",
    "United Kingdom",
    "Canada",
    "Australia",
    "India",
    "China",
    "Japan",
    "South Korea",
    "Singapore",
    "Germany",
    "France",
    "Italy",
    "Spain",
    "Netherlands",
    "Switzerland",
    "Sweden",
    "Denmark",
    "Norway",
    "Finland",
    "Brazil",
    "Mexico",
    "South Africa",
    "Nigeria",
    "Kenya",
    "United Arab Emirates",
    "Saudi Arabia",
    "Qatar",
    "Malaysia",
    "New Zealand",
    "Ireland",
    "Belgium",
    "Austria",
    "Poland",
    "Russia",
    "Turkey",
    "Israel",
    "Thailand",
    "Vietnam",
    "Philippines",
    "Indonesia",
    "Pakistan",
    "Bangladesh",
    "Argentina",
    "Chile",
    "Colombia",
    "Egypt",
    "Ghana",
]


def _university_id(name: str, country: str) -> str:
    import hashlib

    seed = f"{name.strip().lower()}|{country.strip().lower()}"
    return f"uni_{hashlib.sha1(seed.encode()).hexdigest()[:12]}"


def fetch_country(country: str) -> list[dict]:
    url = f"http://universities.hipolabs.com/search?country={requests.utils.quote(country)}"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        print(f"  Error fetching {country}: {exc}")
        return []


def main():
    parser = argparse.ArgumentParser(description="Seed universities into Firestore")
    parser.add_argument("--countries", help="Comma-separated list of countries")
    parser.add_argument("--dry-run", action="store_true", help="Only print counts")
    args = parser.parse_args()

    countries = args.countries.split(",") if args.countries else DEFAULT_COUNTRIES
    countries = [c.strip() for c in countries if c.strip()]

    total = 0
    all_unis: list[dict] = []

    print(f"Fetching universities for {len(countries)} countries...")
    for i, country in enumerate(countries, 1):
        print(f"[{i}/{len(countries)}] {country}...", end=" ", flush=True)
        unis = fetch_country(country)
        enriched = []
        for u in unis:
            domains = u.get("domains", [])
            domain = domains[0] if domains else ""
            enriched.append(
                {
                    "id": _university_id(u.get("name", ""), country),
                    "name": u.get("name", "Unknown"),
                    "country": country,
                    "alpha_two_code": u.get("alpha_two_code", ""),
                    "domains": domains,
                    "web_pages": u.get("web_pages", []),
                    "domain": domain,
                    "logo_url": f"https://logo.clearbit.com/{domain}" if domain else "",
                    "state_province": u.get("state-province") or "",
                }
            )
        print(f"{len(enriched)} universities")
        all_unis.extend(enriched)
        total += len(enriched)
        if i < len(countries):
            time.sleep(0.5)  # Rate limiting

    print(f"\nTotal: {total} universities across {len(countries)} countries")

    if args.dry_run:
        return

    # Write to Firestore
    try:
        import firebase_admin
        from firebase_admin import credentials, firestore

        cred_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
        if cred_path and os.path.exists(cred_path):
            cred = credentials.Certificate(cred_path)
            firebase_admin.initialize_app(cred)
        else:
            firebase_admin.initialize_app()

        db = firestore.client(
            database_id=os.environ.get("FIRESTORE_DATABASE_ID", "axon")
        )

        batch = db.batch()
        cache_ref = db.collection("university_cache")
        ops = 0
        for uni in all_unis:
            doc_ref = cache_ref.document(uni["id"])
            batch.set(
                doc_ref,
                {
                    **uni,
                    "_cached_at": datetime.now(timezone.utc).isoformat(),
                },
                merge=True,
            )
            ops += 1
            if ops >= 400:
                batch.commit()
                print(f"  Committed {ops} documents...")
                batch = db.batch()
                ops = 0

        if ops > 0:
            batch.commit()
            print(f"  Committed final {ops} documents")

        print(f"\nDone! Seeded {total} universities into Firestore university_cache")
    except Exception as e:
        print(f"\nError writing to Firestore: {e}")
        print("Results printed above for manual inspection.")
        sys.exit(1)


if __name__ == "__main__":
    main()
