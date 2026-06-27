from __future__ import annotations

import json
import os
from typing import Any

import firebase_admin
from fastapi import Request
from firebase_admin import auth, credentials


def initialize_firebase() -> None:
    if firebase_admin._apps:
        return

    try:
        service_account_json = os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON", "")
        cred_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
        if service_account_json:
            cred = credentials.Certificate(json.loads(service_account_json))
            firebase_admin.initialize_app(cred)
        elif cred_path and os.path.exists(cred_path):
            cred = credentials.Certificate(cred_path)
            firebase_admin.initialize_app(cred)
        else:
            # Auto-discover firebase-admin.json in project directory
            local_paths = [
                os.path.join(os.getcwd(), "firebase-admin.json"),
                os.path.join(os.path.dirname(__file__), "..", "firebase-admin.json"),
            ]
            found = None
            for p in local_paths:
                normalized = os.path.normpath(p)
                if os.path.exists(normalized):
                    found = normalized
                    break
            if found:
                cred = credentials.Certificate(found)
                firebase_admin.initialize_app(cred)
                print(f"Firebase Admin initialized from {found}")
            else:
                firebase_admin.initialize_app()
                print("Firebase Admin initialized with default credentials")
        print("Firebase Admin initialized successfully")
    except Exception as e:
        print(f"Firebase Admin initialization skipped: {e}")


async def current_user(request: Request) -> dict[str, Any]:
    """Optional auth - returns placeholder if no token, doesn't block requests"""
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return {"uid": "anonymous", "email": "anonymous@example.com"}

    token = auth_header.split(" ", 1)[1].strip()
    if not token:
        return {"uid": "anonymous", "email": "anonymous@example.com"}

    try:
        decoded_token = auth.verify_id_token(token)
        if "uid" not in decoded_token and "sub" in decoded_token:
            decoded_token["uid"] = decoded_token["sub"]
        return decoded_token
    except Exception:
        return {"uid": "anonymous", "email": "anonymous@example.com"}


async def optional_current_user(request: Request) -> dict[str, Any] | None:
    """Optional auth - returns None if no valid token, otherwise returns user dict"""
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return None

    token = auth_header.split(" ", 1)[1].strip()
    if not token:
        return None

    try:
        decoded_token = auth.verify_id_token(token)
        if "uid" not in decoded_token and "sub" in decoded_token:
            decoded_token["uid"] = decoded_token["sub"]
        return decoded_token
    except Exception:
        return None
