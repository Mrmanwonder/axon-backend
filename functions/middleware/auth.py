from __future__ import annotations

import os
from typing import Any

import firebase_admin
from fastapi import HTTPException, Request
from firebase_admin import auth, credentials


def initialize_firebase() -> None:
    if firebase_admin._apps:
        return

    try:
        cred_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
        if cred_path and os.path.exists(cred_path):
            cred = credentials.Certificate(cred_path)
            firebase_admin.initialize_app(cred)
        else:
            firebase_admin.initialize_app()
        print("Firebase Admin initialized successfully")
    except Exception as e:
        print(f"Firebase Admin initialization skipped: {e}")


async def current_user(request: Request) -> dict[str, Any]:
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Token")

    token = auth_header.split(" ", 1)[1].strip()
    if not token:
        raise HTTPException(status_code=401, detail="Invalid Token")

    try:
        decoded_token = auth.verify_id_token(token)
        if "uid" not in decoded_token and "sub" in decoded_token:
            decoded_token["uid"] = decoded_token["sub"]
        return decoded_token
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid Token")
