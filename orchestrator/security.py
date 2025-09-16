import hmac, hashlib, os
from fastapi import Header, HTTPException

from .settings import settings

def compute_signature(body: bytes) -> str:
    secret = settings.WEBHOOK_SECRET.encode("utf-8")
    sig = hmac.new(secret, body, hashlib.sha256).hexdigest()
    return f"sha256={sig}"

def verify_signature_or_400(body: bytes, x_hub_signature_256: str | None = Header(default=None)):
    expected = compute_signature(body)
    if not x_hub_signature_256 or not hmac.compare_digest(x_hub_signature_256, expected):
        raise HTTPException(status_code=401, detail="Invalid HMAC signature")