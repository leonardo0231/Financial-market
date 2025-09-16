import datetime as dt, uuid, json
from fastapi import APIRouter, Request, Depends
from ..settings import settings
from ..security import verify_signature_or_400
from .risk_manager import handle_risk_check_internal

router = APIRouter()

def is_valid_signal(payload: dict) -> bool:
    if not payload: 
        return False
    if str(payload.get("signal","")).upper() == "NEUTRAL":
        return False
    try:
        strength = float(payload.get("strength", 0.0))
    except Exception:
        strength = 0.0
    return strength > float(settings.SIGNAL_THRESHOLD)

@router.post("/webhook/signal-generated")
async def signal_generated(request: Request, _=Depends(verify_signature_or_400)):
    body = await request.body()
    data = await request.json()
    return await handle_signal_generated_internal(data)

async def handle_signal_generated_internal(data: dict):
    if not is_valid_signal(data):
        return {"ignored": True, "reason": "weak_or_neutral_signal"}

    signal_id = f"{dt.datetime.utcnow().strftime('%Y%m%d-%H%M%S')}-{data.get('signal','')}".strip("-")
    signal_payload = {
        "signal_id": signal_id,
        "symbol": data.get("symbol", settings.SYMBOL),
        "signal_data": data,
        "created_at": dt.datetime.utcnow().isoformat()
    }
    # Forward to Risk Manager (internal)
    return await handle_risk_check_internal(signal_payload)