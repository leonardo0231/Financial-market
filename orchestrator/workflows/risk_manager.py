import datetime as dt
from typing import Dict, Any
from fastapi import APIRouter, Request, Depends
from ..settings import settings
from ..utils.http import get_json
from ..security import verify_signature_or_400
from .trade_executor import handle_execute_trade_internal

router = APIRouter()

def _extract_positions(system_status: Dict[str, Any]) -> int:
    if "active_positions" in system_status:
        return int(system_status["active_positions"] or 0)
    comp = system_status.get("components", {})
    if "active_positions" in comp:
        return int(comp["active_positions"] or 0)
    return 0

def _risk_analysis(signal_payload: Dict[str, Any], system_status: Dict[str, Any]) -> Dict[str, Any]:
    # Reproduce the JS code node logic in Python
    signal = signal_payload.get("signal_data", {})
    strength = float(signal.get("strength", 0.0) or 0.0)
    emergency_stop_active = bool(signal.get("emergency_stop_active", False))
    current_positions = _extract_positions(system_status)

    checks = {
        "mt5_connected": bool(system_status.get("mt5_connected", False)),
        "positions_ok": current_positions < settings.MAX_POSITIONS,
        "strength_ok": strength > settings.SIGNAL_THRESHOLD
    }
    approved = all(checks.values()) and not emergency_stop_active

    risk_score = max(0.0, min(1.0, (strength / 1.0) * (1.0 - (current_positions / max(1, settings.MAX_POSITIONS)))))
    reason = "approved" if approved else "rejected_due_to_checks_or_emergency"

    return {
        "signal_id": signal_payload.get("signal_id"),
        "approved": approved,
        "risk_score": risk_score,
        "risk_checks": checks,
        "signal_data": signal,
        "current_positions": current_positions,
        "emergency_stop_active": emergency_stop_active,
        "decision_time": dt.datetime.utcnow().isoformat(),
        "reason": reason
    }

@router.post("/webhook/risk-check")
async def risk_check(request: Request, _=Depends(verify_signature_or_400)):
    data = await request.json()
    return await handle_risk_check_internal(data)

async def handle_risk_check_internal(signal_payload: Dict[str, Any]):
    # Check bot status (health) like 'Check Bot Status' node
    system_status = await get_json(f"{settings.TRADING_BOT_URL}/api/health")
    # Conditions like 'System Ready?' IF node are embedded in analysis:
    risk_result = _risk_analysis(signal_payload, system_status)

    if risk_result["approved"]:
        # forward to executor
        return await handle_execute_trade_internal(risk_result)
    else:
        # log rejection
        return risk_result