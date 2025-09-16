import datetime as dt
from typing import Dict, Any
from fastapi import APIRouter, Request, Depends
from ..settings import settings
from ..utils.http import post_json
from ..security import verify_signature_or_400
from .telegram_controller import notify_telegram_internal

router = APIRouter()

def _build_trade_request(signal_envelope: Dict[str, Any]) -> Dict[str, Any]:
    sd = signal_envelope.get("signal_data", {})
    position_size = (
        sd.get("risk_parameters", {}).get("position_size")
        if isinstance(sd.get("risk_parameters"), dict) else None
    )
    return {
        "symbol": sd.get("symbol", settings.SYMBOL),
        "signal": sd.get("signal"),
        "entry_price": sd.get("entry_price"),
        "stop_loss": sd.get("stop_loss"),
        "take_profit": sd.get("take_profit"),
        "volume": float(position_size) if position_size is not None else 0.01,
    }

@router.post("/webhook/execute-trade")
async def execute_trade(request: Request, _=Depends(verify_signature_or_400)):
    data = await request.json()
    return await handle_execute_trade_internal(data)

async def handle_execute_trade_internal(risk_result: Dict[str, Any]):
    trade_request = _build_trade_request(risk_result)
    execution_id = f"{risk_result.get('signal_id','')}-exec".strip("-")
    try:
        resp = await post_json(f"{settings.TRADING_BOT_URL}/api/execute_trade", trade_request)
        success = bool(resp.get("success", False))
        if success:
            out = {
                "trade_result": {
                    "execution_id": execution_id,
                    "ticket": resp.get("ticket"),
                    "symbol": resp.get("symbol", trade_request["symbol"]),
                    "volume": resp.get("volume"),
                    "price": resp.get("price"),
                    "success": True,
                    "executed_at": resp.get("execution_time"),
                    "message": f"Trade executed successfully: Ticket {resp.get('ticket')}"
                }
            }
        else:
            out = {
                "trade_result": {
                    "execution_id": execution_id,
                    "symbol": trade_request["symbol"],
                    "success": False,
                    "error": resp.get("error"),
                    "retcode": resp.get("retcode"),
                    "failed_at": dt.datetime.utcnow().isoformat(),
                    "message": f"Trade failed: {resp.get('error')}"
                }
            }
    except Exception as exc:
        out = {
            "trade_result": {
                "execution_id": execution_id,
                "symbol": trade_request["symbol"],
                "success": False,
                "error": str(exc),
                "failed_at": dt.datetime.utcnow().isoformat(),
                "message": f"Trade failed: {exc}"
            }
        }

    # forward to telegram notify
    await notify_telegram_internal(out)
    return out