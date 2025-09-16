import datetime as dt
from typing import Dict, Any
from fastapi import APIRouter, Request, Depends, HTTPException
from ..settings import settings
from ..security import verify_signature_or_400
import httpx

router = APIRouter()

def _format_message(payload: Dict[str, Any]) -> str:
    tr = payload.get("trade_result", {})
    success = bool(tr.get("success", False)) or bool(payload.get("success", False))
    if success:
        text = (
            "🟢 Trade Executed Successfully!\n\n"
            f"📊 Symbol: {tr.get('symbol') or payload.get('symbol') or 'Unknown'}\n"
            f"🎯 Signal: {payload.get('signal_data',{}).get('signal') or payload.get('signal') or 'Unknown'}\n"
            f"💰 Entry: {tr.get('price') or 'N/A'}\n"
            f"🛡️ Stop Loss: {payload.get('signal_data',{}).get('stop_loss') or payload.get('stop_loss') or 'N/A'}\n"
            f"🎯 Take Profit: {payload.get('signal_data',{}).get('take_profit') or payload.get('take_profit') or 'N/A'}\n"
            f"📈 Volume: {tr.get('volume') or payload.get('volume') or 'N/A'}\n"
            f"🎫 Ticket: {tr.get('ticket') or 'N/A'}\n"
            f"⏰ Time: {tr.get('executed_at') or dt.datetime.utcnow().isoformat()}"
        )
    else:
        text = (
            "🔴 Trade Execution Failed!\n\n"
            f"📊 Symbol: {tr.get('symbol') or payload.get('symbol') or 'Unknown'}\n"
            f"❌ Error: {tr.get('error') or payload.get('error') or 'Unknown error'}\n"
            f"⏰ Time: {tr.get('failed_at') or dt.datetime.utcnow().isoformat()}"
        )
    return text

@router.post("/webhook/telegram-notify")
async def telegram_notify(request: Request, _=Depends(verify_signature_or_400)):
    data = await request.json()
    return await notify_telegram_internal(data)

async def notify_telegram_internal(data: Dict[str, Any]):
    if not settings.TELEGRAM_TOKEN or not settings.TELEGRAM_CHAT_IDS:
        # silently no-op if not configured
        return {"sent": False, "reason": "telegram_not_configured"}
    chat_id = settings.TELEGRAM_CHAT_IDS.split(",")[0].strip()
    text = _format_message(data)

    url = f"https://api.telegram.org/bot{settings.TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}

    async with httpx.AsyncClient(timeout=20.0) as client:
        try:
            r = await client.post(url, json=payload)
            ok = r.status_code == 200 and (r.json().get("ok", False) if r.headers.get("content-type","").startswith("application/json") else True)
            return {"sent": ok}
        except Exception as e:
            return {"sent": False, "error": str(e)}