import asyncio
import httpx
from ..settings import settings
from .telegram_controller import notify_telegram_internal

_last_update_id = 0

COMMANDS_HELP = (
    "🤖 XAU/USD Trading Bot Active!\n\n"
    "📋 Available Commands:\n"
    "/status - Bot status\n"
    "/positions - Open positions\n"
    "/balance - Account balance\n"
    "/stop - Stop trading\n"
    "/start_trading - Start trading\n"
    "/strategy [name] - Set strategy\n"
    "/risk [percent] - Set risk level"
)

async def poll_and_process_commands():
    global _last_update_id
    if not (settings.TELEGRAM_TOKEN and settings.TELEGRAM_CHAT_IDS):
        return

    url = f"https://api.telegram.org/bot{settings.TELEGRAM_TOKEN}/getUpdates"
    params = {"timeout": 10}
    if _last_update_id:
        params["offset"] = _last_update_id + 1

    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            r = await client.get(url, params=params)
            data = r.json() if r.headers.get("content-type","").startswith("application/json") else {}
        except Exception:
            return

    for update in data.get("result", []):
        _last_update_id = max(_last_update_id, update.get("update_id", 0))
        message = update.get("message") or {}
        chat = message.get("chat",{})
        chat_id = chat.get("id")
        text = (message.get("text") or "").strip().lower()
        if not text:
            continue

        # Simple command router
        if text == "/start":
            await _send(chat_id, COMMANDS_HELP)
        elif text == "/status":
            # delegate to trading-bot /api/health and return summary
            await _send_status(chat_id)
        elif text == "/positions":
            await _send(chat_id, "📈 Checking open positions...")
        elif text == "/balance":
            await _send(chat_id, "💰 Checking account balance...")
        elif text == "/stop":
            await _send(chat_id, "🛑 Stopping trading operations...")
        elif text == "/start_trading":
            await _send(chat_id, "▶️ Starting trading operations...")
        elif text.startswith("/strategy "):
            strategy = text.replace("/strategy ", "")
            await _send(chat_id, f"🎯 Setting strategy to: {strategy}")
        elif text.startswith("/risk "):
            risk = text.replace("/risk ", "")
            await _send(chat_id, f"⚖️ Setting risk level to: {risk}%")
        else:
            # unknown command
            await _send(chat_id, "❓ Unknown command. Send /start")

async def _send(chat_id: int, text: str):
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}
    async with httpx.AsyncClient(timeout=10.0) as client:
        await client.post(f"https://api.telegram.org/bot{settings.TELEGRAM_TOKEN}/sendMessage", json=payload)

async def _send_status(chat_id: int):
    import httpx
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(f"{settings.TRADING_BOT_URL}/api/health")
            ok = r.json().get("ok") if r.headers.get("content-type","").startswith("application/json") else False
    except Exception:
        ok = False
    text = "🟢 System is operational" if ok else "🔴 System has issues"
    await _send(chat_id, text)