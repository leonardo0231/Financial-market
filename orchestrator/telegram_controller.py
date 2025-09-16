from __future__ import annotations
import os
import threading
import time
import requests
from typing import Optional

API_BASE = "https://api.telegram.org"

class TelegramController:
    def __init__(self, token: str, chat_ids: list[str]):
        self.token = token
        self.chat_ids = chat_ids
        self._stop = threading.Event()
        self._offset = 0

    @classmethod
    def from_env(cls) -> Optional["TelegramController"]:
        token = os.getenv("TELEGRAM_TOKEN", "").strip()
        chats = [c.strip() for c in os.getenv("TELEGRAM_CHAT_IDS", "").split(",") if c.strip()]
        if not token or not chats:
            return None
        return cls(token, chats)

    def start_background_polling(self, logger) -> None:
        t = threading.Thread(target=self._poll_loop, args=(logger,), daemon=True)
        t.start()

    def _poll_loop(self, logger) -> None:
        while not self._stop.is_set():
            try:
                updates = self._get_updates()
                for u in updates:
                    self._offset = max(self._offset, u["update_id"] + 1)
                    msg = u.get("message", {}) or u.get("edited_message", {})
                    text = (msg.get("text") or "").strip()
                    chat_id = msg.get("chat", {}).get("id")
                    if not text or not chat_id:
                        continue
                    self._handle_command(text, chat_id)
            except Exception as exc:
                logger.exception("Telegram polling error: %s", exc)
            time.sleep(1.5)

    def stop(self) -> None:
        self._stop.set()

    # --- helpers
    def _get_updates(self):
        url = f"{API_BASE}/bot{self.token}/getUpdates"
        r = requests.get(url, params={"timeout": 10, "offset": self._offset}, timeout=15)
        r.raise_for_status()
        data = r.json()
        return data.get("result", [])

    def _send_text(self, chat_id: int | str, text: str) -> None:
        url = f"{API_BASE}/bot{self.token}/sendMessage"
        requests.post(url, json={"chat_id": chat_id, "text": text}, timeout=10)

    def _broadcast(self, text: str) -> None:
        for cid in self.chat_ids:
            try:
                self._send_text(cid, text)
            except Exception:
                pass

    # --- commands
    def _handle_command(self, text: str, chat_id: int | str) -> None:
        if text == "/start":
            self._send_text(chat_id, "✅ Bot is online (Python Orchestrator 1.0.1)")
        elif text == "/status":
            self._send_text(chat_id, "🩺 Status: OK\nOrchestrator running.")
        elif text == "/run":
            # دستی اجرای فوری job
            from .config import OrchestratorConfig
            from .workflow import run_market_analysis_job
            run_market_analysis_job(OrchestratorConfig())
            self._send_text(chat_id, "⏱️ Job triggered.")
        else:
            self._send_text(chat_id, "🤖 Unknown command. Try /start /status /run")
