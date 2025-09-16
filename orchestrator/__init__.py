from __future__ import annotations
from flask import Flask
from .scheduler import start_scheduler
from .api import orchestrator_bp
from .telegram_controller import TelegramController

def init_orchestrator(app: Flask) -> None:
    """
    Wire-up Python Orchestrator into the existing Flask app.
    Safe to call multiple times (idempotent).
    """
    # 1) Register Blueprint (replaces n8n webhooks)
    app.register_blueprint(orchestrator_bp, url_prefix="/orchestrator")

    # 2) Start background scheduler jobs
    start_scheduler(app)

    # 3) Start Telegram controller (optional; can be disabled via env)
    try:
        tc = TelegramController.from_env()
        if tc is not None:
            tc.start_background_polling(app.logger)
    except Exception as exc:
        app.logger.exception("Telegram controller failed to start: %s", exc)
