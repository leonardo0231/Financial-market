from __future__ import annotations
from apscheduler.schedulers.background import BackgroundScheduler
from flask import current_app
from .config import OrchestratorConfig
from .workflow import run_market_analysis_job

_scheduler: BackgroundScheduler | None = None

def start_scheduler(app) -> None:
    global _scheduler
    cfg = OrchestratorConfig()
    if not cfg.enabled:
        app.logger.info("Python Orchestrator disabled via env.")
        return

    if _scheduler is not None and _scheduler.running:
        app.logger.info("Orchestrator scheduler already running.")
        return

    _scheduler = BackgroundScheduler(timezone="UTC")
    _scheduler.add_job(
        run_market_analysis_job,
        "interval",
        seconds=cfg.interval_seconds,
        id="market_analysis_job",
        replace_existing=True,
        kwargs={"cfg": cfg},
        max_instances=1,
        coalesce=True,
        misfire_grace_time=60,
    )
    _scheduler.start()
    app.logger.info("Python Orchestrator scheduler started (interval=%ss)", cfg.interval_seconds)

def get_scheduler():
    """Get the current scheduler instance."""
    return _scheduler