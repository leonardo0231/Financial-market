from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from .settings import settings
from .workflows.market_analysis import run_market_analysis_job
from .workflows.telegram_commands import poll_and_process_commands

scheduler = AsyncIOScheduler()

def start_scheduler():
    scheduler.add_job(
        run_market_analysis_job,
        CronTrigger(minute=f"*/{settings.MARKET_ANALYSIS_CRON_MINUTES}"),
        id="market_analysis_job",
        replace_existing=True
    )
    # Optional: Telegram command polling
    if settings.TELEGRAM_POLLING_ENABLED:
        scheduler.add_job(poll_and_process_commands, 'interval', minutes=1, id='telegram_poll', replace_existing=True)
    scheduler.start()