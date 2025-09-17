from pydantic_settings import BaseSettings
from pydantic import AnyHttpUrl, Field
from typing import Optional, List
import dotenv
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

dotenv.load_dotenv()

class Settings(BaseSettings):
    # URLs
    TRADING_BOT_URL = os.getenv("TRADING_BOT_URL")
    ORCHESTRATOR_PUBLIC_URL: Optional[AnyHttpUrl] = None  # e.g., http://localhost:5678

    # Security / Auth
    WEBHOOK_SECRET: str = str(os.getenv("WEBHOOK_SECRET"))
    ALLOWED_ORIGINS: str = "*"
    TRADING_API_KEY: Optional[str] = None  # if your trading-bot expects API key

    # Risk / Strategy
    SIGNAL_THRESHOLD = os.getenv("SIGNAL_THRESHOLD") or 0.7
    MAX_POSITIONS = os.getenv("MAX_POSITIONS") or 3

    # Telegram
    TELEGRAM_TOKEN: Optional[str] = os.getenv("TELEGRAM_TOKEN")
    TELEGRAM_CHAT_IDS: Optional[str] = os.getenv("TELEGRAM_CHAT_IDS")

    TELEGRAM_POLLING_ENABLED = os.getenv("TELEGRAM_ENABLED") or True

    # Scheduler
    MARKET_ANALYSIS_CRON_MINUTES = os.getenv("MARKET_ANALYSIS_CRON_MINUTES") or 5
    SYMBOL: str = "XAUUSD"
    TIMEFRAME: str = str(os.getenv("TIMEFRAME"))
    BARS: int = 100

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()