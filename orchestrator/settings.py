from pydantic_settings import BaseSettings
from pydantic import AnyHttpUrl, Field
from typing import Optional, List

class Settings(BaseSettings):
    # URLs
    TRADING_BOT_URL: AnyHttpUrl = "http://trading-bot:5000"
    ORCHESTRATOR_PUBLIC_URL: Optional[AnyHttpUrl] = None  # e.g., http://localhost:5678

    # Security / Auth
    WEBHOOK_SECRET: str = "change_me"
    ALLOWED_ORIGINS: str = "*"
    TRADING_API_KEY: Optional[str] = None  # if your trading-bot expects API key

    # Risk / Strategy
    SIGNAL_THRESHOLD: float = 0.7
    MAX_POSITIONS: int = 3

    # Telegram
    TELEGRAM_TOKEN: Optional[str] = None
    TELEGRAM_CHAT_IDS: Optional[str] = None  # comma separated

    TELEGRAM_POLLING_ENABLED: bool = False

    # Scheduler
    MARKET_ANALYSIS_CRON_MINUTES: int = 5
    SYMBOL: str = "XAUUSD"
    TIMEFRAME: str = "M5"
    BARS: int = 100

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()