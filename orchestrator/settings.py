from __future__ import annotations

from typing import Optional, List
from pydantic import AnyHttpUrl, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # ---------- URLs ----------
    TRADING_BOT_URL: str = "http://127.0.0.1:5000"
    ORCHESTRATOR_PUBLIC_URL: Optional[AnyHttpUrl] = None  # e.g., http://localhost:5678

    # ---------- Security / Auth ----------
    WEBHOOK_SECRET: Optional[str] = None
    ALLOWED_ORIGINS: List[str] = Field(default_factory=lambda: ["*"])
    TRADING_API_KEY: Optional[str] = None  # if your trading-bot expects API key

    # ---------- Risk / Strategy ----------
    SIGNAL_THRESHOLD: float = 0.7
    MAX_POSITIONS: int = 3

    # ---------- Telegram ----------
    TELEGRAM_TOKEN: Optional[str] = None
    TELEGRAM_CHAT_IDS: List[str] = Field(default_factory=list)
    TELEGRAM_POLLING_ENABLED: bool = True

    # ---------- Scheduler ----------
    MARKET_ANALYSIS_CRON_MINUTES: int = 5
    SYMBOL: str = "XAUUSD"
    TIMEFRAME: str = "M5"
    BARS: int = 100

    # ---------- Pydantic Settings Config ----------
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )
    # ---- Normalizers ----
    @field_validator("ALLOWED_ORIGINS", mode="before")
    @classmethod
    def _parse_origins(cls, v):
        if not v:
            return ["*"]
        if isinstance (v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        return v
    
    @field_validator("TELEGRAM_CHAT_IDS", mode="before")
    @classmethod
    def _parse_chat_ids(cls, v):
        if not v:
            return []
        if isinstance(v, str):
            return [s.strip() for s in v.strip(",") if s.strip()]
        return []
    
    @property
    def ORCH_BASE(self) -> Optional[str]:
        if self.ORCHESTRATOR_PUBLIC_URL:
            return str(self.ORCHESTRATOR_PUBLIC_URL)
        return None
    

settings = Settings()