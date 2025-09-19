from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List
import os

class Settings(BaseSettings):
    PORT: int = 8000
    BASE_TZ: str = "UTC"
    DATABASE_URL: str = "sqlite:///./news.db"
    TELEGRAM_BOT_TOKEN: str
    INITIAL_CHAT_IDS: str | None = None
    INVESTING_RSS_URLS: str = "https://www.investing.com/rss/news.rss"
    FF_CALENDAR_PAGE: str = "https://www.forexfactory.com/calendar?week=this"
    REFRESH_INTERVAL_MIN: int = 15
    UPCOMING_DEFAULT_HOURS: int = 8

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)

    def initial_chat_ids_list(self) -> List[int]:
        if not self.INITIAL_CHAT_IDS:
            return []
        return [int(x.strip()) for x in self.INITIAL_CHAT_IDS.split(",") if x.strip()]
    
settings = Settings