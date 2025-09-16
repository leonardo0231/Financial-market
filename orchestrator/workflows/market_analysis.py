import time, json, asyncio, datetime as dt
from fastapi import APIRouter
from ..settings import settings
from ..utils.http import get_json, post_json
from .signal_generator import handle_signal_generated_internal

router = APIRouter()

async def run_market_analysis_job():
    # Step 1: Get Market Data
    ohlc = await get_json(f"{settings.TRADING_BOT_URL}/api/ohlc", params={
        "symbol": settings.SYMBOL,
        "timeframe": settings.TIMEFRAME,
        "bars": settings.BARS
    })
    if not ohlc or not ohlc.get("success", False):
        # Log and exit
        return

    # Step 2: Analyze Market
    analysis_req = {
        "symbol": settings.SYMBOL,
        "strategies": ["all"],
        "timeframe": settings.TIMEFRAME,
        "bars": settings.BARS,
        "enable_ai_analysis": True,
        "enable_multi_symbol": True,
        "enable_news_sentiment": True,
        "correlation_analysis": True,
    }
    analysis = await post_json(f"{settings.TRADING_BOT_URL}/api/analyze", analysis_req)

    # Step 3: Forward to Signal Generator (internal)
    if analysis:
        await handle_signal_generated_internal(analysis)