# orchestrator/trade_executor.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
from xauusd_trading_bot.core.mt5_connector import MT5Connector  # ← موجود در پروژه اصلی

Direction = Literal["BUY", "SELL"]

@dataclass
class TradeRequest:
    symbol: str
    direction: Direction
    volume: float
    entry: float | None
    stop_loss: float | None
    take_profit: float | None
    comment: str = "orchestrator"

def execute_trade(req: TradeRequest) -> dict:
    """
    Execute the trade via existing MT5Connector.
    Returns a dict compatible with previous n8n notify payloads.
    """
    mt5 = MT5Connector.get_global()  # مطابق معماری اصلی (Singleton/Shared)
    result = mt5.open_trade(
        symbol=req.symbol,
        direction=req.direction,
        volume=req.volume,
        price=req.entry,
        sl=req.stop_loss,
        tp=req.take_profit,
        comment=req.comment,
    )
    # شکل‌دهی خروجی مشابه payloadهای قبلی برای حفظ سازگاری
    return {
        "success": bool(result.get("success", False)),
        "symbol": req.symbol,
        "direction": req.direction,
        "volume": req.volume,
        "price": result.get("price"),
        "ticket": result.get("ticket"),
        "retcode": result.get("retcode"),
        "error": result.get("error"),
        "executed_at": result.get("executed_at"),
    }
