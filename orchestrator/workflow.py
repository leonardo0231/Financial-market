# orchestrator/workflow.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import os
import logging
import requests

from .config import OrchestratorConfig
from .risk_manager import RiskInput, evaluate_risk
from .trade_executor import TradeRequest, execute_trade
from xauusd_trading_bot.strategies.strategy_manager import StrategyManager
from xauusd_trading_bot.utils.telegram_notifier import TelegramNotifier
from xauusd_trading_bot.core.mt5_connector import MT5Connector

Direction = Literal["BUY", "SELL", "NEUTRAL"]

@dataclass
class Signal:
    symbol: str
    signal: Direction
    strength: float
    stop_loss: float | None
    take_profit: float | None
    entry: float | None
    comment: str = ""

def _get_equity() -> float:
    return MT5Connector.get_global().get_account_info().get("equity", 0.0)

def _pip_value_per_lot(symbol: str) -> float:
    # خواندن از MT5 یا جدول داخلی (XAUUSD)
    return MT5Connector.get_global().get_symbol_info(symbol).get("pip_value_per_lot", 1.0)

def _fetch_ohlc_via_api(symbol: str, timeframe: str, bars: int) -> dict:
    base = os.getenv("TRADING_BOT_URL", "http://trading-bot:5000")
    r = requests.get(f"{base}/api/ohlc", params={"symbol": symbol, "timeframe": timeframe, "bars": bars}, timeout=10)
    r.raise_for_status()
    return r.json()

def _analyze_market(symbol: str, timeframe: str, bars: int) -> Signal | None:
    # می‌توانید به‌جای HTTP، به‌طور مستقیم از StrategyManager استفاده کنید. اینجا برای عدم
    # تغییر سایر بخش‌ها و حفظ Contract قبلی، از API داخلی استفاده شده است.
    data = _fetch_ohlc_via_api(symbol, timeframe, bars)
    if not data.get("success"):
        return None
    # StrategyManager طبق کد اصلی:
    sm = StrategyManager()
    res = sm.analyze_market(symbol=symbol, timeframe=timeframe, candles=data["data"])
    # انتظار: خروجی شامل signal/strength/sl/tp/entry
    if not res or res.get("signal", "NEUTRAL") == "NEUTRAL" or float(res.get("strength", 0)) < OrchestratorConfig().strength_threshold:
        return None
    return Signal(
        symbol=symbol,
        signal=res["signal"],
        strength=float(res["strength"]),
        stop_loss=res.get("stop_loss"),
        take_profit=res.get("take_profit"),
        entry=res.get("entry"),
        comment=res.get("comment", ""),
    )

def _notify(text: str) -> None:
    try:
        chat_ids = os.getenv("TELEGRAM_CHAT_IDS", "")
        notifier = TelegramNotifier(os.getenv("TELEGRAM_TOKEN", ""), [c.strip() for c in chat_ids.split(",") if c.strip()])
        notifier.send_message(text)
    except Exception:
        logging.getLogger(__name__).exception("Failed to send telegram notification")

def run_market_analysis_job(cfg: OrchestratorConfig) -> None:
    """
    Replacement for n8n 01_Market_Analysis + 02_Signal_Generator + 03_Risk_Manager + 04_Trade_Executor + 05_Telegram
    """
    sig = _analyze_market(cfg.symbol, cfg.timeframe, cfg.bars)
    if sig is None:
        return

    equity = _get_equity()
    pip_val = _pip_value_per_lot(cfg.symbol)

    # محاسبه ریسک
    if sig.stop_loss and sig.entry:
        sl_pips = abs(sig.entry - sig.stop_loss) * 10.0  # مثال: تبدیل به پیپ (بسته به نماد)
    else:
        sl_pips = cfg.sl_pips_min

    decision = evaluate_risk(RiskInput(equity=equity, risk_pct=cfg.risk_pct, sl_pips=sl_pips, pip_value_per_lot=pip_val))
    if not decision.ok or decision.lot_size is None:
        _notify(f"🔴 Risk rejected: {decision.reason or 'unknown'}\n\nSymbol: {cfg.symbol}\nSignal: {sig.signal}\nStrength: {sig.strength}")
        return

    tr = TradeRequest(
        symbol=cfg.symbol,
        direction="BUY" if sig.signal == "BUY" else "SELL",
        volume=decision.lot_size,
        entry=sig.entry,
        stop_loss=sig.stop_loss,
        take_profit=sig.take_profit,
        comment=f"sig:{sig.signal}|str:{sig.strength}",
    )
    res = execute_trade(tr)

    if res.get("success"):
        _notify(
            "🟢 Trade Executed\n"
            f"📊 {tr.symbol}\n"
            f"🎯 {tr.direction}  vol={tr.volume}\n"
            f"💰 entry={res.get('price')}  SL={tr.stop_loss}  TP={tr.take_profit}\n"
            f"🎫 ticket={res.get('ticket')}"
        )
    else:
        _notify(
            "🔴 Trade Failed\n"
            f"📊 {tr.symbol}  {tr.direction}  vol={tr.volume}\n"
            f"⚠️ error={res.get('error')}  code={res.get('retcode')}"
        )
