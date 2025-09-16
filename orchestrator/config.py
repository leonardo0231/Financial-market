# orchestrator/config.py
from __future__ import annotations
import os
from dataclasses import dataclass

TRUE_SET = {"1", "true", "yes", "on"}

@dataclass(frozen=True)
class OrchestratorConfig:
    enabled: bool = os.getenv("ENABLE_ORCHESTRATOR", "1").lower() in TRUE_SET
    interval_seconds: int = int(os.getenv("ORCH_INTERVAL_SECONDS", "300"))  # default: 5 minutes
    symbol: str = os.getenv("ORCH_SYMBOL", "XAUUSD")
    timeframe: str = os.getenv("ORCH_TIMEFRAME", "M5")
    bars: int = int(os.getenv("ORCH_BARS", "100"))
    # risk defaults
    risk_pct: float = float(os.getenv("ORCH_RISK_PCT", "1.0"))  # 1% per trade
    sl_pips_min: float = float(os.getenv("ORCH_MIN_SL_PIPS", "50"))
    strength_threshold: float = float(os.getenv("ORCH_MIN_SIGNAL_STRENGTH", "0.6"))
