# orchestrator/risk_manager.py
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class RiskInput:
    equity: float
    risk_pct: float           # e.g., 1.0 means 1%
    sl_pips: float            # stop-loss distance in pips
    pip_value_per_lot: float  # value of 1 pip for 1.00 lot (account currency)

@dataclass
class RiskDecision:
    ok: bool
    reason: str | None
    lot_size: float | None
    max_loss: float | None

def evaluate_risk(ri: RiskInput) -> RiskDecision:
    if ri.sl_pips <= 0:
        return RiskDecision(False, "Invalid SL distance (pips<=0)", None, None)
    if ri.risk_pct <= 0:
        return RiskDecision(False, "Invalid risk percentage", None, None)

    max_loss = ri.equity * (ri.risk_pct / 100.0)
    # lot_size = max_loss / (sl_pips * pip_value_per_lot)
    denom = ri.sl_pips * ri.pip_value_per_lot
    if denom <= 0:
        return RiskDecision(False, "Invalid pip value or SL distance", None, None)

    lot_size = max_loss / denom

    # Basic sanity rules
    if lot_size <= 0:
        return RiskDecision(False, "Computed lot size <= 0", None, max_loss)
    if lot_size > 50:  # arbitrary hard cap
        return RiskDecision(False, "Lot size too large", None, max_loss)

    return RiskDecision(True, None, round(lot_size, 2), round(max_loss, 2))
