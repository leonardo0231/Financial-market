from __future__ import annotations
from flask import Blueprint, request, jsonify
from .config import OrchestratorConfig
from .workflow import _analyze_market, run_market_analysis_job
from .risk_manager import RiskInput, evaluate_risk
from .trade_executor import TradeRequest, execute_trade

orchestrator_bp = Blueprint("orchestrator", __name__)

@orchestrator_bp.route("/signal-generated", methods=["POST"])
def signal_generated():
    payload = request.get_json(force=True, silent=True) or {}
    return jsonify({"received": True, "payload": payload})

@orchestrator_bp.route("/risk-check", methods=["POST"])
def risk_check():
    j = request.get_json(force=True, silent=True) or {}
    cfg = OrchestratorConfig()
    equity = float(j.get("equity", 0.0))
    sl_pips = float(j.get("sl_pips", cfg.sl_pips_min))
    pip_val = float(j.get("pip_value_per_lot", 1.0))
    ri = RiskInput(equity=equity, risk_pct=cfg.risk_pct, sl_pips=sl_pips, pip_value_per_lot=pip_val)
    decision = evaluate_risk(ri)
    return jsonify({"success": decision.ok, "lot_size": decision.lot_size, "max_loss": decision.max_loss, "reason": decision.reason})

@orchestrator_bp.route("/execute-trade", methods=["POST"])
def execute_trade_ep():
    j = request.get_json(force=True, silent=True) or {}
    req = TradeRequest(
        symbol=j["symbol"],
        direction=j["direction"],
        volume=float(j["volume"]),
        entry=j.get("entry"),
        stop_loss=j.get("stop_loss"),
        take_profit=j.get("take_profit"),
        comment=j.get("comment", "orchestrator"),
    )
    result = execute_trade(req)
    return jsonify(result)

@orchestrator_bp.route("/run-now", methods=["POST"])
def run_now():
    cfg = OrchestratorConfig()
    run_market_analysis_job(cfg)
    return jsonify({"success": True})
