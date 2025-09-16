from __future__ import annotations
from flask import Blueprint, jsonify

bp = Blueprint("root", __name__)

@bp.get("/")
def index():
    return jsonify({
        "ok": True,
        "service": "trading-bot",
        "version": "1.0.1",
        "endpoints": [
            {"method": "GET",  "url": "/api/health"},
            {"method": "GET",  "url": "/api/ohlc?symbol=XAUUSD&timeframe=M5&bars=100"},
            {"method": "POST", "url": "/api/analyze"},
            {"method": "POST", "url": "/api/execute_trade"},
            # orchestrator proxy
            {"method": "GET",  "url": "/api/orch/health"},
            {"method": "POST", "url": "/api/orch/risk-check"},
            {"method": "POST", "url": "/api/orch/execute-trade"}
        ]
    })