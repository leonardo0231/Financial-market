from __future__ import annotations
import os
import requests
from flask import Blueprint, jsonify, request, current_app

bp = Blueprint("orch_proxy", __name__)

def _orch_base() -> str:
    host = os.getenv("ORCH_HOST", "127.0.0.1")
    port = int(os.getenv("ORCH_PORT", "5678"))
    return f"http://{host}:{port}"

@bp.get("/health")
def orch_health():
    try:
        r = requests.get(_orch_base() + "/health", timeout=5)
        return jsonify(r.json()), r.status_code
    except Exception as e:
        current_app.logger.error("Orchestrator health proxy error: %s", e)
        return jsonify({"success": False, "error": str(e)}), 502

@bp.post("/risk-check")
def orch_risk_check():
    payload = request.get_json(silent=True) or {}
    try:
        r = requests.post(_orch_base() + "/webhook/risk-check", json=payload, timeout=10)
        return jsonify(r.json()), r.status_code
    except Exception as e:
        current_app.logger.error("Orchestrator risk-check proxy error: %s", e)
        return jsonify({"success": False, "error": str(e)}), 502

@bp.post("/execute-trade")
def orch_execute_trade():
    payload = request.get_json(silent=True) or {}
    try:
        r = requests.post(_orch_base() + "/webhook/execute-trade", json=payload, timeout=15)
        return jsonify(r.json()), r.status_code
    except Exception as e:
        current_app.logger.error("Orchestrator execute-trade proxy error: %s", e)
        return jsonify({"success": False, "error": str(e)}), 502
