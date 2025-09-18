from __future__ import annotations

import os
import sys
import re
import json
import time
import signal
import threading
import logging
import requests
import pandas as pd

from logging.handlers import RotatingFileHandler
from functools import wraps
from typing import Optional, Dict, Any, List, Tuple, cast

from flask import Flask, jsonify, request
from flask_cors import CORS
from concurrent.futures import ThreadPoolExecutor

from .core.mt5_connector import MT5Connector
from .data.processor import DataProcessor
from .strategies.strategy_manager import StrategyManager
from .utils.risk_calculator import RiskCalculator
from .database.connection import db_manager


# ---------------------------
# Config loader (settings.json)
# ---------------------------
def _load_settings() -> Dict[str, Any]:
    default_setting = {
        "app": {"host": "0.0.0.0", "port": 5000},
        "logging":{
            "level": "INFO",
            "file": "logs/trading.log",
            "rotation_mb": 10,
            "backup_count": 5,
            "consol_output": True,
            "file_output": True
        },
        "trading": {"magic_number": 234000, "max_position": 3}
    }
    try:
        with open("config/settings.json", "r", encoding="utf-8") as f:
            user_cfg = json.load(f)
            # shallow-merge for essentials
            for k, v in default_setting.items():
                if k not in user_cfg:
                    user_cfg[k] = v
                elif isinstance(v, Dict):
                    for sk, sv in v.items():
                        user_cfg[k].setdefault(sk, sv)
            return user_cfg
    except FileNotFoundError:
        return default_setting
    

SETTING: Dict[str, Any] = _load_settings()


# ---------------------------
# Logging
# ---------------------------
def _parse_size_to_bytes(size_value: Any) -> int:
    """
    Accepts:
      - int (bytes)
      - string like '10MB', '5GB', '512K', '1000000'
      - numeric in settings via 'rotation_mb'
    """
    if isinstance(size_value, (int, float)):
        return int(size_value) if isinstance(size_value, int) else int(float(size_value))

    if not isinstance(size_value, str):
        return 10 * 1024 * 1024  # 10MB default

    s = size_value.strip().upper()
    try:
        if s.endswith("GB"):
            return int(float(s[:-2]) * 1024 * 1024 * 1024)
        if s.endswith("G"):
            return int(float(s[:-1]) * 1024 * 1024 * 1024)
        if s.endswith("MB"):
            return int(float(s[:-2]) * 1024 * 1024)
        if s.endswith("M"):
            return int(float(s[:-1]) * 1024 * 1024)
        if s.endswith("KB"):
            return int(float(s[:-2]) * 1024)
        if s.endswith("K"):
            return int(float(s[:-1]) * 1024)
        return int(float(s))  # plain number in bytes
    except Exception:
        return 10 * 1024 * 1024
    
def setup_logging() -> logging.Logger:
    cfg = SETTING.get("logging", {})
    level_name = cfg.get("level", "INFO").upper()
    log_level = getattr(logging, level_name, logging.INFO)
    log_file = cfg.get("file", "logs/trading.log")
    backup_count = int(cfg.get("backup_count", 5))

    # rotation size: support both legacy 'max_size' and new 'rotation_mb'
    rotation_mb = cfg.get("rotation_mb")
    max_size = cfg.get("max_size")
    if rotation_mb is not None and max_size is None:
        max_bytes = _parse_size_to_bytes(f"{rotation_mb}MB")
    else:
        max_bytes = _parse_size_to_bytes(max_size if max_size is not None else "10MB")
    
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    details_fmt = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s"
    )
    simple_fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    handlers: List[logging.Handler] = []
    if cfg.get("file_output", True):
        fh = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8")
        fh.setLevel(log_level)
        fh.setFormatter(details_fmt)
        handlers.append(fh)
    if cfg.get("consol_output", True):
        ch = logging.StreamHandler()
        ch.setLevel(log_level)
        ch.setFormatter(simple_fmt)
        handlers.append(ch)
    
    logging.basicConfig(level=log_level, handlers=handlers, force=True)

    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)
    logging.getLogger("requests.packages.urllib3").setLevel(logging.WARNING)

    logger = logging.getLogger("xauusd_trading_bot.main")
    logger.info("Logging configured (level=%s, file=%s, rotation=%s bytes)", level_name, log_file, max_bytes)
    
    return logger

LOGGER = setup_logging()

# ---------------------------
# Flask app
# ---------------------------
app = Flask(__name__)
CORS(app)

# --- Index ---
@app.get("/")
def index():
    return jsonify({
        "ok": True,
        "service": "trading-bot",
        "version": "1.0.1",
        "endpoints": [
            {"method": "GET",  "url": "/api/health"},
            {"method": "GET",  "url": "/api/ohlc?symbol=XAUUSD&timeframe=M5&bars=100"},
            {"method": "POST", "url": "/api/analyze"},
            {"method": "POST", "url": "/api/execute_trade"}
        ]
    })

# --- Health ---
@app.get("/api/health")
def api_health():
    return jsonify({
        "status": "ok",
        "service": "trading-bot",
        "version": "1.0.1"
    })

# --- OHLC ---
@app.get("/api/ohlc")
def api_ohlc():
    if BOT is None or not BOT._initialized:
        return jsonify({"success": False, "error": "bot not initialized"}), 503

    symbol = request.args.get("symbol", "XAUUSD")
    timeframe = request.args.get("timeframe", "M5")
    try:
        bars = int(request.args.get("bars", "100"))
    except ValueError:
        return jsonify({"success": False, "error": "invalid bars"}), 400

    try:
        data = BOT.get_market_data(symbol=symbol, timeframe=timeframe, bars=bars)
        return jsonify({
            "success": True,
            "symbol": symbol,
            "timeframe": timeframe,
            "data": data
        })
    except Exception as e:
        LOGGER.error(f"/api/ohlc error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500

# --- Analyze ---
@app.post("/api/analyze")
def api_analyze():
    if BOT is None or not BOT._initialized:
        return jsonify({"success": False, "error": "bot not initialized"}), 503

    payload = request.get_json(silent=True) or {}
    symbol    = payload.get("symbol", "XAUUSD")
    timeframe = payload.get("timeframe", "M5")
    bars      = int(payload.get("bars", 100))
    candles   = payload.get("candles")

    if candles is None:
        candles = BOT.get_market_data(symbol=symbol, timeframe=timeframe)
    df= pd.DataFrame(candles)
    if "time" in df.columns:
        df = df.sort_values("time").reset_index(drop=True)
    if "atr" in df.columns:
        high = df["high"].astype(float); low = df["low"].astype("float"); close = df["close"].astype("float")
        prev_close = close.shift(1)
        tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        df["atr"] = tr.rolling(14, min_periods=1).mean()
    
    strategies = payload.get("strategies", "all")
    if strategies == "all":
        requested = ["all"]
    elif isinstance(strategies, str):
        requested = [s.strip() for s in strategies.split(",") if s.strip()]
    else:
        requested = strategies

    bot_ref = BOT

    if not bot_ref or not getattr(bot_ref, "_initialized", False):
        return jsonify({"success": False, "error": "bot not initialized"}), 503

    if bot_ref.strategy_manager is None:
        return jsonify({"success": False, "error": "strategy_manager is None"}), 500

    sm = cast(StrategyManager, bot_ref.strategy_manager)

    try:
        result = sm.analyze(df=df, requested_strategies=requested)
        return jsonify(result or {"success": False, "error": "no result"})
    except Exception as e:
        LOGGER.error(f"/api/analyze error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500

# --- Execute Trade ---
@app.post("/api/execute_trade")
def api_execute_trade():
    if BOT is None or not BOT._initialized:
        return jsonify({"success": False, "error": "bot not initialized"}), 503

    payload = request.get_json(silent=True) or {}
    required = ["symbol", "signal"]
    missing = [k for k in required if k not in payload]
    if missing:
        return jsonify({"success": False, "error": f"missing: {', '.join(missing)}"}), 400

    symbol       = payload["symbol"]
    signal       = str(payload["signal"]).upper()  # "BUY" / "SELL"
    volume       = float(payload.get("volume", 0.01))
    entry_price  = payload.get("entry_price")
    stop_loss    = payload.get("stop_loss")
    take_profit  = payload.get("take_profit")

    if signal not in ("BUY", "SELL"):
        return jsonify({"success": False, "error": "signal must be BUY or SELL"}), 400

    try:
        res = BOT.mt5_connector.open_trade(
            symbol=symbol,
            direction=signal,
            volume=volume,
            price=entry_price,
            sl=stop_loss,
            tp=take_profit,
            comment="api_execute_trade",
        )

        return jsonify(res)
    except Exception as e:
        LOGGER.error(f"/api/execute_trade error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500

# ---------------------------
# Prometheus
# ---------------------------
def setup_prometheus() -> Optional[Dict[str, Any]]:
    enabled = os.getenv("PROMETHEUS_ENABLED", "0").lower() in ("1", "true", "yes", "on")
    if not enabled:
        return None
    try:
        from prometheus_client import start_http_server, Counter, Histogram, Gauge
        port = int(os.getenv("PROMETHEUS_PORT", "8000"))
        start_http_server(port)
        LOGGER.info("📊 Prometheus metrics server started on :%s", port)

        metrics = {
            "trades_total": Counter("trading_bot_trades_total", "Total trades", ["status", "symbol"]),
            "request_latency": Histogram("trading_bot_request_duration_seconds", "Request latency", ["endpoint"]),
            "active_positions": Gauge("trading_bot_active_positions", "Number of active positions"),
            "port": port,
        }
        return metrics
    except Exception as exc:
        LOGGER.warning("Prometheus init failed or not installed: %s", exc)
        return None
    
PROM_METRICS = setup_prometheus()

# ---------------------------
# MT5 thread-safety
# ---------------------------
_mt5_lock = threading.RLock()

def mt5_threadsafe(fn):
    @wraps(fn)
    def _wrapper(*args, **kwargs):
        with _mt5_lock:
            return fn(*args, **kwargs)
    return _wrapper

# ---------------------------
# TradingBot class
# ---------------------------
class TradingBot:
    def __init__(self) -> None:
        self.running = False
        self.mt5_connector: Optional[MT5Connector] = None
        self.data_processor: Optional[DataProcessor] = None
        self.strategy_manager: Optional[StrategyManager] = None
        self.risk_calculator: Optional[RiskCalculator] = None
        self._initialization_lock = threading.Lock()
        self._initialized = False
        self.database_unavailable = False
    
    def initialize(self) -> bool:
        """Initialize DB, MT5, strategies—synchronously for determinism."""
        with self._initialization_lock:
            if self._initialized:
                return True
            try:
                LOGGER.info("Initializing Trading Bot...")

                # 1- Database
                if not db_manager.initialize():
                    raise RuntimeError("Failed to initialize database")
                if not db_manager.create_tables():
                    raise RuntimeError("Failed to create database tables")

                # 2- MT5
                self.mt5_connector = MT5Connector()
                if not self.mt5_connector.connect():
                    raise RuntimeError("Failed to connect to MT5")
                
                # 3- Components
                self.data_processor = DataProcessor()
                self.strategy_manager = StrategyManager()
                self.risk_calculator = RiskCalculator()
                self.strategy_manager.load_strategies()

                # 4- DB health
                if not db_manager.health_check():
                    LOGGER.error("Database health check failed—attempting recovery...")
                    if not db_manager.initialize():
                        LOGGER.critical("Database recovery failed—running in LIMITED mode (no DB).")
                        self.database_unavailable = True
                    else:
                        self.database_unavailable = False
                        logging.info("Database recovery successful")
                else:
                    self.database_unavailable = False

                self._initialized = True
                if self.database_unavailable:
                    LOGGER.warning("Initialized with LIMITED functionality (no database).")
                else:
                    LOGGER.info("Trading Bot initialized successfully.")
                return True
            
            except Exception as exc:
                LOGGER.error("Initialization failed: %s", exc, exc_info=True)
                return False
            
    def shutdown(self) -> None:
        LOGGER.info("Shutting down Trading Bot...")
        self.running = False
        try:
            if self.mt5_connector:
                self.mt5_connector.disconnect()
            db_manager.close()
            self._initialized = False
            LOGGER.info("Trading Bot shutdown complete")
        except Exception as exc:
            LOGGER.error("Error during shutdown: %s", exc)
            self._initialized = False

    @mt5_threadsafe
    def get_market_data(self, symbol: str = "XAUUSD", timeframe: str = "M5", bars: int = 100):
        if not self._initialized:
            raise RuntimeError("Bot not initialized")
        return self.mt5_connector.get_ohlc_data(symbol, timeframe, bars)

# ---------------------------
# URL announcer
# ---------------------------
def _build_urls() -> Tuple[List[str], List[str], List[str]]:
    hostnames = ["127.0.0.1", "localhost"]
    port = int(SETTING.get("app", {}).get("port", 5000))

    endpoints = [
        ("GET",  "/","Root (if any)"),
        ("GET",  "/api/health","Health"),
        ("GET",  "/api/ohlc?symbol=XAUUSD&timeframe=M5&bars=100","OHLC (sample)"),
        ("POST", "/api/analyze","Analyze"),
        ("POST", "/api/execute_trade","Execute Trade"),
    ]

    bot_triplets = [
        (method, f"http://{h}:{port}{path}", desc) 
        for h in hostnames
        for (method, path, desc) in endpoints
    ]
    
    prom_triplets = []
    if PROM_METRICS and "port" in PROM_METRICS:
        pport = PROM_METRICS["port"]
        prom_triplets = [("GET", f"http://{h}:{pport}/", "Prometheus metrics") for h in hostnames]

    # Orchestrator
    orch_triplets = []
    orch_port = os.getenv("ORCH_PORT")
    orch_host = os.getenv("ORCH_HOST", "127.0.0.1")
    if orch_port:
        orch_candidates = hostnames if  orch_host in hostnames else [orch_host]
        orch_endpoints = [
            ("GET",  "/health",  "Orchestrator Health"),
            ("GET",  "/docs",    "Orchestrator API Docs"),
            ("GET",  "/metrics", "Orchestrator Metrics"),
        ]
        orch_triplets = [
            (method, f"http://{h}:{int(orch_port)}{path}", desc)
            for h in orch_candidates
            for (method, path, desc) in orch_endpoints
        ]

    fmt = lambda trips: [f"{m:<6} {u}    # {d}" for (m, u, d) in trips]

    return fmt(bot_triplets), fmt(prom_triplets), fmt(orch_triplets)

def _wait_until_responsive(url:str, timeout_s:int = 30) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            r = requests.get(url, timeout=2.0)
            if r.status_code < 500:
                return True
        except Exception:
            time.sleep(0.5)
    return False

def start_url_announcer() -> None:
    port = int(SETTING.get("app", {}).get("port", 5000))
    health_url = f"http://127.0.0.1:{port}/api/health"

    def _runner():
        ok = _wait_until_responsive(health_url, timeout_s=40)
        bot_urls, prom_urls, orch_urls = _build_urls()
        
        lines: List[str] = []
        lines.append("✅ Service is up. Useful URLs:")
        lines.append("— Trading Bot:")
        lines.extend([f"   {ln}" for ln in bot_urls])
        if prom_urls:
            lines.append("— Metrics:")
            lines.extend([f"   {ln}" for ln in prom_urls])
        if orch_urls:
            lines.append("— Orchestrator (if running):")
            lines.extend([f"   {ln}" for ln in orch_urls])

        txt = "\n".join(lines)
        print(txt, flush=True)
        LOGGER.info("\n%s", txt)

        if not ok:
            LOGGER.warning("Health check did not respond in time; URLs printed anyway.")

    t = threading.Thread(target=_runner, name="url_announcer", daemon=True)
    t.start()

# ---------------------------
# Global bot instance + signals
# ---------------------------
BOT: Optional[TradingBot] = None

def _signal_handler(sig, frame):
    LOGGER.info("Received signal %s — shutting down...", sig)
    try:
        if BOT:
            BOT.shutdown()
    finally:
        sys.exit(0)


# ---------------------------
# Main
# ---------------------------
def main() -> int:
    global BOT

    # Ensure dir
    os.makedirs("logs", exist_ok=True)
    os.makedirs("config", exist_ok=True)

    # Start bot
    LOGGER.info("Starting XAU/USD Trading Bot...")
    BOT = TradingBot()
    if not BOT.initialize():
        LOGGER.info("Starting XAU/USD Trading Bot...")
        return 1
    
    # Start log housekeeping if you have it
    try:
        from .utils.log_manager import setup_log_cleanup_scheduler
        setup_log_cleanup_scheduler()
        LOGGER.info("Log cleanup scheduler started.")
    except Exception as exc:
        LOGGER.debug("No log cleanup scheduler or failed to start: %s", exc)

    # Announce URLs after service becomes responsive
    start_url_announcer()

    # Run Flask
    app.config["PROPAGATE_EXCEPTIONS"] = None
    host = SETTING.get("app", {}).get("host", "0.0.0.0")
    port = int(SETTING.get("app", {}).get("port", 5000))
    LOGGER.info("Starting Flask server on http://%s:%s", host, port)

    # Mark running then serve
    BOT.running = True
    try:
        app.run(host=host, port=port, debug=False, threaded=True, use_reloader=False)
        return 0
    except KeyboardInterrupt:
        LOGGER.info("Interrupted by user.")
        return 0
    except Exception as exc:
        LOGGER.error("Unexpected error in main: %s", exc, exc_info=True)
        return 1
    finally:
        if BOT:
            BOT.shutdown()

if __name__ == "__main__":
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    sys.exit(main())
