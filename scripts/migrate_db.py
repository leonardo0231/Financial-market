"""
Database Migration Management Script
Handles database schema migrations using Alembic
"""

import os
import sys
import time
import threading
import logging
from datetime import datetime
from flask import Flask
from flask_cors import CORS

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import components
from xauusd_trading_bot.core.mt5_connector import MT5Connector
from xauusd_trading_bot.data.processor import DataProcessor
from xauusd_trading_bot.strategies.strategy_manager import StrategyManager
from xauusd_trading_bot.utils.risk_calculator import RiskCalculator
from xauusd_trading_bot.database.connection import DatabaseManager
from xauusd_trading_bot._version import VERSION

# Setup logging first
from xauusd_trading_bot.utils.log_manager import setup_logging
setup_logging()
logger = logging.getLogger(__name__)

# Import for shutdown handler
import atexit
import signal
import sys

# Flask app
app = Flask(__name__)
CORS(app)

# Global instances (will be initialized by TradingBot)
mt5_connector = None
data_processor = None
strategy_manager = None
risk_calculator = None
bot = None
# Database manager singleton
_db_manager_instance = None

def get_db_manager():
    """Get singleton database manager instance"""
    global _db_manager_instance
    if _db_manager_instance is None:
        _db_manager_instance = DatabaseManager()
    return _db_manager_instance

def cleanup_resources():
    """Clean up resources on shutdown"""
    try:
        logger.info("Starting resource cleanup...")
        
        # Shutdown MT5 connection
        global mt5_connector
        if mt5_connector:
            try:
                import MetaTrader5 as mt5
                mt5.shutdown()
                logger.info("MT5 connection closed successfully")
            except Exception as e:
                logger.error(f"Error closing MT5 connection: {e}")
        
        # Close database connections
        global _db_manager_instance
        if _db_manager_instance:
            try:
                _db_manager_instance.close()
                logger.info("Database connections closed successfully")
            except Exception as e:
                logger.error(f"Error closing database connections: {e}")
        
        logger.info("Resource cleanup completed")
        
    except Exception as e:
        logger.error(f"Error during resource cleanup: {e}")

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    cleanup_resources()
    sys.exit(0)

# Register shutdown handlers
atexit.register(cleanup_resources)
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

# On Windows, also handle SIGBREAK
if sys.platform == "win32":
    signal.signal(signal.SIGBREAK, signal_handler)

class TradingBot:
    """Main trading bot class (simplified for Flask Blueprint architecture)"""
    
    def __init__(self):
        self.running = False
        self.start_time = time.time()
        self._initialized = False
        self._initialization_lock = threading.Lock()
        
    def initialize(self):
        """Initialize all components"""
        global mt5_connector, data_processor, strategy_manager, risk_calculator
        
        with self._initialization_lock:
            if self._initialized:
                return True
                
            try:
                logger.info("Initializing Trading Bot...")
                
                # Initialize database using singleton
                from xauusd_trading_bot.database.connection import db_manager
                if not db_manager.initialize():
                    logger.warning("Database initialization failed, continuing without database")
                
                # Initialize MT5 connection
                mt5_connector = MT5Connector()
                if not mt5_connector.connect():
                    logger.warning("MT5 connection failed")
                
                # Initialize components
                data_processor = DataProcessor()
                strategy_manager = StrategyManager()
                risk_calculator = RiskCalculator()
                
                # Load strategies
                strategy_manager.load_strategies()
                
                self._initialized = True
                logger.info(f"Trading Bot initialized successfully (v{VERSION})")
                return True
                
            except Exception as e:
                logger.error(f"Failed to initialize trading bot: {str(e)}")
                return False
    
    def get_market_data(self, symbol: str, timeframe: str, bars: int):
        """Get market data through MT5 connector"""
        if mt5_connector:
            return mt5_connector.get_ohlc_data(symbol, timeframe, bars)
        else:
            import pandas as pd
            return pd.DataFrame()

def create_app():
    """Factory function to create Flask app with blueprints"""
    
    # Register blueprints
    from xauusd_trading_bot.api import register_blueprints
    register_blueprints(app)
    
    return app

if __name__ == '__main__':
    logger.info(f"Starting XAU/USD Trading Bot v{VERSION}")
    
    # Initialize trading bot
    bot = TradingBot()
    if bot.initialize():
        logger.info("Bot initialization complete")
    else:
        logger.error("Bot initialization failed")
        sys.exit(1)
    
    # Create Flask app with blueprints
    app = create_app()
    
    # Run Flask app
    port = int(os.getenv('FLASK_PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting Flask API server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug, threaded=True)