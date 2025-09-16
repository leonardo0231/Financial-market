from xauusd_trading_bot._version import __version__
__author__ = "Trading Bot Team"

# Import configuration classes and loader from their dedicated modules
from .schemas import TradingBotConfig, DatabaseConfig, LoggingConfig, TradingConfig
from .loader import ConfigLoader, load_config

__all__ = [
    'TradingBotConfig',
    'DatabaseConfig', 
    'LoggingConfig',
    'TradingConfig',
    'ConfigLoader',
    'load_config'
]