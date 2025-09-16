"""
Centralized logging configuration for the trading bot
"""
import os
import sys
import logging
import logging.handlers
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import json

class TradingBotLogger:
    """Centralized logger configuration for the trading bot"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or 'config/logging.json'
        self.log_config = self._load_log_config()
        self._setup_logging()
    
    def _load_log_config(self) -> Dict[str, Any]:
        """Load logging configuration from JSON file or use defaults"""
        default_config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S"
                },
                "simple": {
                    "format": "%(asctime)s - %(levelname)s - %(message)s",
                    "datefmt": "%H:%M:%S"
                },
                "json": {
                    "format": '{"timestamp": "%(asctime)s", "logger": "%(name)s", "level": "%(levelname)s", "file": "%(filename)s", "line": %(lineno)d, "function": "%(funcName)s", "message": "%(message)s"}',
                    "datefmt": "%Y-%m-%d %H:%M:%S"
                },
                "trading": {
                    "format": "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S"
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": "INFO",
                    "formatter": "simple",
                    "stream": "ext://sys.stdout"
                },
                "file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": "DEBUG",
                    "formatter": "detailed",
                    "filename": "logs/trading.log",
                    "maxBytes": 10485760,
                    "backupCount": 5,
                    "encoding": "utf-8"
                },
                "error_file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": "ERROR",
                    "formatter": "detailed",
                    "filename": "logs/errors.log",
                    "maxBytes": 10485760,
                    "backupCount": 3,
                    "encoding": "utf-8"
                },
                "trading_file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": "INFO",
                    "formatter": "trading",
                    "filename": "logs/trading_signals.log",
                    "maxBytes": 5242880,
                    "backupCount": 10,
                    "encoding": "utf-8"
                },
                "json_file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": "DEBUG",
                    "formatter": "json",
                    "filename": "logs/trading_bot.json",
                    "maxBytes": 10485760,
                    "backupCount": 3,
                    "encoding": "utf-8"
                }
            },
            "loggers": {
                "": {
                    "level": "DEBUG",
                    "handlers": ["console", "file", "error_file"],
                    "propagate": False
                },
                "trading_bot": {
                    "level": "DEBUG",
                    "handlers": ["console", "file", "trading_file", "json_file"],
                    "propagate": False
                },
                "trading_signals": {
                    "level": "INFO",
                    "handlers": ["trading_file", "json_file"],
                    "propagate": False
                },
                "database": {
                    "level": "WARNING",
                    "handlers": ["file", "error_file"],
                    "propagate": False
                },
                "mt5": {
                    "level": "INFO",
                    "handlers": ["file", "trading_file"],
                    "propagate": False
                },
                "strategies": {
                    "level": "DEBUG",
                    "handlers": ["file", "trading_file"],
                    "propagate": False
                },
                "risk": {
                    "level": "INFO",
                    "handlers": ["file", "trading_file", "error_file"],
                    "propagate": False
                },
                "api": {
                    "level": "INFO",
                    "handlers": ["file", "json_file"],
                    "propagate": False
                },
                "telegram": {
                    "level": "INFO",
                    "handlers": ["file", "trading_file"],
                    "propagate": False
                }
            }
        }
        
        # Try to load from config file
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    file_config = json.load(f)
                    # Merge with defaults
                    default_config.update(file_config)
            except Exception as e:
                print(f"Warning: Could not load logging config from {self.config_file}: {e}")
                print("Using default configuration")
        
        return default_config
    
    def _setup_logging(self):
        """Setup logging configuration"""
        # Create logs directory
        os.makedirs('logs', exist_ok=True)
        
        # Configure logging
        logging.config.dictConfig(self.log_config)
        
        # Set up root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        
        # Suppress noisy third-party loggers
        noisy_loggers = [
            'urllib3.connectionpool',
            'requests.packages.urllib3',
            'sqlalchemy.engine',
            'sqlalchemy.pool',
            'matplotlib',
            'PIL',
            'asyncio'
        ]
        
        for logger_name in noisy_loggers:
            logging.getLogger(logger_name).setLevel(logging.WARNING)
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger instance"""
        return logging.getLogger(name)
    
    def get_trading_logger(self) -> logging.Logger:
        """Get the main trading logger"""
        return logging.getLogger('trading_bot')
    
    def get_signal_logger(self) -> logging.Logger:
        """Get the trading signals logger"""
        return logging.getLogger('trading_signals')
    
    def get_database_logger(self) -> logging.Logger:
        """Get the database logger"""
        return logging.getLogger('database')
    
    def get_mt5_logger(self) -> logging.Logger:
        """Get the MT5 logger"""
        return logging.getLogger('mt5')
    
    def get_strategy_logger(self) -> logging.Logger:
        """Get the strategies logger"""
        return logging.getLogger('strategies')
    
    def get_risk_logger(self) -> logging.Logger:
        """Get the risk management logger"""
        return logging.getLogger('risk')
    
    def get_api_logger(self) -> logging.Logger:
        """Get the API logger"""
        return logging.getLogger('api')
    
    def get_telegram_logger(self) -> logging.Logger:
        """Get the Telegram logger"""
        return logging.getLogger('telegram')
    
    def log_trade_signal(self, signal_type: str, symbol: str, price: float, 
                        confidence: float, strategy: str, **kwargs):
        """Log a trading signal with structured data"""
        signal_logger = self.get_signal_logger()
        
        signal_data = {
            'timestamp': datetime.now().isoformat(),
            'signal_type': signal_type,
            'symbol': symbol,
            'price': price,
            'confidence': confidence,
            'strategy': strategy,
            **kwargs
        }
        
        signal_logger.info(f"TRADE_SIGNAL: {signal_type} {symbol} @ {price} | "
                          f"Confidence: {confidence:.2%} | Strategy: {strategy}")
        
        # Log structured data for analysis
        signal_logger.debug(f"SIGNAL_DATA: {json.dumps(signal_data)}")
    
    def log_trade_execution(self, action: str, symbol: str, volume: float, 
                           price: float, ticket: str = None, **kwargs):
        """Log trade execution with structured data"""
        trading_logger = self.get_trading_logger()
        
        execution_data = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'symbol': symbol,
            'volume': volume,
            'price': price,
            'ticket': ticket,
            **kwargs
        }
        
        trading_logger.info(f"TRADE_EXECUTION: {action} {volume} {symbol} @ {price} | "
                           f"Ticket: {ticket or 'N/A'}")
        
        # Log structured data
        trading_logger.debug(f"EXECUTION_DATA: {json.dumps(execution_data)}")
    
    def log_error(self, error: Exception, context: str = "", **kwargs):
        """Log errors with context"""
        error_logger = logging.getLogger('trading_bot')
        
        error_data = {
            'timestamp': datetime.now().isoformat(),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context,
            **kwargs
        }
        
        error_logger.error(f"ERROR in {context}: {type(error).__name__}: {error}")
        error_logger.debug(f"ERROR_DATA: {json.dumps(error_data)}")
    
    def log_performance(self, metric: str, value: float, **kwargs):
        """Log performance metrics"""
        perf_logger = self.get_trading_logger()
        
        perf_data = {
            'timestamp': datetime.now().isoformat(),
            'metric': metric,
            'value': value,
            **kwargs
        }
        
        perf_logger.info(f"PERFORMANCE: {metric} = {value}")
        perf_logger.debug(f"PERF_DATA: {json.dumps(perf_data)}")
    
    def create_logging_config_file(self, filename: str = "config/logging.json"):
        """Create a sample logging configuration file"""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.log_config, f, indent=2, ensure_ascii=False)
        
        print(f"Logging configuration saved to {filename}")


# Global logger instance
_logger_instance = None

def get_logger(name: str = 'trading_bot') -> logging.Logger:
    """Get a logger instance (convenience function)"""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = TradingBotLogger()
    return _logger_instance.get_logger(name)

def get_trading_logger() -> logging.Logger:
    """Get the main trading logger"""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = TradingBotLogger()
    return _logger_instance.get_trading_logger()

def get_signal_logger() -> logging.Logger:
    """Get the trading signals logger"""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = TradingBotLogger()
    return _logger_instance.get_signal_logger()

def setup_logging(config_file: Optional[str] = None):
    """Setup logging configuration"""
    global _logger_instance
    _logger_instance = TradingBotLogger(config_file)
    return _logger_instance

# Example usage functions
def log_trade_signal(signal_type: str, symbol: str, price: float, 
                    confidence: float, strategy: str, **kwargs):
    """Log a trading signal"""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = TradingBotLogger()
    _logger_instance.log_trade_signal(signal_type, symbol, price, confidence, strategy, **kwargs)

def log_trade_execution(action: str, symbol: str, volume: float, 
                       price: float, ticket: str = None, **kwargs):
    """Log trade execution"""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = TradingBotLogger()
    _logger_instance.log_trade_execution(action, symbol, volume, price, ticket, **kwargs)

def log_error(error: Exception, context: str = "", **kwargs):
    """Log errors"""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = TradingBotLogger()
    _logger_instance.log_error(error, context, **kwargs)

def log_performance(metric: str, value: float, **kwargs):
    """Log performance metrics"""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = TradingBotLogger()
    _logger_instance.log_performance(metric, value, **kwargs)

