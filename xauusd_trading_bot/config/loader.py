import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

try:
    from .schemas import TradingBotConfig
    SCHEMAS_AVAILABLE = True
except ImportError:
    SCHEMAS_AVAILABLE = False
    
    # Fallback TradingBotConfig
    class TradingBotConfig:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

logger = logging.getLogger(__name__)

class ConfigLoader:
    """Configuration loader with validation and environment variable support"""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
    
    def load_from_file(self, file_path: Optional[str] = None):
        """Load configuration from file with validation"""
        
        if file_path is None:
            # Try multiple file formats
            for filename in ["config.yaml", "config.yml", "settings.json", "config.json"]:
                config_file = self.config_dir / filename
                if config_file.exists():
                    file_path = str(config_file)
                    break
            else:
                logger.warning("No configuration file found, using environment variables only")
                return self.load_from_env()
        
        logger.info(f"Loading configuration from: {file_path}")
        
        try:
            config_path = Path(file_path)
            
            if not config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {file_path}")
            
            # Load based on file extension
            if config_path.suffix.lower() in ['.yaml', '.yml'] and YAML_AVAILABLE:
                with open(config_path, 'r', encoding='utf-8') as f:
                    raw_config = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                with open(config_path, 'r', encoding='utf-8') as f:
                    raw_config = json.load(f)
            else:
                raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
            
            # Override with environment variables
            config_data = self._merge_env_vars(raw_config)
            
            # Validate and create configuration
            if SCHEMAS_AVAILABLE:
                return TradingBotConfig(**config_data)
            else:
                return TradingBotConfig(**config_data)
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            # Return default config as fallback
            return self.load_from_env()
    
    def load_from_env(self):
        """Load configuration from environment variables only"""
        logger.info("Loading configuration from environment variables")
        
        config_data = {
            "database": {
                "host": os.getenv("POSTGRES_HOST", "postgres"),
                "port": int(os.getenv("POSTGRES_PORT", "5432")),
                "user": os.getenv("POSTGRES_USER", "trading_bot"),
                "password": os.getenv("POSTGRES_PASSWORD") or None,
                "database": os.getenv("POSTGRES_DB", "trading_db")
            },
            "logging": {
                "level": os.getenv("LOG_LEVEL", "INFO"),
                "file": os.getenv("LOG_FILE", "logs/trading.log")
            },
            "mt5": {
                "login": int(os.getenv("MT5_LOGIN")) if os.getenv("MT5_LOGIN") else None,
                "password": os.getenv("MT5_PASSWORD"),
                "server": os.getenv("MT5_SERVER")
            },
            "telegram": {
                "bot_token": os.getenv("TELEGRAM_TOKEN"),
                "admin_chat_ids": [
                    int(chat_id.strip()) 
                    for chat_id in os.getenv("TELEGRAM_CHAT_IDS", "").split(",")
                    if chat_id.strip()
                ]
            },
            "trading": {
                "symbols": ["XAUUSD"],
                "max_positions": int(os.getenv("MAX_POSITIONS", "3")),
                "default_risk_percent": float(os.getenv("DEFAULT_RISK_PERCENT", "1.0"))
            }
        }
        
        return TradingBotConfig(**config_data)
    
    def _merge_env_vars(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Merge environment variables into configuration"""
        
        # Database overrides
        if "database" not in config_data:
            config_data["database"] = {}
        
        db_overrides = {
            "host": os.getenv("POSTGRES_HOST"),
            "port": os.getenv("POSTGRES_PORT"),
            "user": os.getenv("POSTGRES_USER"),
            "password": os.getenv("POSTGRES_PASSWORD"),
            "database": os.getenv("POSTGRES_DB")
        }
        
        for key, value in db_overrides.items():
            if value is not None:
                if key == "port":
                    config_data["database"][key] = int(value)
                else:
                    config_data["database"][key] = value
        
        # MT5 overrides
        if "mt5" not in config_data:
            config_data["mt5"] = {}
        
        mt5_overrides = {
            "login": os.getenv("MT5_LOGIN"),
            "password": os.getenv("MT5_PASSWORD"),
            "server": os.getenv("MT5_SERVER")
        }
        
        for key, value in mt5_overrides.items():
            if value is not None:
                if key == "login":
                    config_data["mt5"][key] = int(value)
                else:
                    config_data["mt5"][key] = value
        
        # Telegram overrides
        if "telegram" not in config_data:
            config_data["telegram"] = {}
        
        if os.getenv("TELEGRAM_TOKEN"):
            config_data["telegram"]["bot_token"] = os.getenv("TELEGRAM_TOKEN")
        
        if os.getenv("TELEGRAM_CHAT_IDS"):
            chat_ids = [
                int(chat_id.strip()) 
                for chat_id in os.getenv("TELEGRAM_CHAT_IDS").split(",")
                if chat_id.strip()
            ]
            config_data["telegram"]["admin_chat_ids"] = chat_ids
        
        # Logging overrides
        if "logging" not in config_data:
            config_data["logging"] = {}
        
        if os.getenv("LOG_LEVEL"):
            config_data["logging"]["level"] = os.getenv("LOG_LEVEL")
        
        if os.getenv("LOG_FILE"):
            config_data["logging"]["file"] = os.getenv("LOG_FILE")
        
        return config_data

# Global config instance
_config: Optional[TradingBotConfig] = None

def load_config(config_file: Optional[str] = None):
    """Load and cache configuration"""
    global _config
    
    if _config is None:
        loader = ConfigLoader()
        _config = loader.load_from_file(config_file)
        logger.info("Configuration loaded and cached successfully")
    
    return _config

def get_config():
    """Get cached configuration"""
    if _config is None:
        raise RuntimeError("Configuration not loaded. Call load_config() first.")
    return _config