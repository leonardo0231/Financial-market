"""
Core module for XAU/USD Trading Bot
Contains MT5 connector, Finnhub client and data processing components
"""

"""
XAU/USD Trading Bot Core Module
Contains MT5 connector and market data components
"""

from .mt5_connector import MT5Connector
from .finnhub_client import FinnhubClient

__all__ = [
    'MT5Connector',
    'FinnhubClient'
]