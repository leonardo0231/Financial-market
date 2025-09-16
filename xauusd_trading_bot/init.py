"""
XAU/USD Trading Bot
Advanced AI-powered trading bot with n8n orchestration
"""

from ._version import VERSION, __version__, VERSION_INFO

# Make version available at package level
__version__ = VERSION

# Package metadata
__title__ = "xauusd-trading-bot"
__description__ = "Advanced AI-powered XAU/USD trading bot with n8n orchestration"
__author__ = "Trading Bot Team"
__license__ = "MIT"
# __url__ = "https://github.com/yourusername/xauusd-trading-bot"

# Export version info
__all__ = [
    '__version__',
    'VERSION',
    'VERSION_INFO',
    '__title__',
    '__description__',
    '__author__',
    '__license__'
    # '__url__'
]