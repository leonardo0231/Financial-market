"""
Database module for XAU/USD Trading Bot
SQLAlchemy ORM models and database connection management
"""

"""
Database module for XAU/USD Trading Bot  
Contains ORM models and database management
"""

from .models import Trade, Performance, Signal
from .connection import DatabaseManager, get_db_session, db_manager

__all__ = [
    'Trade',
    'Performance',
    'Signal',
    'DatabaseManager',
    'get_db_session',
    'db_manager'  # Export singleton instance
]