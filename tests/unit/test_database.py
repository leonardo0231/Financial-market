import pytest
from unittest.mock import Mock, patch
import sys
import os

# Path configuration handled by pyproject.toml pythonpath setting

from xauusd_trading_bot.database.connection import DatabaseManager


class TestDatabaseManager:
    """Test suite for DatabaseManager class"""
    
    @pytest.fixture
    def db_manager(self):
        """Create DatabaseManager instance for testing"""
        return DatabaseManager()
    
    def test_close_method_exists(self, db_manager):
        """Test that close() method exists and is callable"""
        assert hasattr(db_manager, 'close'), "close() method must exist to prevent AttributeError"
        assert callable(getattr(db_manager, 'close')), "close() must be callable"
    
    @patch('xauusd_trading_bot.database.connection.create_engine')
    @patch('xauusd_trading_bot.database.connection.sessionmaker')
    @patch('xauusd_trading_bot.database.connection.scoped_session')
    def test_close_method_cleanup(self, mock_scoped_session, mock_sessionmaker, mock_create_engine, db_manager):
        """Test that close() method properly cleans up resources"""
        # Setup mocks for initialization
        mock_engine = Mock()
        mock_connection = Mock()
        mock_engine.connect.return_value.__enter__.return_value = mock_connection
        mock_create_engine.return_value = mock_engine
        
        mock_session_factory = Mock()
        mock_sessionmaker.return_value = mock_session_factory
        
        mock_scoped_session_registry = Mock()
        mock_scoped_session.return_value = mock_scoped_session_registry
        
        # Initialize the manager
        assert db_manager.initialize() is True
        
        # Call close method - this should NOT raise AttributeError
        try:
            db_manager.close()
            close_successful = True
        except AttributeError as e:
            close_successful = False
            pytest.fail(f"close() method missing or broken: {e}")
        
        assert close_successful
        assert db_manager._initialized is False
        
        # Verify proper cleanup was called
        mock_scoped_session_registry.remove.assert_called_once()
        mock_engine.dispose.assert_called_once()