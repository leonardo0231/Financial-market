import pytest
from unittest.mock import Mock, patch
import sys
import os

# Path configuration handled by pyproject.toml pythonpath setting

from xauusd_trading_bot.core.mt5_connector import MT5Connector


class TestMT5Connector:
    """Test suite for MT5Connector class"""
    
    @pytest.fixture
    def mt5_connector(self):
        """Create MT5Connector instance for testing"""
        return MT5Connector()
    
    @patch('xauusd_trading_bot.core.mt5_connector.mt5')
    def test_execute_trade_signal_type_compatibility(self, mock_mt5, mt5_connector):
        """Test execute_trade supports both 'signal' and 'type' fields (critical regression test)"""
        # Mock MT5 connection and successful trade
        mt5_connector.connected = True
        mock_mt5.symbol_info.return_value = Mock(visible=True, point=0.01)
        mock_mt5.symbol_info_tick.return_value = Mock(ask=2000.5, bid=2000.0)
        mock_mt5.order_send.return_value = Mock(retcode=10009, order=12345)
        
        # Test with 'signal' field (new format)
        signal_new = {
            'signal': 'BUY',
            'symbol': 'XAUUSD',
            'volume': 0.1,
            'stop_loss': 1990.0,
            'take_profit': 2020.0
        }
        
        result = mt5_connector.execute_trade(signal_new)
        assert result['success'] is True
        
        # Test with 'type' field (backward compatibility)
        signal_old = {
            'type': 'BUY',
            'symbol': 'XAUUSD',
            'volume': 0.1,
            'stop_loss': 1990.0,
            'take_profit': 2020.0
        }
        
        result = mt5_connector.execute_trade(signal_old)
        assert result['success'] is True
        
        # Test missing both fields (should fail gracefully)
        signal_invalid = {
            'symbol': 'XAUUSD',
            'volume': 0.1,
            'stop_loss': 1990.0
        }
        
        result = mt5_connector.execute_trade(signal_invalid)
        assert result['success'] is False
        assert 'Missing signal/type field' in result['error']
    
    @patch('xauusd_trading_bot.core.mt5_connector.mt5')
    def test_get_symbol_info_success(self, mock_mt5, mt5_connector):
        """Test get_symbol_info method works correctly"""
        # Mock MT5 connection and symbol info
        mt5_connector.connected = True
        mock_symbol_info = Mock()
        mock_symbol_info._asdict.return_value = {
            'symbol': 'XAUUSD',
            'bid': 2000.0,
            'ask': 2000.5,
            'point': 0.01
        }
        mock_mt5.symbol_info.return_value = mock_symbol_info
        
        result = mt5_connector.get_symbol_info('XAUUSD')
        assert result['symbol'] == 'XAUUSD'
        assert 'bid' in result
        assert 'ask' in result
    
    @patch('xauusd_trading_bot.core.mt5_connector.mt5')
    def test_get_open_positions_success(self, mock_mt5, mt5_connector):
        """Test get_open_positions method works correctly"""
        # Mock MT5 connection and positions
        mt5_connector.connected = True
        mock_position = Mock()
        mock_position._asdict.return_value = {
            'ticket': 12345,
            'symbol': 'XAUUSD',
            'volume': 0.1,
            'profit': 10.5
        }
        mock_mt5.positions_get.return_value = [mock_position]
        
        result = mt5_connector.get_open_positions('XAUUSD')
        assert len(result) == 1
        assert result[0]['ticket'] == 12345
        assert result[0]['symbol'] == 'XAUUSD'