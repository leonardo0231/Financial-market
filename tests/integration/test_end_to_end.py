import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Path configuration handled by pyproject.toml pythonpath setting

from xauusd_trading_bot.core.mt5_connector import MT5Connector
from xauusd_trading_bot.utils.risk_calculator import RiskCalculator
from xauusd_trading_bot.database.connection import DatabaseManager


class TestEndToEndWorkflow:
    """Test suite for complete trading workflow"""
    
    @pytest.fixture
    def setup_components(self):
        """Setup all trading components"""
        return {
            'mt5_connector': MT5Connector(),
            'risk_calculator': RiskCalculator(),
            'db_manager': DatabaseManager()
        }
    
    def test_complete_trade_workflow_signal_field(self, setup_components):
        """Test complete workflow from signal generation to execution using 'signal' field"""
        mt5_connector = setup_components['mt5_connector']
        risk_calculator = setup_components['risk_calculator']
        
        # Mock successful connections and operations
        with patch('xauusd_trading_bot.core.mt5_connector.mt5') as mock_mt5:
            mock_mt5.symbol_info.return_value = Mock(visible=True, point=0.01)
            mock_mt5.symbol_info_tick.return_value = Mock(ask=2000.5, bid=2000.0)
            mock_mt5.order_send.return_value = Mock(retcode=10009, order=12345)
            
            mt5_connector.connected = True
            
            # Step 1: Generate signal (using 'signal' field)
            trade_signal = {
                'signal': 'BUY',
                'symbol': 'XAUUSD', 
                'entry_price': 2000.0,
                'stop_loss': 1990.0,
                'take_profit': 2020.0
            }
            
            # Step 2: Calculate position size
            position_result = risk_calculator.calculate_position_size(
                signal=trade_signal,
                account_balance=10000.0
            )
            
            assert 'position_size' in position_result
            assert position_result['position_size'] > 0
            
            # Step 3: Add position size to signal
            trade_signal['volume'] = position_result['position_size']
            
            # Step 4: Execute trade
            execution_result = mt5_connector.execute_trade(trade_signal)
            
            # Enhanced assertion: check both success and retcode
            assert execution_result['success'] is True, f"Trade execution failed: {execution_result}"
            if 'retcode' in execution_result:
                assert execution_result['retcode'] == 10009, f"Expected retcode 10009, got {execution_result.get('retcode')}"
            
    def test_workflow_backward_compatibility_type_field(self, setup_components):
        """Test workflow maintains backward compatibility with 'type' field"""
        mt5_connector = setup_components['mt5_connector']
        
        with patch('xauusd_trading_bot.core.mt5_connector.mt5') as mock_mt5:
            mock_mt5.symbol_info.return_value = Mock(visible=True, point=0.01)
            mock_mt5.symbol_info_tick.return_value = Mock(ask=2000.5, bid=2000.0)
            mock_mt5.order_send.return_value = Mock(retcode=10009, order=12345)
            
            mt5_connector.connected = True
            
            # Test with old 'type' field format
            trade_signal_old = {
                'type': 'BUY',  # Old format
                'symbol': 'XAUUSD',
                'volume': 0.1,
                'stop_loss': 1990.0,
                'take_profit': 2020.0
            }
            
            result = mt5_connector.execute_trade(trade_signal_old)
            assert result['success'] is True
    
    def test_database_manager_shutdown_without_error(self, setup_components):
        """Test that DatabaseManager can be shutdown without AttributeError"""
        db_manager = setup_components['db_manager']
        
        # This should NOT raise AttributeError
        try:
            db_manager.close()
            shutdown_successful = True
        except AttributeError:
            shutdown_successful = False
            
        assert shutdown_successful, "DatabaseManager.close() should exist and be callable"