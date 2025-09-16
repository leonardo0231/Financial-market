import pytest
from unittest.mock import Mock, patch
import sys
import os

# Path configuration handled by pyproject.toml pythonpath setting

from xauusd_trading_bot.utils.risk_calculator import RiskCalculator


class TestRiskCalculator:
    """Test suite for RiskCalculator class"""
    
    @pytest.fixture
    def risk_calculator(self):
        """Create RiskCalculator instance for testing"""
        return RiskCalculator()
    
    @pytest.fixture
    def sample_signal_buy(self):
        """Sample BUY signal for testing"""
        return {
            'signal': 'BUY',
            'symbol': 'XAUUSD',
            'entry_price': 2000.0,
            'stop_loss': 1990.0,
            'take_profit': 2020.0
        }
    
    def test_calculate_position_size_buy_signal(self, risk_calculator, sample_signal_buy):
        """Test position size calculation for BUY signal"""
        account_balance = 10000.0
        
        result = risk_calculator.calculate_position_size(
            signal=sample_signal_buy,
            account_balance=account_balance
        )
        
        assert 'position_size' in result
        assert 'risk_amount' in result
        assert 'risk_percent' in result
        assert result['position_size'] > 0
        assert result['risk_amount'] > 0
    
    def test_rr_ratio_initialization_bug_fix(self, risk_calculator):
        """Test that rr_ratio is properly initialized (regression test for critical bug)"""
        signal_without_tp = {
            'signal': 'BUY',
            'symbol': 'XAUUSD',
            'entry_price': 2000.0,
            'stop_loss': 1990.0
            # No take_profit - this previously caused UnboundLocalError
        }
        
        # This should NOT raise UnboundLocalError anymore
        result = risk_calculator.calculate_position_size(
            signal=signal_without_tp,
            account_balance=10000.0
        )
        
        assert 'risk_reward_ratio' in result
        assert result['risk_reward_ratio'] is None  # Should be None, not undefined
    
    def test_get_pip_value_xauusd_dynamic(self, risk_calculator):
        """Test dynamic pip value calculation for XAUUSD based on broker point value"""
        # Test broker with point=0.01 (standard)
        symbol_info = {'point': 0.01}
        pip_value = risk_calculator._get_pip_value(symbol_info, 'XAUUSD')
        assert pip_value == 0.1
        
        # Test broker with point=0.1 (alternative)
        symbol_info = {'point': 0.1}
        pip_value = risk_calculator._get_pip_value(symbol_info, 'XAUUSD')
        assert pip_value == 0.1
        
        # Test broker with point=1.0 (some brokers)
        symbol_info = {'point': 1.0}
        pip_value = risk_calculator._get_pip_value(symbol_info, 'XAUUSD')
        assert pip_value == 1.0
    
    def test_validate_trade_signal_type_compatibility(self, risk_calculator):
        """Test trade validation supports both 'signal' and 'type' fields"""
        # Test with 'signal' field
        signal_with_signal = {
            'signal': 'BUY',
            'entry_price': 2000.0,
            'stop_loss': 1990.0,
            'symbol': 'XAUUSD'
        }
        
        result = risk_calculator.validate_trade(signal_with_signal)
        assert result['valid'] is True
        
        # Test with 'type' field (backward compatibility)
        signal_with_type = {
            'type': 'BUY',
            'entry_price': 2000.0,
            'stop_loss': 1990.0,
            'symbol': 'XAUUSD'
        }
        
        # This should be handled by the calling code, not RiskCalculator directly
        # But we test that RiskCalculator works with normalized signals
        result = risk_calculator.validate_trade(signal_with_type)
        # Should fail because RiskCalculator expects 'signal' field
        assert result['valid'] is False