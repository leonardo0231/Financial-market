import logging
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class RiskPolicyManager:
    """Centralized risk management policies to avoid duplication"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.risk_params = self.config.get('risk_management', {})
        
        # Default risk parameters
        self.max_risk_per_trade = self.risk_params.get('max_risk_per_trade', 0.02)  # 2%
        self.max_daily_risk = self.risk_params.get('max_daily_risk', 0.06)  # 6%
        self.min_risk_reward_ratio = self.risk_params.get('min_risk_reward_ratio', 1.5)
        self.max_position_size = self.risk_params.get('max_position_size', 0.1)  # 10% of balance
        self.max_open_positions = self.risk_params.get('max_open_positions', 5)
        
    def validate_trade_risk(self, trade_signal: Dict, account_info: Dict, 
                           open_positions: List[Dict] = None) -> Dict:
        """
        Comprehensive trade risk validation
        Returns: {'valid': bool, 'reason': str, 'adjusted_params': Dict}
        """
        open_positions = open_positions or []
        
        try:
            # 1. Check position count limit
            if len(open_positions) >= self.max_open_positions:
                return {
                    'valid': False, 
                    'reason': f'Maximum open positions exceeded ({self.max_open_positions})',
                    'adjusted_params': None
                }
            
            # 2. Validate risk/reward ratio
            entry_price = trade_signal.get('entry_price', 0)
            stop_loss = trade_signal.get('stop_loss', 0)
            take_profit = trade_signal.get('take_profit', 0)
            
            if entry_price and stop_loss and take_profit:
                risk = abs(entry_price - stop_loss)
                reward = abs(take_profit - entry_price)
                
                if risk > 0:
                    rr_ratio = reward / risk
                    if rr_ratio < self.min_risk_reward_ratio:
                        # Try to adjust take profit to meet minimum R:R
                        direction = 1 if trade_signal.get('signal') == 'BUY' else -1
                        adjusted_tp = entry_price + (direction * risk * self.min_risk_reward_ratio)
                        
                        return {
                            'valid': True,
                            'reason': f'Take profit adjusted to meet minimum R:R ratio ({self.min_risk_reward_ratio})',
                            'adjusted_params': {'take_profit': adjusted_tp}
                        }
            
            # 3. Check daily risk exposure
            balance = account_info.get('balance', 1000)
            daily_loss = sum([pos.get('profit', 0) for pos in open_positions if pos.get('profit', 0) < 0])
            daily_risk_pct = abs(daily_loss) / balance if balance > 0 else 0
            
            if daily_risk_pct >= self.max_daily_risk:
                return {
                    'valid': False,
                    'reason': f'Daily risk limit exceeded: {daily_risk_pct:.2%} >= {self.max_daily_risk:.2%}',
                    'adjusted_params': None
                }
            
            # 4. Validate position size
            volume = trade_signal.get('volume', 0)
            if volume > self.max_position_size * balance / 1000:  # Assuming 1 lot = $1000 exposure
                adjusted_volume = (self.max_position_size * balance / 1000) * 0.9  # 90% of max for safety
                return {
                    'valid': True,
                    'reason': f'Position size reduced to comply with maximum exposure',
                    'adjusted_params': {'volume': adjusted_volume}
                }
            
            # All checks passed
            return {
                'valid': True,
                'reason': 'Trade passes all risk checks',
                'adjusted_params': None
            }
            
        except Exception as e:
            logger.error(f"Error in risk validation: {e}")
            return {
                'valid': False,
                'reason': f'Risk validation error: {str(e)}',
                'adjusted_params': None
            }
    
    def calculate_position_size(self, trade_signal: Dict, account_balance: float, 
                              symbol_info: Dict = None) -> float:
        """Calculate optimal position size based on risk parameters"""
        try:
            entry_price = trade_signal.get('entry_price', 0)
            stop_loss = trade_signal.get('stop_loss', 0)
            
            if not entry_price or not stop_loss:
                return 0.01  # Minimum position size
            
            # Calculate risk per trade in account currency
            risk_amount = account_balance * self.max_risk_per_trade
            price_difference = abs(entry_price - stop_loss)
            
            if price_difference == 0:
                return 0.01
            
            # For XAUUSD, 1 lot = 100 oz, 1 pip = $1 for 0.01 lot
            # Position size = Risk Amount / (Stop Loss Distance in pips * Pip Value)
            if symbol_info:
                pip_value = symbol_info.get('trade_tick_value', 1.0)
                contract_size = symbol_info.get('trade_contract_size', 100)
            else:
                pip_value = 1.0
                contract_size = 100
            
            position_size = risk_amount / (price_difference * pip_value)
            
            # Apply maximum position size limit
            max_size = self.max_position_size * account_balance / (entry_price * contract_size)
            position_size = min(position_size, max_size)
            
            # Round to valid lot sizes (typically 0.01 increments)
            position_size = max(0.01, round(position_size, 2))
            
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.01


class RiskCalculator:
    """Risk management and position sizing calculator"""
    
    def __init__(self, config: Dict = None):
        """Initialize risk calculator with emergency stop mechanism"""
        self.default_risk_percent = 1.0  # Default 1% risk per trade
        self.policy_manager = RiskPolicyManager(config)
        self.max_risk_percent = 2.0      # Maximum 2% risk per trade
        self.max_positions = 3           # Maximum concurrent positions
        self.risk_reward_min = 1.5       # Minimum risk:reward ratio
        
        # Emergency stop configuration
        self.emergency_stop_enabled = True
        self.emergency_loss_threshold = 10.0  # 10% daily loss threshold
        self.max_daily_loss = 5.0           # 5% max daily loss
        self.max_weekly_loss = 10.0         # 10% max weekly loss
        self.max_monthly_loss = 20.0        # 20% max monthly loss
        self.max_drawdown = 15.0            # 15% max drawdown
        
        # Risk parameters by account size
        self.risk_tiers = {
            'micro': {'min_balance': 0, 'max_balance': 1000, 'risk_percent': 0.5},
            'mini': {'min_balance': 1000, 'max_balance': 10000, 'risk_percent': 1.0},
            'standard': {'min_balance': 10000, 'max_balance': 100000, 'risk_percent': 1.5},
            'professional': {'min_balance': 100000, 'max_balance': float('inf'), 'risk_percent': 2.0}
        }
        
    def calculate_position_size(self, signal: Dict, account_balance: float, 
                              symbol_info: Dict = None, open_positions: List[Dict] = None) -> Dict:
        """Calculate optimal position size based on risk parameters"""
        try:
            # Validate inputs
            if account_balance <= 0:
                raise ValueError("Invalid account balance")
            
            if 'stop_loss' not in signal or 'entry_price' not in signal:
                raise ValueError("Signal missing required price levels")
            
            # Emergency stop check
            if self.emergency_stop_enabled and not self._check_emergency_stop_conditions(signal, account_balance):
                # Send emergency notification
                try:
                    from .telegram_notifier import TelegramNotifier
                    notifier = TelegramNotifier()
                    if notifier.is_configured():
                        notifier.send_alert(
                            'error', 
                            f'🚨 EMERGENCY STOP ACTIVATED 🚨\n\nTrading has been suspended due to risk limits being exceeded.\n\nSymbol: {signal.get("symbol", "Unknown")}\nTime: {datetime.now().isoformat()}',
                            priority='critical'
                        )
                except Exception as e:
                    logger.warning(f"Failed to send emergency stop notification: {e}")
                
                return {
                    'position_size': 0,
                    'risk_amount': 0,
                    'error': 'Emergency stop activated - trading suspended'
                }
            
            # Determine risk percentage based on account tier
            risk_percent = self._get_risk_percentage(account_balance)
            
            # Adjust for open positions
            if open_positions >= self.max_positions:
                return {
                    'position_size': 0,
                    'risk_amount': 0,
                    'error': f'Maximum positions ({self.max_positions}) reached'
                }
            
            # Use centralized risk policy validation
            open_positions = open_positions or []
            account_info = {'balance': account_balance}
            risk_validation = self.policy_manager.validate_trade_risk(signal, account_info, open_positions)
            
            if not risk_validation['valid']:
                return {
                    'position_size': 0,
                    'risk_amount': 0,
                    'error': risk_validation['reason']
                }
            
            # Apply any adjustments suggested by risk policy
            if risk_validation.get('adjusted_params'):
                signal.update(risk_validation['adjusted_params'])
            
            # Reduce risk if multiple positions open
            if len(open_positions) > 0:
                risk_percent *= (1 - 0.2 * len(open_positions))  # Reduce by 20% per position
            
            # Calculate risk amount
            risk_amount = account_balance * (risk_percent / 100)
            
            # Calculate stop loss distance
            entry_price = signal['entry_price']
            stop_loss = signal['stop_loss']
            
            if signal['signal'] == 'BUY':
                sl_distance = entry_price - stop_loss
            else:  # SELL
                sl_distance = stop_loss - entry_price
            
            if sl_distance <= 0:
                raise ValueError("Invalid stop loss distance")
            
            # Extract symbol from signal
            symbol = signal.get('symbol', 'XAUUSD')
            
            # Get pip value and calculate position size
            pip_value = self._get_pip_value(symbol_info, symbol)
            sl_pips = sl_distance / pip_value
            
            # Calculate position size based on symbol type
            if 'XAU' in symbol.upper() or 'GOLD' in symbol.upper():
                # For XAUUSD: 1 lot = 100 ounces, pip value calculation different
                pip_value_usd = pip_value  # Already in USD for XAUUSD
                position_size = risk_amount / (sl_pips * pip_value_usd * 100)
            else:
                # Standard forex calculation
                position_size = risk_amount / (sl_pips * 10)  # Standard lot = 100,000 units
            
            # Apply symbol constraints
            if symbol_info:
                position_size = self._apply_symbol_constraints(position_size, symbol_info)
            
            # Validate risk/reward ratio and initialize
            rr_ratio = None
            if 'take_profit' in signal:
                rr_ratio = self._calculate_risk_reward(signal)
                if rr_ratio < self.risk_reward_min:
                    position_size *= 0.5  # Reduce size for poor R:R
            
            return {
                'position_size': round(position_size, 2),
                'risk_amount': risk_amount,
                'risk_percent': risk_percent,
                'sl_distance': sl_distance,
                'sl_pips': sl_pips,
                'pip_value': pip_value,
                'risk_reward_ratio': rr_ratio
            }
            
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return {
                'position_size': 0,
                'risk_amount': 0,
                'error': str(e)
            }
    
    def validate_trade(self, trade_signal: Dict, account_info: Dict = None) -> Dict:
        """Validate trade signal against risk rules"""
        try:
            validation = {
                'valid': True,
                'errors': [],
                'warnings': []
            }
            
            # Check required fields - support both 'signal' and 'type' for backward compatibility
            signal_field = None
            if 'signal' in trade_signal:
                signal_field = 'signal'
            elif 'type' in trade_signal:
                signal_field = 'type'
                # Normalize old 'type' field to 'signal' for internal processing
                trade_signal['signal'] = trade_signal['type']
            else:
                validation['valid'] = False
                validation['errors'].append("Missing required field: 'signal' or 'type'")
                return validation
            
            required_fields = ['entry_price', 'stop_loss']
            for field in required_fields:
                if field not in trade_signal:
                    validation['valid'] = False
                    validation['errors'].append(f"Missing required field: {field}")
            
            if not validation['valid']:
                return validation
            
            # Validate signal type
            signal_value = trade_signal['signal']
            if signal_value not in ['BUY', 'SELL']:
                validation['valid'] = False
                validation['errors'].append("Invalid signal type - must be 'BUY' or 'SELL'")
                return validation
            
            # Validate price levels
            entry = trade_signal['entry_price']
            sl = trade_signal['stop_loss']
            tp = trade_signal.get('take_profit')
            
            if trade_signal['signal'] == 'BUY':
                if sl >= entry:
                    validation['valid'] = False
                    validation['errors'].append("Stop loss must be below entry for BUY")
                
                if tp and tp <= entry:
                    validation['valid'] = False
                    validation['errors'].append("Take profit must be above entry for BUY")
            
            else:  # SELL
                if sl <= entry:
                    validation['valid'] = False
                    validation['errors'].append("Stop loss must be above entry for SELL")
                
                if tp and tp >= entry:
                    validation['valid'] = False
                    validation['errors'].append("Take profit must be below entry for SELL")
            
            # Check risk/reward ratio
            if tp:
                rr_ratio = self._calculate_risk_reward(trade_signal)
                if rr_ratio < self.risk_reward_min:
                    validation['warnings'].append(f"Low risk/reward ratio: {rr_ratio:.2f}")
            
            # Check account constraints
            if account_info:
                # Check margin requirements
                if account_info.get('margin_free', 0) < account_info.get('balance', 0) * 0.2:
                    validation['warnings'].append("Low free margin")
                
                # Check drawdown
                current_drawdown = self._calculate_drawdown(account_info)
                if current_drawdown > 10:
                    validation['warnings'].append(f"High drawdown: {current_drawdown:.1f}%")
                
                if current_drawdown > 20:
                    validation['valid'] = False
                    validation['errors'].append("Maximum drawdown exceeded")
            
            # Check emergency stop conditions
            if hasattr(self, 'emergency_stop_enabled') and self.emergency_stop_enabled:
                emergency_threshold = getattr(self, 'emergency_loss_threshold', 10.0)
                if account_info and current_drawdown > emergency_threshold:
                    validation['valid'] = False
                    validation['errors'].append(f"EMERGENCY STOP: Drawdown {current_drawdown:.1f}% exceeds threshold {emergency_threshold}%")
                    logger.critical(f"🚨 EMERGENCY STOP TRIGGERED: Drawdown {current_drawdown:.1f}% > {emergency_threshold}%")
            
            return validation
            
        except Exception as e:
            logger.error(f"Error validating trade: {str(e)}")
            return {
                'valid': False,
                'errors': [str(e)],
                'warnings': []
            }
    
    def calculate_trade_metrics(self, entry_price: float, exit_price: float, 
                              position_size: float, trade_type: str) -> Dict:
        """Calculate trade performance metrics"""
        try:
            if trade_type == 'BUY':
                pips = (exit_price - entry_price) * 10000  # For forex pairs
                profit_loss = (exit_price - entry_price) * position_size * 100000
            else:  # SELL
                pips = (entry_price - exit_price) * 10000
                profit_loss = (entry_price - exit_price) * position_size * 100000
            
            return {
                'pips': round(pips, 1),
                'profit_loss': round(profit_loss, 2),
                'return_percent': round((profit_loss / (entry_price * position_size * 100000)) * 100, 2)
            }
            
        except Exception as e:
            logger.error(f"Error calculating trade metrics: {str(e)}")
            return {'pips': 0, 'profit_loss': 0, 'return_percent': 0}
    
    def calculate_portfolio_risk(self, open_positions: List[Dict], account_balance: float) -> Dict:
        """Calculate overall portfolio risk"""
        try:
            total_risk = 0
            position_risks = []
            
            for position in open_positions:
                # Calculate risk for each position
                if position['type'] == 0:  # BUY
                    risk = (position['price_open'] - position['sl']) * position['volume'] * 100000
                else:  # SELL
                    risk = (position['sl'] - position['price_open']) * position['volume'] * 100000
                
                total_risk += abs(risk)
                position_risks.append({
                    'symbol': position['symbol'],
                    'risk_amount': abs(risk),
                    'risk_percent': (abs(risk) / account_balance) * 100
                })
            
            return {
                'total_risk_amount': total_risk,
                'total_risk_percent': (total_risk / account_balance) * 100,
                'position_risks': position_risks,
                'positions_count': len(open_positions),
                'avg_risk_per_position': total_risk / len(open_positions) if open_positions else 0
            }
            
        except Exception as e:
            logger.error(f"Error calculating portfolio risk: {str(e)}")
            return {
                'total_risk_amount': 0,
                'total_risk_percent': 0,
                'position_risks': []
            }
    
    def adjust_stop_loss(self, position: Dict, current_price: float, 
                        atr: float, trailing: bool = True) -> Optional[float]:
        """Calculate adjusted stop loss (trailing or breakeven)"""
        try:
            if not trailing:
                # Move to breakeven
                if position['type'] == 0:  # BUY
                    if current_price > position['price_open'] + atr:
                        return position['price_open'] + (atr * 0.1)
                else:  # SELL
                    if current_price < position['price_open'] - atr:
                        return position['price_open'] - (atr * 0.1)
            else:
                # Trailing stop
                if position['type'] == 0:  # BUY
                    new_sl = current_price - (atr * 1.5)
                    if new_sl > position['sl']:
                        return new_sl
                else:  # SELL
                    new_sl = current_price + (atr * 1.5)
                    if new_sl < position['sl']:
                        return new_sl
            
            return None
            
        except Exception as e:
            logger.error(f"Error adjusting stop loss: {str(e)}")
            return None
    
    def calculate_kelly_criterion(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """Calculate optimal position size using Kelly Criterion"""
        try:
            if avg_loss == 0:
                return self.default_risk_percent / 100
            
            # Kelly formula: f = (p * b - q) / b
            # where p = win rate, q = loss rate, b = win/loss ratio
            b = avg_win / avg_loss
            p = win_rate
            q = 1 - win_rate
            
            kelly_percent = ((p * b) - q) / b
            
            # Apply Kelly fraction (usually 25% of full Kelly)
            kelly_fraction = 0.25
            adjusted_kelly = kelly_percent * kelly_fraction * 100
            
            # Cap at maximum risk
            return min(adjusted_kelly, self.max_risk_percent)
            
        except Exception as e:
            logger.error(f"Error calculating Kelly criterion: {str(e)}")
            return self.default_risk_percent
    
    # Private helper methods
    def _get_risk_percentage(self, account_balance: float) -> float:
        """Get risk percentage based on account balance"""
        for tier_name, tier_info in self.risk_tiers.items():
            if tier_info['min_balance'] <= account_balance < tier_info['max_balance']:
                return tier_info['risk_percent']
        
        return self.default_risk_percent
    
    def _get_pip_value(self, symbol_info: Dict = None, symbol: str = 'XAUUSD') -> float:
        """Get pip value for symbol with dynamic broker-specific handling"""
        try:
            symbol_upper = symbol.upper()
            
            if symbol_info and 'point' in symbol_info:
                point = symbol_info['point']
                
                # Dynamic calculation based on actual point value from broker
                if 'XAU' in symbol_upper or 'GOLD' in symbol_upper:
                    # Some brokers use point=0.01, pip=0.1
                    # Others use point=0.1, pip=0.1  
                    # Others use point=1, pip=1
                    if point <= 0.01:
                        pip_value = 0.1  # Standard: point=0.01, pip=0.1
                    elif point <= 0.1:
                        pip_value = point  # Alternative: point=0.1, pip=0.1
                    else:
                        pip_value = 1.0  # Some brokers: point=1, pip=1
                    
                    logger.debug(f"XAUUSD pip calculation: point={point}, pip_value={pip_value}")
                    return pip_value
                
                # For other symbols, use point as pip
                return point
            
            # Fallback defaults when symbol_info not available
            if 'XAU' in symbol_upper or 'GOLD' in symbol_upper:
                # Configurable default via settings
                xau_pip = getattr(self, 'xau_pip_value', 0.1)
                logger.warning(f"Using default XAUUSD pip value: {xau_pip} (configure in settings for broker-specific value)")
                return xau_pip
            elif 'JPY' in symbol_upper:
                return 0.01  # JPY pairs
            elif any(pair in symbol_upper for pair in ['EUR', 'GBP', 'AUD', 'NZD', 'USD']):
                return 0.0001  # Major forex pairs
            else:
                return 0.00001  # Default
                
        except Exception as e:
            logger.error(f"Error getting pip value for {symbol}: {str(e)}")
            return 0.1 if 'XAU' in symbol.upper() else 0.0001
    
    def _check_emergency_stop_conditions(self, signal: Dict, account_balance: float) -> bool:
        """Check emergency stop conditions to prevent catastrophic losses"""
        try:
            if not self.emergency_stop_enabled:
                return True
            
            # Import here to avoid circular dependencies
            from datetime import datetime, timedelta
            from ..database.connection import db_manager
            
            symbol = signal.get('symbol', 'XAUUSD')
            
            # Get performance data from database
            if db_manager and db_manager._initialized:
                try:
                    with db_manager.session_scope() as session:
                        from xauusd_trading_bot.database.models import Performance
                        
                        today = datetime.now().date()
                        week_start = today - timedelta(days=today.weekday())
                        month_start = today.replace(day=1)
                        
                        # Check daily loss
                        daily_perf = session.query(Performance).filter(
                            Performance.date == today,
                            Performance.period_type == 'DAILY'
                        ).first()
                        
                        if daily_perf and daily_perf.net_profit < 0:
                            daily_loss_pct = abs(daily_perf.net_profit) / daily_perf.starting_balance * 100
                            if daily_loss_pct >= self.max_daily_loss:
                                logger.critical(f"Emergency stop: Daily loss {daily_loss_pct:.2f}% exceeds limit {self.max_daily_loss}%")
                                return False
                        
                        # Check weekly loss
                        weekly_perf = session.query(Performance).filter(
                            Performance.date >= week_start,
                            Performance.period_type == 'WEEKLY'
                        ).first()
                        
                        if weekly_perf and weekly_perf.net_profit < 0:
                            weekly_loss_pct = abs(weekly_perf.net_profit) / weekly_perf.starting_balance * 100
                            if weekly_loss_pct >= self.max_weekly_loss:
                                logger.critical(f"Emergency stop: Weekly loss {weekly_loss_pct:.2f}% exceeds limit {self.max_weekly_loss}%")
                                return False
                        
                        # Check monthly loss
                        monthly_perf = session.query(Performance).filter(
                            Performance.date >= month_start,
                            Performance.period_type == 'MONTHLY'
                        ).first()
                        
                        if monthly_perf and monthly_perf.net_profit < 0:
                            monthly_loss_pct = abs(monthly_perf.net_profit) / monthly_perf.starting_balance * 100
                            if monthly_loss_pct >= self.max_monthly_loss:
                                logger.critical(f"Emergency stop: Monthly loss {monthly_loss_pct:.2f}% exceeds limit {self.max_monthly_loss}%")
                                return False
                        
                        # Check drawdown
                        if monthly_perf and monthly_perf.max_drawdown_percent >= self.max_drawdown:
                            logger.critical(f"Emergency stop: Drawdown {monthly_perf.max_drawdown_percent:.2f}% exceeds limit {self.max_drawdown}%")
                            return False
                        
                except Exception as e:
                    logger.warning(f"Could not check emergency stop conditions from database: {e}")
                    # Continue without database check - don't block trading entirely
            
            return True
            
        except Exception as e:
            logger.error(f"Error in emergency stop check: {e}")
            # If check fails, allow trading to continue but log the error
            return True
    
    def _apply_symbol_constraints(self, position_size: float, symbol_info: Dict) -> float:
        """Apply symbol-specific constraints to position size"""
        # Minimum volume
        min_volume = symbol_info.get('volume_min', 0.01)
        if position_size < min_volume:
            position_size = min_volume
        
        # Maximum volume
        max_volume = symbol_info.get('volume_max', 100.0)
        if position_size > max_volume:
            position_size = max_volume
        
        # Volume step
        volume_step = symbol_info.get('volume_step', 0.01)
        if volume_step > 0:
            position_size = round(position_size / volume_step) * volume_step
        
        return position_size
    
    def _calculate_risk_reward(self, signal: Dict) -> float:
        """Calculate risk/reward ratio"""
        try:
            entry = signal['entry_price']
            sl = signal['stop_loss']
            tp = signal['take_profit']
            
            if signal['signal'] == 'BUY':
                risk = entry - sl
                reward = tp - entry
            else:  # SELL
                risk = sl - entry
                reward = entry - tp
            
            if risk <= 0:
                return 0
            
            return reward / risk
            
        except Exception as e:
            logger.error(f"Error calculating R:R ratio: {str(e)}")
            return 0
    
    def _calculate_drawdown(self, account_info: Dict) -> float:
        """Calculate current drawdown percentage"""
        try:
            balance = account_info.get('balance', 0)
            equity = account_info.get('equity', balance)
            
            if balance == 0:
                return 0
            
            drawdown = ((balance - equity) / balance) * 100
            return max(0, drawdown)
            
        except Exception as e:
            logger.error(f"Error calculating drawdown: {str(e)}")
            return 0