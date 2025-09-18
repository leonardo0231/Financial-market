import logging
import json
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from .al_brooks import AlBrooksStrategy
from .linda_raschke import LindaRaschkeStrategy
from .ict import ICTStrategy

logger = logging.getLogger(__name__)


class StrategyManager:
    """Manages and coordinates multiple trading strategies"""
    
    def __init__(self):
        """Initialize strategy manager"""
        self.strategies = {
            'al_brooks': AlBrooksStrategy(),
            'linda_raschke': LindaRaschkeStrategy(),
            'ict': ICTStrategy()
        }
        
        # Strategy weights for combination
        self.weights = {
            'al_brooks': 1.0,
            'linda_raschke': 1.0,
            'ict': 1.0
        }
        
        # Strategy states
        self.enabled_strategies = ['al_brooks', 'linda_raschke', 'ict']
        self.combination_mode = 'weighted'  # 'weighted', 'majority', 'unanimous'
        
        # Performance tracking
        self.performance = {
            strategy: {
                'trades': 0,
                'wins': 0,
                'losses': 0,
                'total_pips': 0.0,
                'win_rate': 0.0
            } for strategy in self.strategies
        }
        
    def load_strategies(self) -> None:
        """Load strategy configurations (synchronous)"""
        try:
            with open('config/strategies.json', 'r') as f:
                strategy_config = json.load(f)  # Use different variable name
            
            # Update strategy parameters
            for strategy_name, params in strategy_config.get('parameters', {}).items():
                if strategy_name in self.strategies:
                    self.strategies[strategy_name].update_parameters(params)
            
            # Update weights
            self.weights.update(strategy_config.get('weights', {}))
            
            # Update enabled strategies
            self.enabled_strategies = strategy_config.get('enabled', self.enabled_strategies)
            
            # Update combination mode
            self.combination_mode = strategy_config.get('combination_mode', self.combination_mode)
            
            logger.info(f"Strategies loaded successfully. Enabled: {self.enabled_strategies}")
            logger.info(f"Combination mode: {self.combination_mode}")
            
        except FileNotFoundError:
            logger.warning("strategies.json not found, using default configuration")
            self._create_default_config()
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in strategies.json: {str(e)}")
            logger.info("Using default strategy configuration")
        except Exception as e:
            logger.error(f"Error loading strategies: {str(e)}")
            logger.info("Using default strategy configuration")
    
    def _create_default_config(self):
        """Create default strategy configuration"""
        default_config = {
            "enabled": ["al_brooks", "linda_raschke", "ict"],
            "weights": {
                "al_brooks": 1.0,
                "linda_raschke": 1.0,
                "ict": 1.0
            },
            "combination_mode": "weighted",
            "parameters": {
                "al_brooks": {
                    "min_trend_strength": 0.6,
                    "pullback_depth": 0.5,
                    "risk_reward_ratio": 2.0
                },
                "linda_raschke": {
                    "turtle_soup_lookback": 20,
                    "holy_grail_adx_threshold": 30,
                    "80_20_threshold": 0.8
                },
                "ict": {
                    "liquidity_lookback": 20,
                    "order_block_strength": 0.7,
                    "risk_reward_min": 2.0
                }
            }
        }
        
        try:
            import os
            os.makedirs('config', exist_ok=True)
            with open('config/strategies.json', 'w') as f:
                json.dump(default_config, f, indent=4)
            logger.info("Created default strategies.json")
        except Exception as e:
            logger.error(f"Failed to create default config: {str(e)}")
    
    def analyze(self, df: pd.DataFrame, requested_strategies: List[str]) -> Dict:
        """Analyze market with specified strategies"""
        try:
            # Determine which strategies to use
            if requested_strategies and requested_strategies != ['all']:
                strategies_to_use = [s for s in requested_strategies if s in self.enabled_strategies]
            else:
                strategies_to_use = self.enabled_strategies
            
            if not strategies_to_use:
                return self._neutral_signal("No strategies enabled")
            
            # Collect signals from each strategy
            signals = {}
            for strategy_name in strategies_to_use:
                if strategy_name in self.strategies:
                    strategy = self.strategies[strategy_name]
                    signal = strategy.analyze(df)
                    signals[strategy_name] = signal
                    
                    logger.info(f"{strategy_name} signal: {signal['signal']} "
                              f"(strength: {signal.get('strength', 0):.2f})")
            
            # Combine signals based on mode
            combined_signal = self._combine_signals(signals, df)
            
            # Add individual strategy signals for transparency
            combined_signal['individual_signals'] = signals
            combined_signal['strategies_used'] = strategies_to_use
            
            return combined_signal
            
        except Exception as e:
            logger.error(f"Error in strategy analysis: {str(e)}")
            return self._neutral_signal(f"Analysis error: {str(e)}")
    
    def _combine_signals(self, signals: Dict, df: pd.DataFrame) -> Dict:
        """Combine multiple strategy signals"""
        try:
            if not signals:
                return self._neutral_signal("No signals to combine")
            
            if self.combination_mode == 'weighted':
                return self._weighted_combination(signals, df)
            elif self.combination_mode == 'majority':
                return self._majority_combination(signals, df)
            elif self.combination_mode == 'unanimous':
                return self._unanimous_combination(signals, df)
            else:
                return self._weighted_combination(signals, df)
                
        except Exception as e:
            logger.error(f"Error combining signals: {str(e)}")
            return self._neutral_signal(f"Signal combination error: {str(e)}")
    
    def _weighted_combination(self, signals: Dict, df: pd.DataFrame) -> Dict:
        """Combine signals using weighted average"""
        try:
            buy_score = 0.0
            sell_score = 0.0
            neutral_score = 0.0
            total_weight = 0.0
            
            entry_prices = []
            stop_losses = []
            take_profits = []
            reasons = []
            patterns = []
            
            for strategy_name, signal in signals.items():
                weight = self.weights.get(strategy_name, 1.0)
                strength = signal.get('strength', 0.0)
                
                if signal['signal'] == 'BUY':
                    buy_score += weight * strength
                    if 'entry_price' in signal:
                        entry_prices.append(signal['entry_price'])
                        stop_losses.append(signal.get('stop_loss', 0))
                        take_profits.append(signal.get('take_profit', 0))
                    
                elif signal['signal'] == 'SELL':
                    sell_score += weight * strength
                    if 'entry_price' in signal:
                        entry_prices.append(signal['entry_price'])
                        stop_losses.append(signal.get('stop_loss', 0))
                        take_profits.append(signal.get('take_profit', 0))
                    
                else:  # NEUTRAL
                    neutral_score += weight
                
                total_weight += weight
                
                # Collect reasons and patterns
                if 'reason' in signal:
                    reasons.append(f"{strategy_name}: {signal['reason']}")
                if 'patterns' in signal:
                    patterns.extend(signal['patterns'])
            
            # Normalize scores
            if total_weight > 0:
                buy_score /= total_weight
                sell_score /= total_weight
                neutral_score /= total_weight
            
            # Determine final signal
            if buy_score > sell_score and buy_score > 0.5:
                direction = 'BUY'
                strength = buy_score
            elif sell_score > buy_score and sell_score > 0.5:
                direction = 'SELL'
                strength = sell_score
            else:
                return self._neutral_signal("Combined signal strength too low")
            
            # Calculate appropriate levels based on signal direction and market data
            current_price = df['close'].iloc[-1]
            
            # Safe ATR calculation with NaN handling
            if 'atr' in df.columns:
                atr = df['atr'].iloc[-1]
                # Handle NaN or invalid ATR values
                if pd.isna(atr) or atr <= 0:
                    atr = (df['high'] - df['low']).iloc[-20:].mean()
                    if pd.isna(atr) or atr <= 0:
                        atr = current_price * 0.001  # Fallback to 0.1% of price
            else:
                atr = (df['high'] - df['low']).iloc[-20:].mean()
                if pd.isna(atr) or atr <= 0:
                    atr = current_price * 0.001  # Fallback to 0.1% of price
            
            if entry_prices:
                avg_entry = np.mean(entry_prices)
            else:
                avg_entry = current_price
            
            # Improved SL/TP calculation for better risk management
            if direction == 'BUY':
                # For BUY: SL below entry, TP above entry
                valid_stop_losses = [sl for sl in stop_losses if sl < avg_entry and sl > 0]
                valid_take_profits = [tp for tp in take_profits if tp > avg_entry and tp > 0]
                
                if valid_stop_losses:
                    avg_sl = max(valid_stop_losses)  # Closest to entry for better risk
                else:
                    avg_sl = avg_entry - atr * 2
                    
                if valid_take_profits:
                    avg_tp = min(valid_take_profits)  # Closest realistic target
                else:
                    avg_tp = avg_entry + atr * 3
                    
            else:  # SELL
                # For SELL: SL above entry, TP below entry
                valid_stop_losses = [sl for sl in stop_losses if sl > avg_entry and sl > 0]
                valid_take_profits = [tp for tp in take_profits if tp < avg_entry and tp > 0]
                
                if valid_stop_losses:
                    avg_sl = min(valid_stop_losses)  # Closest to entry for better risk
                else:
                    avg_sl = avg_entry + atr * 2
                    
                if valid_take_profits:
                    avg_tp = max(valid_take_profits)  # Closest realistic target
                else:
                    avg_tp = avg_entry - atr * 3
            
            # Validate risk/reward ratio
            risk = abs(avg_entry - avg_sl)
            reward = abs(avg_tp - avg_entry)
            rr_ratio = reward / risk if risk > 0 else 0
            
            if rr_ratio < 1.5:
                # Adjust TP to maintain minimum 1.5:1 RR
                if direction == 'BUY':
                    avg_tp = avg_entry + (risk * 1.5)
                else:
                    avg_tp = avg_entry - (risk * 1.5)
            
            return {
                'signal': direction,
                'strength': strength,
                'entry_price': avg_entry,
                'stop_loss': avg_sl,
                'take_profit': avg_tp,
                'confidence': strength,
                'combination_mode': 'weighted',
                'reasons': reasons,
                'patterns': list(set(patterns)),
                'scores': {
                    'buy': buy_score,
                    'sell': sell_score,
                    'neutral': neutral_score
                }
            }
            
        except Exception as e:
            logger.error(f"Error in weighted combination: {str(e)}")
            return self._neutral_signal(f"Weighted combination error: {str(e)}")
    
    def _majority_combination(self, signals: Dict, df: pd.DataFrame) -> Dict:
        """Combine signals using majority vote"""
        try:
            votes = {'BUY': 0, 'SELL': 0, 'NEUTRAL': 0}
            
            entry_prices = []
            stop_losses = []
            take_profits = []
            reasons = []
            patterns = []
            
            for strategy_name, signal in signals.items():
                votes[signal['signal']] += 1
                
                if signal['signal'] in ['BUY', 'SELL'] and 'entry_price' in signal:
                    entry_prices.append(signal['entry_price'])
                    stop_losses.append(signal.get('stop_loss', 0))
                    take_profits.append(signal.get('take_profit', 0))
                
                if 'reason' in signal:
                    reasons.append(f"{strategy_name}: {signal['reason']}")
                if 'patterns' in signal:
                    patterns.extend(signal['patterns'])
            
            # Determine majority
            total_votes = len(signals)
            majority_threshold = total_votes / 2
            
            if votes['BUY'] > majority_threshold:
                direction = 'BUY'
                confidence = votes['BUY'] / total_votes
            elif votes['SELL'] > majority_threshold:
                direction = 'SELL'
                confidence = votes['SELL'] / total_votes
            else:
                return self._neutral_signal("No clear majority signal")
            
            # Calculate levels
            current_price = df['close'].iloc[-1]
            
            if entry_prices:
                avg_entry = np.mean(entry_prices)
            else:
                avg_entry = current_price
            
            if direction == 'BUY':
                avg_sl = min(stop_losses) if stop_losses else current_price - df['atr'].iloc[-1] * 2
                avg_tp = max(take_profits) if take_profits else current_price + df['atr'].iloc[-1] * 3
            else:
                avg_sl = max(stop_losses) if stop_losses else current_price + df['atr'].iloc[-1] * 2
                avg_tp = min(take_profits) if take_profits else current_price - df['atr'].iloc[-1] * 3
            
            return {
                'signal': direction,
                'strength': confidence,
                'entry_price': avg_entry,
                'stop_loss': avg_sl,
                'take_profit': avg_tp,
                'confidence': confidence,
                'combination_mode': 'majority',
                'reasons': reasons,
                'patterns': list(set(patterns)),
                'votes': votes
            }
            
        except Exception as e:
            logger.error(f"Error in majority combination: {str(e)}")
            return self._neutral_signal(f"Majority combination error: {str(e)}")
    
    def _unanimous_combination(self, signals: Dict, df: pd.DataFrame) -> Dict:
        """Combine signals requiring unanimous agreement"""
        try:
            # Check if all signals agree
            signal_types = [s['signal'] for s in signals.values()]
            
            if len(set(signal_types)) != 1:
                return self._neutral_signal("Strategies do not agree unanimously")
            
            unanimous_signal = signal_types[0]
            
            if unanimous_signal == 'NEUTRAL':
                return self._neutral_signal("All strategies neutral")
            
            # Collect data from all signals
            entry_prices = []
            stop_losses = []
            take_profits = []
            strengths = []
            reasons = []
            patterns = []
            
            for strategy_name, signal in signals.items():
                if 'entry_price' in signal:
                    entry_prices.append(signal['entry_price'])
                    stop_losses.append(signal.get('stop_loss', 0))
                    take_profits.append(signal.get('take_profit', 0))
                
                strengths.append(signal.get('strength', 0.5))
                
                if 'reason' in signal:
                    reasons.append(f"{strategy_name}: {signal['reason']}")
                if 'patterns' in signal:
                    patterns.extend(signal['patterns'])
            
            # Calculate averages
            avg_strength = np.mean(strengths)
            current_price = df['close'].iloc[-1]
            
            if entry_prices:
                avg_entry = np.mean(entry_prices)
            else:
                avg_entry = current_price
            
            if unanimous_signal == 'BUY':
                avg_sl = min(stop_losses) if stop_losses else current_price - df['atr'].iloc[-1] * 2
                avg_tp = max(take_profits) if take_profits else current_price + df['atr'].iloc[-1] * 3
            else:
                avg_sl = max(stop_losses) if stop_losses else current_price + df['atr'].iloc[-1] * 2
                avg_tp = min(take_profits) if take_profits else current_price - df['atr'].iloc[-1] * 3
            
            return {
                'signal': unanimous_signal,
                'strength': avg_strength,
                'entry_price': avg_entry,
                'stop_loss': avg_sl,
                'take_profit': avg_tp,
                'confidence': 0.9,  # High confidence due to unanimity
                'combination_mode': 'unanimous',
                'reasons': reasons,
                'patterns': list(set(patterns))
            }
            
        except Exception as e:
            logger.error(f"Error in unanimous combination: {str(e)}")
            return self._neutral_signal(f"Unanimous combination error: {str(e)}")
    
    def update_strategy(self, strategy_name: str, parameters: Dict) -> Dict:
        """Update strategy parameters"""
        try:
            if strategy_name not in self.strategies:
                raise ValueError(f"Unknown strategy: {strategy_name}")
            
            # Update parameters
            self.strategies[strategy_name].update_parameters(parameters)
            
            # Save to config
            self._save_config()
            
            return {
                'strategy': strategy_name,
                'parameters': parameters,
                'status': 'updated'
            }
            
        except Exception as e:
            logger.error(f"Error updating strategy: {str(e)}")
            raise
    
    def set_strategy_weight(self, strategy_name: str, weight: float) -> None:
        """Set strategy weight for combination"""
        if strategy_name in self.strategies:
            self.weights[strategy_name] = max(0.0, min(10.0, weight))
            self._save_config()
            logger.info(f"Set {strategy_name} weight to {weight}")
    
    def enable_strategy(self, strategy_name: str) -> None:
        """Enable a strategy"""
        if strategy_name in self.strategies and strategy_name not in self.enabled_strategies:
            self.enabled_strategies.append(strategy_name)
            self._save_config()
            logger.info(f"Enabled strategy: {strategy_name}")
    
    def disable_strategy(self, strategy_name: str) -> None:
        """Disable a strategy"""
        if strategy_name in self.enabled_strategies:
            self.enabled_strategies.remove(strategy_name)
            self._save_config()
            logger.info(f"Disabled strategy: {strategy_name}")
    
    def set_combination_mode(self, mode: str) -> None:
        """Set signal combination mode"""
        valid_modes = ['weighted', 'majority', 'unanimous']
        if mode in valid_modes:
            self.combination_mode = mode
            self._save_config()
            logger.info(f"Set combination mode to: {mode}")
    
    def update_performance(self, strategy_name: str, trade_result: Dict) -> None:
        """Update strategy performance metrics"""
        try:
            if strategy_name not in self.performance:
                return
            
            perf = self.performance[strategy_name]
            perf['trades'] += 1
            
            if trade_result['profit'] > 0:
                perf['wins'] += 1
            else:
                perf['losses'] += 1
            
            perf['total_pips'] += trade_result.get('pips', 0)
            perf['win_rate'] = perf['wins'] / perf['trades'] if perf['trades'] > 0 else 0
            
            logger.info(f"Updated {strategy_name} performance: "
                      f"Win rate: {perf['win_rate']:.2%}, Total pips: {perf['total_pips']:.1f}")
            
        except Exception as e:
            logger.error(f"Error updating performance: {str(e)}")
    
    def get_performance_report(self) -> Dict:
        """Get performance report for all strategies"""
        return {
            'individual_performance': self.performance,
            'enabled_strategies': self.enabled_strategies,
            'combination_mode': self.combination_mode,
            'weights': self.weights
        }
    
    def _save_config(self) -> None:
        """Save current configuration to file"""
        try:
            config = {
                'enabled': self.enabled_strategies,
                'weights': self.weights,
                'combination_mode': self.combination_mode,
                'parameters': {
                    name: strategy.params 
                    for name, strategy in self.strategies.items()
                }
            }
            
            with open('config/strategies.json', 'w') as f:
                json.dump(config, f, indent=4)
                
        except Exception as e:
            logger.error(f"Error saving config: {str(e)}")
    
    def _neutral_signal(self, reason: str) -> Dict:
        """Generate neutral signal"""
        return {
            'signal': 'NEUTRAL',
            'strength': 0.0,
            'confidence': 0.0,
            'reason': reason
        }