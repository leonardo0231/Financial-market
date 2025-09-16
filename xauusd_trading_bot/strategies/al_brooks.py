import logging
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class AlBrooksStrategy(BaseStrategy):
    """Al Brooks Price Action Strategy Implementation"""
    
    def __init__(self):
        """Initialize Al Brooks strategy"""
        super().__init__(name="Al Brooks Price Action")
        
        # Strategy parameters
        self.params = {
            'min_trend_strength': 0.6,
            'pullback_depth': 0.5,
            'breakout_confirmation': 2,
            'bar_analysis_window': 20,
            'trend_bar_threshold': 0.7,
            'risk_reward_ratio': 2.0
        }
        
        # Pattern tracking
        self.patterns_found = []
    
    def get_minimum_bars(self) -> int:
        """Al Brooks strategy needs at least 50 bars"""
        return 50
        
    def analyze(self, df: pd.DataFrame) -> Dict:
        """Analyze market using Al Brooks concepts"""
        try:
            if len(df) < 50:
                return self._neutral_signal("Insufficient data")
            
            # Reset patterns
            self.patterns_found = []
            
            # Analyze market structure
            market_analysis = self._analyze_market_structure(df)
            
            # Identify price action patterns
            patterns = self._identify_patterns(df)
            
            # Analyze individual bars
            bar_analysis = self._analyze_bars(df)
            
            # Find entry setups
            setups = self._find_setups(df, market_analysis, patterns)
            
            # Generate signal
            signal = self._generate_signal(setups, market_analysis, bar_analysis)
            
            return signal
            
        except Exception as e:
            logger.error(f"Error in Al Brooks analysis: {str(e)}")
            return self._neutral_signal(f"Analysis error: {str(e)}")
    
    def _analyze_market_structure(self, df: pd.DataFrame) -> Dict:
        """Analyze overall market structure"""
        try:
            analysis = {
                'trend_type': 'NEUTRAL',
                'trend_strength': 0.0,
                'in_trading_range': False,
                'breakout_mode': False,
                'pullback_active': False
            }
            
            # Trend analysis using swing points
            swing_highs = df[df['swing_high'] == 1]['high'].values
            swing_lows = df[df['swing_low'] == 1]['low'].values
            
            if len(swing_highs) >= 2 and len(swing_lows) >= 2:
                # Check for higher highs and higher lows (uptrend)
                if swing_highs[-1] > swing_highs[-2] and swing_lows[-1] > swing_lows[-2]:
                    analysis['trend_type'] = 'UPTREND'
                    analysis['trend_strength'] = self._calculate_trend_strength(df, 'UP')
                
                # Check for lower highs and lower lows (downtrend)
                elif swing_highs[-1] < swing_highs[-2] and swing_lows[-1] < swing_lows[-2]:
                    analysis['trend_type'] = 'DOWNTREND'
                    analysis['trend_strength'] = self._calculate_trend_strength(df, 'DOWN')
                
                # Trading range detection
                else:
                    recent_high = df['high'].iloc[-20:].max()
                    recent_low = df['low'].iloc[-20:].min()
                    range_size = recent_high - recent_low
                    
                    if range_size < df['atr'].iloc[-1] * 3:
                        analysis['in_trading_range'] = True
            
            # Breakout mode detection
            if df['close'].iloc[-1] > df['high'].iloc[-20:-1].max():
                analysis['breakout_mode'] = True
                analysis['breakout_direction'] = 'UP'
            elif df['close'].iloc[-1] < df['low'].iloc[-20:-1].min():
                analysis['breakout_mode'] = True
                analysis['breakout_direction'] = 'DOWN'
            
            # Pullback detection with ATR zero protection
            atr_current = df['atr'].iloc[-1]
            if atr_current > 0:  # Avoid division by zero
                if analysis['trend_type'] == 'UPTREND':
                    recent_high = df['high'].iloc[-10:].max()
                    pullback_depth = (recent_high - df['close'].iloc[-1]) / atr_current
                    if 0.5 < pullback_depth < 2.0:
                        analysis['pullback_active'] = True
                        
                elif analysis['trend_type'] == 'DOWNTREND':
                    recent_low = df['low'].iloc[-10:].min()
                    pullback_depth = (df['close'].iloc[-1] - recent_low) / atr_current
                    if 0.5 < pullback_depth < 2.0:
                        analysis['pullback_active'] = True
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing market structure: {str(e)}")
            return {'trend_type': 'NEUTRAL', 'trend_strength': 0.0}
    
    def _identify_patterns(self, df: pd.DataFrame) -> Dict:
        """Identify Al Brooks price action patterns"""
        patterns = {
            'two_bar_reversal': False,
            'failed_breakout': False,
            'wedge': False,
            'channel': False,
            'flag': False,
            'ii_pattern': False,  # Inside-inside pattern
            'breakout_pullback': False
        }
        
        try:
            # Two-bar reversal
            if self._is_two_bar_reversal(df):
                patterns['two_bar_reversal'] = True
                self.patterns_found.append('Two-Bar Reversal')
            
            # Failed breakout
            if self._is_failed_breakout(df):
                patterns['failed_breakout'] = True
                self.patterns_found.append('Failed Breakout')
            
            # Wedge pattern
            if self._is_wedge_pattern(df):
                patterns['wedge'] = True
                self.patterns_found.append('Wedge')
            
            # Channel
            if self._is_channel(df):
                patterns['channel'] = True
                self.patterns_found.append('Channel')
            
            # Flag pattern
            if self._is_flag_pattern(df):
                patterns['flag'] = True
                self.patterns_found.append('Flag')
            
            # Inside-inside pattern
            if self._is_ii_pattern(df):
                patterns['ii_pattern'] = True
                self.patterns_found.append('II Pattern')
            
            # Breakout pullback
            if self._is_breakout_pullback(df):
                patterns['breakout_pullback'] = True
                self.patterns_found.append('Breakout Pullback')
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error identifying patterns: {str(e)}")
            return patterns
    
    def _analyze_bars(self, df: pd.DataFrame) -> Dict:
        """Analyze individual bars for Al Brooks concepts"""
        try:
            last_bars = df.iloc[-5:]
            
            analysis = {
                'trend_bars': 0,
                'reversal_bars': 0,
                'doji_bars': 0,
                'strong_bars': 0,
                'weak_bars': 0,
                'bull_pressure': 0.0,
                'bear_pressure': 0.0
            }
            
            for idx, bar in last_bars.iterrows():
                # Trend bar detection
                if self._is_trend_bar(bar):
                    analysis['trend_bars'] += 1
                
                # Reversal bar detection
                if self._is_reversal_bar(bar, df.loc[:idx]):
                    analysis['reversal_bars'] += 1
                
                # Doji detection
                if abs(bar['close'] - bar['open']) < bar['atr'] * 0.1:
                    analysis['doji_bars'] += 1
                
                # Strong vs weak bars
                if bar['body_size'] > bar['atr'] * 0.7:
                    if bar['close'] > bar['open']:
                        analysis['strong_bars'] += 1
                        analysis['bull_pressure'] += bar['body_size'] / bar['atr']
                    else:
                        analysis['strong_bars'] += 1
                        analysis['bear_pressure'] += bar['body_size'] / bar['atr']
                else:
                    analysis['weak_bars'] += 1
            
            # Normalize pressures
            analysis['bull_pressure'] /= len(last_bars)
            analysis['bear_pressure'] /= len(last_bars)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing bars: {str(e)}")
            return {}
    
    def _find_setups(self, df: pd.DataFrame, market_analysis: Dict, patterns: Dict) -> List[Dict]:
        """Find Al Brooks trading setups"""
        setups = []
        
        try:
            # Major trend continuation setup
            if market_analysis['trend_type'] == 'UPTREND' and market_analysis['pullback_active']:
                setup = self._check_trend_continuation_setup(df, 'LONG')
                if setup:
                    setups.append(setup)
            
            elif market_analysis['trend_type'] == 'DOWNTREND' and market_analysis['pullback_active']:
                setup = self._check_trend_continuation_setup(df, 'SHORT')
                if setup:
                    setups.append(setup)
            
            # Failed breakout setup
            if patterns['failed_breakout']:
                setup = self._check_failed_breakout_setup(df)
                if setup:
                    setups.append(setup)
            
            # Trading range setup
            if market_analysis['in_trading_range']:
                setup = self._check_trading_range_setup(df)
                if setup:
                    setups.append(setup)
            
            # Breakout pullback setup
            if patterns['breakout_pullback']:
                setup = self._check_breakout_pullback_setup(df)
                if setup:
                    setups.append(setup)
            
            # Two-bar reversal setup
            if patterns['two_bar_reversal']:
                setup = self._check_two_bar_reversal_setup(df)
                if setup:
                    setups.append(setup)
            
            return setups
            
        except Exception as e:
            logger.error(f"Error finding setups: {str(e)}")
            return []
    
    def _generate_signal(self, setups: List[Dict], market_analysis: Dict, bar_analysis: Dict) -> Dict:
        """Generate trading signal from analysis"""
        try:
            if not setups:
                return self._neutral_signal("No valid setups found")
            
            # Score each setup
            best_setup = None
            best_score = 0
            
            for setup in setups:
                score = self._score_setup(setup, market_analysis, bar_analysis)
                if score > best_score:
                    best_score = score
                    best_setup = setup
            
            # Generate signal from best setup
            if best_setup and best_score > 0.6:
                return {
                    'signal': best_setup['direction'],
                    'strength': best_score,
                    'entry_price': best_setup['entry_price'],
                    'stop_loss': best_setup['stop_loss'],
                    'take_profit': best_setup['take_profit'],
                    'setup_type': best_setup['type'],
                    'confidence': best_score,
                    'patterns': self.patterns_found,
                    'market_analysis': market_analysis,
                    'bar_analysis': bar_analysis,
                    'reason': best_setup['reason']
                }
            
            return self._neutral_signal("Setup score too low")
            
        except Exception as e:
            logger.error(f"Error generating signal: {str(e)}")
            return self._neutral_signal(f"Signal generation error: {str(e)}")
    
    # Helper methods
    def _calculate_trend_strength(self, df: pd.DataFrame, direction: str) -> float:
        """Calculate trend strength"""
        try:
            if direction == 'UP':
                # Measure consistency of higher highs and higher lows
                highs = df['high'].iloc[-20:]
                lows = df['low'].iloc[-20:]
                
                hh_count = sum(highs.diff() > 0)
                hl_count = sum(lows.diff() > 0)
                
                strength = (hh_count + hl_count) / (2 * len(highs))
                
            else:  # DOWN
                # Measure consistency of lower highs and lower lows
                highs = df['high'].iloc[-20:]
                lows = df['low'].iloc[-20:]
                
                lh_count = sum(highs.diff() < 0)
                ll_count = sum(lows.diff() < 0)
                
                strength = (lh_count + ll_count) / (2 * len(highs))
            
            # Factor in ADX
            if 'adx' in df.columns:
                adx_factor = min(df['adx'].iloc[-1] / 50, 1.0)
                strength = strength * 0.7 + adx_factor * 0.3
            
            return min(strength, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating trend strength: {str(e)}")
            return 0.0
    
    def _is_trend_bar(self, bar: pd.Series) -> bool:
        """Check if bar is a trend bar"""
        try:
            # Strong close near high/low
            if bar['close'] > bar['open']:  # Bull bar
                return bar['close_position'] > self.params['trend_bar_threshold']
            else:  # Bear bar
                return bar['close_position'] < (1 - self.params['trend_bar_threshold'])
                
        except Exception as e:
            logger.error(f"Error checking trend bar: {str(e)}")
            return False
    
    def _is_reversal_bar(self, bar: pd.Series, prev_bars: pd.DataFrame) -> bool:
        """Check if bar is a reversal bar"""
        try:
            if len(prev_bars) < 2:
                return False
            
            prev_bar = prev_bars.iloc[-2]
            
            # Bull reversal: bear bar followed by bull bar that closes above its high
            if prev_bar['close'] < prev_bar['open'] and bar['close'] > bar['open']:
                if bar['close'] > prev_bar['high']:
                    return True
            
            # Bear reversal: bull bar followed by bear bar that closes below its low
            if prev_bar['close'] > prev_bar['open'] and bar['close'] < bar['open']:
                if bar['close'] < prev_bar['low']:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking reversal bar: {str(e)}")
            return False
    
    def _is_two_bar_reversal(self, df: pd.DataFrame) -> bool:
        """Check for two-bar reversal pattern"""
        try:
            if len(df) < 3:
                return False
            
            last_bars = df.iloc[-3:]
            
            # Bull reversal
            if (last_bars.iloc[0]['close'] < last_bars.iloc[0]['open'] and
                last_bars.iloc[1]['close'] < last_bars.iloc[1]['open'] and
                last_bars.iloc[2]['close'] > last_bars.iloc[2]['open'] and
                last_bars.iloc[2]['close'] > last_bars.iloc[1]['high']):
                return True
            
            # Bear reversal
            if (last_bars.iloc[0]['close'] > last_bars.iloc[0]['open'] and
                last_bars.iloc[1]['close'] > last_bars.iloc[1]['open'] and
                last_bars.iloc[2]['close'] < last_bars.iloc[2]['open'] and
                last_bars.iloc[2]['close'] < last_bars.iloc[1]['low']):
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking two-bar reversal: {str(e)}")
            return False
    
    def _is_failed_breakout(self, df: pd.DataFrame) -> bool:
        """Check for failed breakout pattern"""
        try:
            if len(df) < 10:
                return False
            
            # Check recent high/low
            recent_high = df['high'].iloc[-10:-3].max()
            recent_low = df['low'].iloc[-10:-3].min()
            
            last_bars = df.iloc[-3:]
            
            # Failed breakout above resistance
            if (last_bars.iloc[0]['high'] > recent_high and
                last_bars.iloc[2]['close'] < recent_high):
                return True
            
            # Failed breakout below support
            if (last_bars.iloc[0]['low'] < recent_low and
                last_bars.iloc[2]['close'] > recent_low):
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking failed breakout: {str(e)}")
            return False
    
    def _is_wedge_pattern(self, df: pd.DataFrame) -> bool:
        """Check for wedge pattern"""
        try:
            if len(df) < 20:
                return False
            
            # Get recent swing points
            recent_data = df.iloc[-20:]
            highs = recent_data[recent_data['swing_high'] == 1]['high']
            lows = recent_data[recent_data['swing_low'] == 1]['low']
            
            if len(highs) < 2 or len(lows) < 2:
                return False
            
            # Check for converging trendlines
            high_slope = (highs.iloc[-1] - highs.iloc[0]) / len(highs)
            low_slope = (lows.iloc[-1] - lows.iloc[0]) / len(lows)
            
            # Rising wedge: both slopes positive but high slope < low slope
            if high_slope > 0 and low_slope > 0 and high_slope < low_slope:
                return True
            
            # Falling wedge: both slopes negative but high slope > low slope
            if high_slope < 0 and low_slope < 0 and high_slope > low_slope:
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking wedge pattern: {str(e)}")
            return False
    
    def _is_channel(self, df: pd.DataFrame) -> bool:
        """Check for channel pattern"""
        try:
            if len(df) < 20:
                return False
            
            recent_data = df.iloc[-20:]
            
            # Simple channel detection using parallel lines
            highs = recent_data['high'].values
            lows = recent_data['low'].values
            
            # Calculate slopes
            x = np.arange(len(highs))
            high_slope = np.polyfit(x, highs, 1)[0]
            low_slope = np.polyfit(x, lows, 1)[0]
            
            # Check if slopes are similar (parallel)
            slope_diff = abs(high_slope - low_slope)
            avg_slope = abs(high_slope + low_slope) / 2
            
            if avg_slope > 0 and slope_diff / avg_slope < 0.2:
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking channel: {str(e)}")
            return False
    
    def _is_flag_pattern(self, df: pd.DataFrame) -> bool:
        """Check for flag pattern"""
        try:
            if len(df) < 15:
                return False
            
            # Look for strong move followed by consolidation
            initial_move = df.iloc[-15:-10]
            consolidation = df.iloc[-10:]
            
            # Calculate initial move strength
            move_size = abs(initial_move['close'].iloc[-1] - initial_move['close'].iloc[0])
            move_strength = move_size / initial_move['atr'].mean()
            
            # Check consolidation
            consol_range = consolidation['high'].max() - consolidation['low'].min()
            consol_ratio = consol_range / move_size
            
            # Flag criteria: strong move followed by tight consolidation
            if move_strength > 2.0 and consol_ratio < 0.5:
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking flag pattern: {str(e)}")
            return False
    
    def _is_ii_pattern(self, df: pd.DataFrame) -> bool:
        """Check for inside-inside pattern"""
        try:
            if len(df) < 3:
                return False
            
            last_bars = df.iloc[-3:]
            
            # Check if last two bars are inside bars
            if (last_bars.iloc[1]['high'] < last_bars.iloc[0]['high'] and
                last_bars.iloc[1]['low'] > last_bars.iloc[0]['low'] and
                last_bars.iloc[2]['high'] < last_bars.iloc[1]['high'] and
                last_bars.iloc[2]['low'] > last_bars.iloc[1]['low']):
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking II pattern: {str(e)}")
            return False
    
    def _is_breakout_pullback(self, df: pd.DataFrame) -> bool:
        """Check for breakout pullback pattern"""
        try:
            if len(df) < 10:
                return False
            
            # Find recent breakout
            lookback = df.iloc[-10:]
            prev_high = df['high'].iloc[-20:-10].max()
            prev_low = df['low'].iloc[-20:-10].min()
            
            # Check for breakout and pullback
            breakout_bar = None
            for i in range(len(lookback) - 3):
                if lookback.iloc[i]['close'] > prev_high:
                    breakout_bar = i
                    break
                elif lookback.iloc[i]['close'] < prev_low:
                    breakout_bar = i
                    break
            
            if breakout_bar is not None:
                # Check for pullback after breakout
                if breakout_bar < len(lookback) - 3:
                    current_price = lookback.iloc[-1]['close']
                    breakout_price = lookback.iloc[breakout_bar]['close']
                    
                    if lookback.iloc[breakout_bar]['close'] > prev_high:
                        # Bullish breakout pullback
                        if prev_high < current_price < breakout_price:
                            return True
                    else:
                        # Bearish breakout pullback
                        if breakout_price < current_price < prev_low:
                            return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking breakout pullback: {str(e)}")
            return False
    
    def _check_trend_continuation_setup(self, df: pd.DataFrame, direction: str) -> Optional[Dict]:
        """Check for trend continuation setup"""
        try:
            current_price = df['close'].iloc[-1]
            atr = df['atr'].iloc[-1]
            
            if direction == 'LONG':
                # Look for pullback to support in uptrend
                support = df['sma_20'].iloc[-1]
                
                if abs(current_price - support) < atr * 0.5:
                    # Check for bullish bar at support
                    if df['close'].iloc[-1] > df['open'].iloc[-1]:
                        return {
                            'type': 'Trend Continuation Long',
                            'direction': 'BUY',
                            'entry_price': current_price,
                            'stop_loss': support - atr,
                            'take_profit': current_price + atr * self.params['risk_reward_ratio'],
                            'reason': 'Pullback to support in uptrend with bullish confirmation'
                        }
            
            else:  # SHORT
                # Look for pullback to resistance in downtrend
                resistance = df['sma_20'].iloc[-1]
                
                if abs(current_price - resistance) < atr * 0.5:
                    # Check for bearish bar at resistance
                    if df['close'].iloc[-1] < df['open'].iloc[-1]:
                        return {
                            'type': 'Trend Continuation Short',
                            'direction': 'SELL',
                            'entry_price': current_price,
                            'stop_loss': resistance + atr,
                            'take_profit': current_price - atr * self.params['risk_reward_ratio'],
                            'reason': 'Pullback to resistance in downtrend with bearish confirmation'
                        }
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking trend continuation: {str(e)}")
            return None
    
    def _check_failed_breakout_setup(self, df: pd.DataFrame) -> Optional[Dict]:
        """Check for failed breakout setup"""
        try:
            current_price = df['close'].iloc[-1]
            atr = df['atr'].iloc[-1]
            
            # Recent levels
            recent_high = df['high'].iloc[-10:-3].max()
            recent_low = df['low'].iloc[-10:-3].min()
            
            # Failed breakout above - short setup
            if df['high'].iloc[-3] > recent_high and current_price < recent_high:
                return {
                    'type': 'Failed Breakout Short',
                    'direction': 'SELL',
                    'entry_price': current_price,
                    'stop_loss': df['high'].iloc[-3] + atr * 0.5,
                    'take_profit': current_price - atr * self.params['risk_reward_ratio'],
                    'reason': 'Failed breakout above resistance'
                }
            
            # Failed breakout below - long setup
            if df['low'].iloc[-3] < recent_low and current_price > recent_low:
                return {
                    'type': 'Failed Breakout Long',
                    'direction': 'BUY',
                    'entry_price': current_price,
                    'stop_loss': df['low'].iloc[-3] - atr * 0.5,
                    'take_profit': current_price + atr * self.params['risk_reward_ratio'],
                    'reason': 'Failed breakout below support'
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking failed breakout setup: {str(e)}")
            return None
    
    def _check_trading_range_setup(self, df: pd.DataFrame) -> Optional[Dict]:
        """Check for trading range setup"""
        try:
            current_price = df['close'].iloc[-1]
            atr = df['atr'].iloc[-1]
            
            # Define range
            range_high = df['high'].iloc[-20:].max()
            range_low = df['low'].iloc[-20:].min()
            range_size = range_high - range_low
            
            # Check position in range
            position_in_range = (current_price - range_low) / range_size
            
            # Buy at range low
            if position_in_range < 0.2 and df['close'].iloc[-1] > df['open'].iloc[-1]:
                return {
                    'type': 'Range Low Buy',
                    'direction': 'BUY',
                    'entry_price': current_price,
                    'stop_loss': range_low - atr * 0.5,
                    'take_profit': range_low + range_size * 0.8,
                    'reason': 'Buy at trading range support'
                }
            
            # Sell at range high
            if position_in_range > 0.8 and df['close'].iloc[-1] < df['open'].iloc[-1]:
                return {
                    'type': 'Range High Sell',
                    'direction': 'SELL',
                    'entry_price': current_price,
                    'stop_loss': range_high + atr * 0.5,
                    'take_profit': range_high - range_size * 0.8,
                    'reason': 'Sell at trading range resistance'
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking trading range setup: {str(e)}")
            return None
    
    def _check_breakout_pullback_setup(self, df: pd.DataFrame) -> Optional[Dict]:
        """Check for breakout pullback setup"""
        try:
            current_price = df['close'].iloc[-1]
            atr = df['atr'].iloc[-1]
            
            # Find breakout level
            prev_high = df['high'].iloc[-20:-10].max()
            prev_low = df['low'].iloc[-20:-10].min()
            
            # Bullish breakout pullback
            if current_price > prev_high and df['low'].iloc[-3:].min() <= prev_high:
                if df['close'].iloc[-1] > df['open'].iloc[-1]:
                    return {
                        'type': 'Breakout Pullback Long',
                        'direction': 'BUY',
                        'entry_price': current_price,
                        'stop_loss': prev_high - atr * 0.5,
                        'take_profit': current_price + atr * self.params['risk_reward_ratio'],
                        'reason': 'Pullback to previous resistance turned support'
                    }
            
            # Bearish breakout pullback
            if current_price < prev_low and df['high'].iloc[-3:].max() >= prev_low:
                if df['close'].iloc[-1] < df['open'].iloc[-1]:
                    return {
                        'type': 'Breakout Pullback Short',
                        'direction': 'SELL',
                        'entry_price': current_price,
                        'stop_loss': prev_low + atr * 0.5,
                        'take_profit': current_price - atr * self.params['risk_reward_ratio'],
                        'reason': 'Pullback to previous support turned resistance'
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking breakout pullback setup: {str(e)}")
            return None
    
    def _check_two_bar_reversal_setup(self, df: pd.DataFrame) -> Optional[Dict]:
        """Check for two-bar reversal setup"""
        try:
            current_price = df['close'].iloc[-1]
            atr = df['atr'].iloc[-1]
            
            last_bars = df.iloc[-3:]
            
            # Bullish reversal
            if (last_bars.iloc[0]['close'] < last_bars.iloc[0]['open'] and
                last_bars.iloc[1]['close'] < last_bars.iloc[1]['open'] and
                last_bars.iloc[2]['close'] > last_bars.iloc[2]['open'] and
                last_bars.iloc[2]['close'] > last_bars.iloc[1]['high']):
                
                return {
                    'type': 'Two-Bar Bullish Reversal',
                    'direction': 'BUY',
                    'entry_price': current_price,
                    'stop_loss': last_bars.iloc[1]['low'] - atr * 0.2,
                    'take_profit': current_price + atr * self.params['risk_reward_ratio'],
                    'reason': 'Two-bar bullish reversal pattern'
                }
            
            # Bearish reversal
            if (last_bars.iloc[0]['close'] > last_bars.iloc[0]['open'] and
                last_bars.iloc[1]['close'] > last_bars.iloc[1]['open'] and
                last_bars.iloc[2]['close'] < last_bars.iloc[2]['open'] and
                last_bars.iloc[2]['close'] < last_bars.iloc[1]['low']):
                
                return {
                    'type': 'Two-Bar Bearish Reversal',
                    'direction': 'SELL',
                    'entry_price': current_price,
                    'stop_loss': last_bars.iloc[1]['high'] + atr * 0.2,
                    'take_profit': current_price - atr * self.params['risk_reward_ratio'],
                    'reason': 'Two-bar bearish reversal pattern'
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking two-bar reversal setup: {str(e)}")
            return None
    
    def _score_setup(self, setup: Dict, market_analysis: Dict, bar_analysis: Dict) -> float:
        """Score a setup based on multiple factors"""
        try:
            score = 0.5  # Base score
            
            # Market alignment
            if setup['direction'] == 'BUY' and market_analysis['trend_type'] == 'UPTREND':
                score += 0.2
            elif setup['direction'] == 'SELL' and market_analysis['trend_type'] == 'DOWNTREND':
                score += 0.2
            
            # Trend strength
            score += market_analysis['trend_strength'] * 0.1
            
            # Bar analysis alignment
            if setup['direction'] == 'BUY' and bar_analysis['bull_pressure'] > bar_analysis['bear_pressure']:
                score += 0.1
            elif setup['direction'] == 'SELL' and bar_analysis['bear_pressure'] > bar_analysis['bull_pressure']:
                score += 0.1
            
            # Pattern quality
            if len(self.patterns_found) > 1:
                score += 0.05 * min(len(self.patterns_found), 3)
            
            # Volatility consideration
            if market_analysis.get('volatility_regime') == 'NORMAL':
                score += 0.05
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.error(f"Error scoring setup: {str(e)}")
            return 0.0
    
    def _neutral_signal(self, reason: str) -> Dict:
        """Generate neutral signal"""
        return {
            'signal': 'NEUTRAL',
            'strength': 0.0,
            'confidence': 0.0,
            'patterns': self.patterns_found,
            'reason': reason
        }
    
    def update_parameters(self, params: Dict) -> None:
        """Update strategy parameters"""
        self.params.update(params)
        logger.info(f"Updated Al Brooks strategy parameters: {params}")