import logging
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class LindaRaschkeStrategy(BaseStrategy):
    """Linda Raschke Pattern Trading Strategy Implementation"""
    
    def __init__(self):
        """Initialize Linda Raschke strategy"""
        super().__init__(name="Linda Raschke Patterns")
        
        # Strategy parameters
        self.params = {
            'turtle_soup_lookback': 20,
            'holy_grail_adx_threshold': 30,
            'holy_grail_adx_retracement': 20,
            '80_20_threshold': 0.8,
            'momentum_period': 10,
            'volatility_expansion_factor': 1.5,
            'keltner_period': 20,
            'keltner_multiplier': 2.0
        }
    
    def get_minimum_bars(self) -> int:
        """Linda Raschke strategy needs at least 50 bars"""
        return 50
        
    def analyze(self, df: pd.DataFrame) -> Dict:
        """Analyze market using Linda Raschke patterns"""
        try:
            if len(df) < 50:
                return self._neutral_signal("Insufficient data")
            
            # Reset patterns
            self.patterns_found = []
            self.setups = []
            
            # Add Keltner Channels
            df = self._add_keltner_channels(df)
            
            # Find patterns
            patterns = self._identify_patterns(df)
            
            # Analyze momentum
            momentum_analysis = self._analyze_momentum(df)
            
            # Check for setups
            self._check_turtle_soup(df, patterns)
            self._check_holy_grail(df, patterns)
            self._check_80_20(df, patterns)
            self._check_momentum_pinball(df, patterns)
            self._check_three_bar_triangle(df, patterns)
            
            # Generate signal
            signal = self._generate_signal(momentum_analysis)
            
            return signal
            
        except Exception as e:
            logger.error(f"Error in Linda Raschke analysis: {str(e)}")
            return self._neutral_signal(f"Analysis error: {str(e)}")
    
    def _add_keltner_channels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Keltner Channels to dataframe"""
        try:
            period = self.params['keltner_period']
            multiplier = self.params['keltner_multiplier']
            
            # Calculate typical price
            df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
            
            # EMA of typical price
            df['keltner_middle'] = df['typical_price'].ewm(span=period, adjust=False).mean()
            
            # ATR for channel width
            df['keltner_upper'] = df['keltner_middle'] + (df['atr'] * multiplier)
            df['keltner_lower'] = df['keltner_middle'] - (df['atr'] * multiplier)
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding Keltner channels: {str(e)}")
            return df
    
    def _identify_patterns(self, df: pd.DataFrame) -> Dict:
        """Identify Linda Raschke patterns in the data"""
        patterns = {
            'turtle_soup': [],
            'holy_grail': [],
            '80_20': [],
            'momentum_pinball': [],
            'three_bar_triangle': [],
            'first_cross': [],
            'anti_pattern': []
        }
        
        try:
            # Look for patterns in recent bars
            lookback = min(len(df), 50)
            
            for i in range(lookback, len(df)):
                window = df.iloc[i-lookback:i+1]
                
                # Check each pattern
                if self._is_turtle_soup_pattern(window, i):
                    patterns['turtle_soup'].append(i)
                
                if self._is_holy_grail_pattern(window, i):
                    patterns['holy_grail'].append(i)
                
                if self._is_80_20_pattern(window, i):
                    patterns['80_20'].append(i)
                
                if self._is_momentum_pinball_pattern(window, i):
                    patterns['momentum_pinball'].append(i)
                
                if self._is_three_bar_triangle_pattern(window, i):
                    patterns['three_bar_triangle'].append(i)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error identifying patterns: {str(e)}")
            return patterns
    
    def _analyze_momentum(self, df: pd.DataFrame) -> Dict:
        """Analyze market momentum"""
        try:
            period = self.params['momentum_period']
            
            # Rate of change
            df['roc'] = ((df['close'] - df['close'].shift(period)) / df['close'].shift(period)) * 100
            
            # Momentum oscillator
            df['momentum'] = df['close'] - df['close'].shift(period)
            
            # 2-day momentum
            df['momentum_2d'] = df['close'] - df['close'].shift(2)
            
            # Current momentum state
            current_roc = df['roc'].iloc[-1]
            current_momentum = df['momentum'].iloc[-1]
            momentum_2d = df['momentum_2d'].iloc[-1]
            
            # Momentum trend
            momentum_increasing = df['momentum'].iloc[-5:].diff().mean() > 0
            
            analysis = {
                'current_roc': current_roc,
                'current_momentum': current_momentum,
                'momentum_2d': momentum_2d,
                'momentum_increasing': momentum_increasing,
                'momentum_state': self._classify_momentum_state(current_roc, current_momentum)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing momentum: {str(e)}")
            return {}
    
    def _is_turtle_soup_pattern(self, window: pd.DataFrame, idx: int) -> bool:
        """Check for Turtle Soup pattern (failed breakout)"""
        try:
            if len(window) < self.params['turtle_soup_lookback']:
                return False
            
            lookback = self.params['turtle_soup_lookback']
            current = window.iloc[-1]
            
            # Get N-day high/low
            period_high = window['high'].iloc[-lookback-1:-1].max()
            period_low = window['low'].iloc[-lookback-1:-1].min()
            
            # Check for failed breakout above
            if (window['high'].iloc[-2] > period_high and 
                current['close'] < period_high and
                current['close'] < current['open']):
                self.patterns_found.append(f"Turtle Soup Short at {idx}")
                return True
            
            # Check for failed breakout below
            if (window['low'].iloc[-2] < period_low and 
                current['close'] > period_low and
                current['close'] > current['open']):
                self.patterns_found.append(f"Turtle Soup Long at {idx}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking Turtle Soup: {str(e)}")
            return False
    
    def _is_holy_grail_pattern(self, window: pd.DataFrame, idx: int) -> bool:
        """Check for Holy Grail pattern"""
        try:
            if 'adx' not in window.columns or 'ema_20' not in window.columns:
                return False
            
            current = window.iloc[-1]
            
            # ADX must have been above threshold
            adx_peak = window['adx'].iloc[-10:].max()
            if adx_peak < self.params['holy_grail_adx_threshold']:
                return False
            
            # ADX retracement
            if current['adx'] > self.params['holy_grail_adx_retracement']:
                return False
            
            # Price pullback to EMA
            if current['trend'] == 'UPTREND':
                # Bullish Holy Grail
                if (current['low'] <= current['ema_20'] and
                    current['close'] > current['ema_20'] and
                    current['close'] > current['open']):
                    self.patterns_found.append(f"Holy Grail Long at {idx}")
                    return True
            
            elif current['trend'] == 'DOWNTREND':
                # Bearish Holy Grail
                if (current['high'] >= current['ema_20'] and
                    current['close'] < current['ema_20'] and
                    current['close'] < current['open']):
                    self.patterns_found.append(f"Holy Grail Short at {idx}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking Holy Grail: {str(e)}")
            return False
    
    def _is_80_20_pattern(self, window: pd.DataFrame, idx: int) -> bool:
        """Check for 80-20 pattern"""
        try:
            current = window.iloc[-1]
            prev = window.iloc[-2]
            
            # Calculate position in daily range
            daily_range = current['high'] - current['low']
            if daily_range == 0:
                return False
            
            close_position = (current['close'] - current['low']) / daily_range
            
            # Bullish 80-20
            if (close_position > self.params['80_20_threshold'] and
                current['close'] > prev['close'] and
                current['close'] > current['open']):
                self.patterns_found.append(f"80-20 Bullish at {idx}")
                return True
            
            # Bearish 80-20
            if (close_position < (1 - self.params['80_20_threshold']) and
                current['close'] < prev['close'] and
                current['close'] < current['open']):
                self.patterns_found.append(f"80-20 Bearish at {idx}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking 80-20: {str(e)}")
            return False
    
    def _is_momentum_pinball_pattern(self, window: pd.DataFrame, idx: int) -> bool:
        """Check for Momentum Pinball pattern"""
        try:
            if len(window) < 5:
                return False
            
            current = window.iloc[-1]
            
            # Strong momentum move
            momentum_bars = window.iloc[-5:]
            
            # Bullish momentum pinball
            if all(momentum_bars['close'] > momentum_bars['open']):
                # All 5 bars are bullish
                total_move = momentum_bars['close'].iloc[-1] - momentum_bars['open'].iloc[0]
                avg_range = momentum_bars['atr'].mean()
                
                if total_move > avg_range * 3:
                    self.patterns_found.append(f"Momentum Pinball Long at {idx}")
                    return True
            
            # Bearish momentum pinball
            if all(momentum_bars['close'] < momentum_bars['open']):
                # All 5 bars are bearish
                total_move = momentum_bars['open'].iloc[0] - momentum_bars['close'].iloc[-1]
                avg_range = momentum_bars['atr'].mean()
                
                if total_move > avg_range * 3:
                    self.patterns_found.append(f"Momentum Pinball Short at {idx}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking Momentum Pinball: {str(e)}")
            return False
    
    def _is_three_bar_triangle_pattern(self, window: pd.DataFrame, idx: int) -> bool:
        """Check for Three Bar Triangle Breakout pattern"""
        try:
            if len(window) < 4:
                return False
            
            bars = window.iloc[-4:]
            
            # Check for narrowing range (triangle)
            ranges = bars['high'] - bars['low']
            
            # Ranges should be decreasing
            if not (ranges.iloc[1] < ranges.iloc[0] and ranges.iloc[2] < ranges.iloc[1]):
                return False
            
            # Breakout bar
            current = bars.iloc[-1]
            triangle_high = bars['high'].iloc[:-1].max()
            triangle_low = bars['low'].iloc[:-1].min()
            
            # Bullish breakout
            if current['close'] > triangle_high and current['close'] > current['open']:
                self.patterns_found.append(f"Three Bar Triangle Breakout Long at {idx}")
                return True
            
            # Bearish breakout
            if current['close'] < triangle_low and current['close'] < current['open']:
                self.patterns_found.append(f"Three Bar Triangle Breakout Short at {idx}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking Three Bar Triangle: {str(e)}")
            return False
    
    def _check_turtle_soup(self, df: pd.DataFrame, patterns: Dict) -> None:
        """Check for Turtle Soup trading setup"""
        try:
            if not patterns['turtle_soup']:
                return
            
            # Get most recent pattern
            last_pattern_idx = patterns['turtle_soup'][-1]
            if last_pattern_idx < len(df) - 3:
                return
            
            current = df.iloc[-1]
            pattern_bar = df.iloc[last_pattern_idx]
            
            # Determine direction from pattern
            lookback = self.params['turtle_soup_lookback']
            period_high = df['high'].iloc[last_pattern_idx-lookback:last_pattern_idx].max()
            period_low = df['low'].iloc[last_pattern_idx-lookback:last_pattern_idx].min()
            
            if pattern_bar['high'] > period_high:
                # Failed breakout above - short setup
                setup = {
                    'type': 'Turtle Soup Short',
                    'direction': 'SELL',
                    'entry_price': current['close'],
                    'stop_loss': pattern_bar['high'] + current['atr'] * 0.5,
                    'take_profit': current['close'] - current['atr'] * 2,
                    'confidence': 0.7,
                    'reason': 'Failed breakout above 20-day high'
                }
                self.setups.append(setup)
            
            elif pattern_bar['low'] < period_low:
                # Failed breakout below - long setup
                setup = {
                    'type': 'Turtle Soup Long',
                    'direction': 'BUY',
                    'entry_price': current['close'],
                    'stop_loss': pattern_bar['low'] - current['atr'] * 0.5,
                    'take_profit': current['close'] + current['atr'] * 2,
                    'confidence': 0.7,
                    'reason': 'Failed breakout below 20-day low'
                }
                self.setups.append(setup)
                
        except Exception as e:
            logger.error(f"Error checking Turtle Soup setup: {str(e)}")
    
    def _check_holy_grail(self, df: pd.DataFrame, patterns: Dict) -> None:
        """Check for Holy Grail trading setup"""
        try:
            if not patterns['holy_grail']:
                return
            
            # Get most recent pattern
            last_pattern_idx = patterns['holy_grail'][-1]
            if last_pattern_idx < len(df) - 2:
                return
            
            current = df.iloc[-1]
            
            if current['trend'] == 'UPTREND':
                # Long setup
                setup = {
                    'type': 'Holy Grail Long',
                    'direction': 'BUY',
                    'entry_price': current['close'],
                    'stop_loss': current['ema_20'] - current['atr'] * 0.5,
                    'take_profit': current['close'] + current['atr'] * 2.5,
                    'confidence': 0.8,
                    'reason': 'ADX retracement to EMA in uptrend'
                }
                self.setups.append(setup)
            
            elif current['trend'] == 'DOWNTREND':
                # Short setup
                setup = {
                    'type': 'Holy Grail Short',
                    'direction': 'SELL',
                    'entry_price': current['close'],
                    'stop_loss': current['ema_20'] + current['atr'] * 0.5,
                    'take_profit': current['close'] - current['atr'] * 2.5,
                    'confidence': 0.8,
                    'reason': 'ADX retracement to EMA in downtrend'
                }
                self.setups.append(setup)
                
        except Exception as e:
            logger.error(f"Error checking Holy Grail setup: {str(e)}")
    
    def _check_80_20(self, df: pd.DataFrame, patterns: Dict) -> None:
        """Check for 80-20 trading setup"""
        try:
            if not patterns['80_20']:
                return
            
            # Get most recent pattern
            last_pattern_idx = patterns['80_20'][-1]
            if last_pattern_idx < len(df) - 2:
                return
            
            current = df.iloc[-1]
            pattern_bar = df.iloc[last_pattern_idx]
            
            # Calculate position in range
            daily_range = pattern_bar['high'] - pattern_bar['low']
            close_position = (pattern_bar['close'] - pattern_bar['low']) / daily_range
            
            if close_position > self.params['80_20_threshold']:
                # Bullish continuation expected
                setup = {
                    'type': '80-20 Long',
                    'direction': 'BUY',
                    'entry_price': current['close'],
                    'stop_loss': pattern_bar['low'],
                    'take_profit': current['close'] + daily_range,
                    'confidence': 0.65,
                    'reason': 'Close in top 20% of range signals continuation'
                }
                self.setups.append(setup)
            
            elif close_position < (1 - self.params['80_20_threshold']):
                # Bearish continuation expected
                setup = {
                    'type': '80-20 Short',
                    'direction': 'SELL',
                    'entry_price': current['close'],
                    'stop_loss': pattern_bar['high'],
                    'take_profit': current['close'] - daily_range,
                    'confidence': 0.65,
                    'reason': 'Close in bottom 20% of range signals continuation'
                }
                self.setups.append(setup)
                
        except Exception as e:
            logger.error(f"Error checking 80-20 setup: {str(e)}")
    
    def _check_momentum_pinball(self, df: pd.DataFrame, patterns: Dict) -> None:
        """Check for Momentum Pinball setup"""
        try:
            if not patterns['momentum_pinball']:
                return
            
            # Get most recent pattern
            last_pattern_idx = patterns['momentum_pinball'][-1]
            if last_pattern_idx < len(df) - 2:
                return
            
            current = df.iloc[-1]
            momentum_bars = df.iloc[last_pattern_idx-4:last_pattern_idx+1]
            
            # Determine direction
            if all(momentum_bars['close'] > momentum_bars['open']):
                # Bullish momentum
                setup = {
                    'type': 'Momentum Pinball Long',
                    'direction': 'BUY',
                    'entry_price': current['close'],
                    'stop_loss': momentum_bars['low'].min(),
                    'take_profit': current['close'] + (momentum_bars['high'].max() - momentum_bars['low'].min()),
                    'confidence': 0.75,
                    'reason': 'Strong bullish momentum surge'
                }
                self.setups.append(setup)
            
            else:
                # Bearish momentum
                setup = {
                    'type': 'Momentum Pinball Short',
                    'direction': 'SELL',
                    'entry_price': current['close'],
                    'stop_loss': momentum_bars['high'].max(),
                    'take_profit': current['close'] - (momentum_bars['high'].max() - momentum_bars['low'].min()),
                    'confidence': 0.75,
                    'reason': 'Strong bearish momentum surge'
                }
                self.setups.append(setup)
                
        except Exception as e:
            logger.error(f"Error checking Momentum Pinball setup: {str(e)}")
    
    def _check_three_bar_triangle(self, df: pd.DataFrame, patterns: Dict) -> None:
        """Check for Three Bar Triangle Breakout setup"""
        try:
            if not patterns['three_bar_triangle']:
                return
            
            # Get most recent pattern
            last_pattern_idx = patterns['three_bar_triangle'][-1]
            if last_pattern_idx != len(df) - 1:
                return
            
            current = df.iloc[-1]
            triangle_bars = df.iloc[last_pattern_idx-3:last_pattern_idx]
            
            # Determine breakout direction
            triangle_high = triangle_bars['high'].max()
            triangle_low = triangle_bars['low'].min()
            
            if current['close'] > triangle_high:
                # Bullish breakout
                setup = {
                    'type': 'Three Bar Triangle Breakout Long',
                    'direction': 'BUY',
                    'entry_price': current['close'],
                    'stop_loss': triangle_low,
                    'take_profit': current['close'] + (triangle_high - triangle_low) * 1.5,
                    'confidence': 0.7,
                    'reason': 'Breakout from three bar triangle consolidation'
                }
                self.setups.append(setup)
            
            elif current['close'] < triangle_low:
                # Bearish breakout
                setup = {
                    'type': 'Three Bar Triangle Breakout Short',
                    'direction': 'SELL',
                    'entry_price': current['close'],
                    'stop_loss': triangle_high,
                    'take_profit': current['close'] - (triangle_high - triangle_low) * 1.5,
                    'confidence': 0.7,
                    'reason': 'Breakdown from three bar triangle consolidation'
                }
                self.setups.append(setup)
                
        except Exception as e:
            logger.error(f"Error checking Three Bar Triangle setup: {str(e)}")
    
    def _classify_momentum_state(self, roc: float, momentum: float) -> str:
        """Classify current momentum state"""
        try:
            if roc > 2 and momentum > 0:
                return 'STRONG_BULLISH'
            elif roc > 0 and momentum > 0:
                return 'BULLISH'
            elif roc < -2 and momentum < 0:
                return 'STRONG_BEARISH'
            elif roc < 0 and momentum < 0:
                return 'BEARISH'
            else:
                return 'NEUTRAL'
                
        except Exception as e:
            logger.error(f"Error classifying momentum: {str(e)}")
            return 'NEUTRAL'
    
    def _generate_signal(self, momentum_analysis: Dict) -> Dict:
        """Generate trading signal from setups and analysis"""
        try:
            if not self.setups:
                return self._neutral_signal("No valid setups found")
            
            # Score and rank setups
            best_setup = None
            best_score = 0
            
            for setup in self.setups:
                score = self._score_setup(setup, momentum_analysis)
                if score > best_score:
                    best_score = score
                    best_setup = setup
            
            if best_setup and best_score > 0.6:
                return {
                    'signal': best_setup['direction'],
                    'strength': best_score,
                    'entry_price': best_setup['entry_price'],
                    'stop_loss': best_setup['stop_loss'],
                    'take_profit': best_setup['take_profit'],
                    'setup_type': best_setup['type'],
                    'confidence': best_setup['confidence'],
                    'patterns': self.patterns_found,
                    'momentum_analysis': momentum_analysis,
                    'reason': best_setup['reason']
                }
            
            return self._neutral_signal("Setup score too low")
            
        except Exception as e:
            logger.error(f"Error generating signal: {str(e)}")
            return self._neutral_signal(f"Signal generation error: {str(e)}")
    
    def _score_setup(self, setup: Dict, momentum_analysis: Dict) -> float:
        """Score a setup based on multiple factors"""
        try:
            base_score = setup['confidence']
            
            # Momentum alignment
            momentum_state = momentum_analysis.get('momentum_state', 'NEUTRAL')
            
            if setup['direction'] == 'BUY':
                if momentum_state in ['STRONG_BULLISH', 'BULLISH']:
                    base_score += 0.15
                elif momentum_state in ['STRONG_BEARISH', 'BEARISH']:
                    base_score -= 0.15
            
            elif setup['direction'] == 'SELL':
                if momentum_state in ['STRONG_BEARISH', 'BEARISH']:
                    base_score += 0.15
                elif momentum_state in ['STRONG_BULLISH', 'BULLISH']:
                    base_score -= 0.15
            
            # Multiple pattern confirmation
            if len(self.patterns_found) > 2:
                base_score += 0.1
            
            # Clamp score
            return max(0, min(1, base_score))
            
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
        logger.info(f"Updated Linda Raschke strategy parameters: {params}")