"""
ICT (Inner Circle Trader) Strategy
Implements ICT Smart Money Concepts
"""

import logging
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
from datetime import datetime, time

from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class ICTStrategy(BaseStrategy):
    """ICT Smart Money Concepts Strategy Implementation"""
    
    def __init__(self):
        """Initialize ICT strategy"""
        super().__init__(name="ICT Smart Money")
        
        # Strategy parameters
        self.params = {
            'liquidity_lookback': 20,
            'order_block_strength': 0.7,
            'fvg_min_size': 0.5,  # Minimum FVG size in ATR units
            'market_structure_lookback': 50,
            'optimal_trade_entry': [0.62, 0.705, 0.79],  # Fibonacci levels
            'risk_reward_min': 2.0,
            'session_times': {
                'asian': {'start': '00:00', 'end': '09:00'},
                'london': {'start': '08:00', 'end': '17:00'},
                'ny': {'start': '13:00', 'end': '22:00'}
            }
        }
    
    def get_minimum_bars(self) -> int:
        """ICT strategy needs at least 50 bars"""
        return 50
        
    def analyze(self, df: pd.DataFrame) -> Dict:
        """Analyze market using ICT concepts"""
        try:
            if len(df) < self.params['market_structure_lookback']:
                return self._neutral_signal("Insufficient data")
            
            # Reset tracking
            self._reset_tracking()
            
            # Identify market structure
            market_structure = self._analyze_market_structure(df)
            
            # Find liquidity zones
            self._identify_liquidity_zones(df)
            
            # Find order blocks
            self._identify_order_blocks(df)
            
            # Find fair value gaps
            self._identify_fair_value_gaps(df)
            
            # Check for market structure shift
            mss = self._check_market_structure_shift(df)
            
            # Identify current session
            current_session = self._get_current_session()
            
            # Find optimal trade entries
            trade_setups = self._find_trade_setups(df, market_structure, mss, current_session)
            
            # Generate signal
            signal = self._generate_signal(trade_setups, market_structure, current_session)
            
            return signal
            
        except Exception as e:
            logger.error(f"Error in ICT analysis: {str(e)}")
            return self._neutral_signal(f"Analysis error: {str(e)}")
    
    def _reset_tracking(self):
        """Reset ICT concept tracking"""
        self.liquidity_zones = []
        self.order_blocks = []
        self.fair_value_gaps = []
        self.market_structure_breaks = []
    
    def _analyze_market_structure(self, df: pd.DataFrame) -> Dict:
        """Analyze market structure using ICT concepts"""
        try:
            structure = {
                'bias': 'NEUTRAL',
                'trend': 'NEUTRAL',
                'key_levels': {},
                'swing_points': {'highs': [], 'lows': []},
                'current_range': {}
            }
            
            # Find significant swing points
            swing_highs = df[df['swing_high'] == 1].copy()
            swing_lows = df[df['swing_low'] == 1].copy()
            
            if len(swing_highs) >= 2 and len(swing_lows) >= 2:
                # Get recent swing points
                recent_highs = swing_highs.iloc[-3:]
                recent_lows = swing_lows.iloc[-3:]
                
                structure['swing_points']['highs'] = recent_highs.index.tolist()
                structure['swing_points']['lows'] = recent_lows.index.tolist()
                
                # Determine trend based on swing points
                if (recent_highs['high'].iloc[-1] > recent_highs['high'].iloc[-2] and
                    recent_lows['low'].iloc[-1] > recent_lows['low'].iloc[-2]):
                    structure['trend'] = 'BULLISH'
                    structure['bias'] = 'LONG'
                elif (recent_highs['high'].iloc[-1] < recent_highs['high'].iloc[-2] and
                      recent_lows['low'].iloc[-1] < recent_lows['low'].iloc[-2]):
                    structure['trend'] = 'BEARISH'
                    structure['bias'] = 'SHORT'
                else:
                    structure['trend'] = 'RANGING'
                
                # Define current range
                structure['current_range'] = {
                    'high': recent_highs['high'].max(),
                    'low': recent_lows['low'].min(),
                    'mid': (recent_highs['high'].max() + recent_lows['low'].min()) / 2
                }
                
                # Key levels
                structure['key_levels'] = {
                    'resistance': recent_highs['high'].iloc[-1],
                    'support': recent_lows['low'].iloc[-1],
                    'previous_high': recent_highs['high'].iloc[-2] if len(recent_highs) > 1 else None,
                    'previous_low': recent_lows['low'].iloc[-2] if len(recent_lows) > 1 else None
                }
            
            return structure
            
        except Exception as e:
            logger.error(f"Error analyzing market structure: {str(e)}")
            return {'bias': 'NEUTRAL', 'trend': 'NEUTRAL'}
    
    def _identify_liquidity_zones(self, df: pd.DataFrame) -> None:
        """Identify buy-side and sell-side liquidity zones"""
        try:
            lookback = self.params['liquidity_lookback']
            
            # Find areas with multiple equal highs/lows
            for i in range(lookback, len(df)):
                window = df.iloc[i-lookback:i]
                
                # Buy-side liquidity (above swing highs)
                high_counts = window['high'].value_counts()
                for price, count in high_counts.items():
                    if count >= 2:  # Multiple touches
                        self.liquidity_zones.append({
                            'type': 'BUY_SIDE',
                            'price': price,
                            'strength': count,
                            'index': i
                        })
                
                # Sell-side liquidity (below swing lows)
                low_counts = window['low'].value_counts()
                for price, count in low_counts.items():
                    if count >= 2:  # Multiple touches
                        self.liquidity_zones.append({
                            'type': 'SELL_SIDE',
                            'price': price,
                            'strength': count,
                            'index': i
                        })
            
            # Also add liquidity above/below obvious swing points
            swing_highs = df[df['swing_high'] == 1]['high'].iloc[-5:]
            swing_lows = df[df['swing_low'] == 1]['low'].iloc[-5:]
            
            for high in swing_highs:
                self.liquidity_zones.append({
                    'type': 'BUY_SIDE',
                    'price': high + df['atr'].iloc[-1] * 0.1,
                    'strength': 1,
                    'index': len(df) - 1
                })
            
            for low in swing_lows:
                self.liquidity_zones.append({
                    'type': 'SELL_SIDE',
                    'price': low - df['atr'].iloc[-1] * 0.1,
                    'strength': 1,
                    'index': len(df) - 1
                })
                
        except Exception as e:
            logger.error(f"Error identifying liquidity zones: {str(e)}")
    
    def _identify_order_blocks(self, df: pd.DataFrame) -> None:
        """Identify bullish and bearish order blocks"""
        try:
            # Look for strong moves away from consolidation
            for i in range(10, len(df) - 1):
                current = df.iloc[i]
                next_bar = df.iloc[i + 1]
                
                # Bullish order block
                if (current['close'] > current['open'] and
                    next_bar['close'] > next_bar['open'] and
                    next_bar['close'] > current['high'] and
                    current['body_size'] > current['atr'] * self.params['order_block_strength']):
                    
                    self.order_blocks.append({
                        'type': 'BULLISH',
                        'high': current['high'],
                        'low': current['low'],
                        'index': i,
                        'strength': current['body_size'] / current['atr']
                    })
                
                # Bearish order block
                if (current['close'] < current['open'] and
                    next_bar['close'] < next_bar['open'] and
                    next_bar['close'] < current['low'] and
                    current['body_size'] > current['atr'] * self.params['order_block_strength']):
                    
                    self.order_blocks.append({
                        'type': 'BEARISH',
                        'high': current['high'],
                        'low': current['low'],
                        'index': i,
                        'strength': current['body_size'] / current['atr']
                    })
                    
        except Exception as e:
            logger.error(f"Error identifying order blocks: {str(e)}")
    
    def _identify_fair_value_gaps(self, df: pd.DataFrame) -> None:
        """Identify fair value gaps (imbalances)"""
        try:
            min_gap_size = self.params['fvg_min_size']
            
            for i in range(2, len(df) - 1):
                prev_bar = df.iloc[i - 1]
                current = df.iloc[i]
                next_bar = df.iloc[i + 1]
                
                # Bullish FVG
                if prev_bar['high'] < next_bar['low']:
                    gap_size = next_bar['low'] - prev_bar['high']
                    if gap_size > current['atr'] * min_gap_size:
                        self.fair_value_gaps.append({
                            'type': 'BULLISH',
                            'high': next_bar['low'],
                            'low': prev_bar['high'],
                            'index': i,
                            'size': gap_size,
                            'filled': False
                        })
                
                # Bearish FVG
                if prev_bar['low'] > next_bar['high']:
                    gap_size = prev_bar['low'] - next_bar['high']
                    if gap_size > current['atr'] * min_gap_size:
                        self.fair_value_gaps.append({
                            'type': 'BEARISH',
                            'high': prev_bar['low'],
                            'low': next_bar['high'],
                            'index': i,
                            'size': gap_size,
                            'filled': False
                        })
            
            # Check if gaps have been filled
            for gap in self.fair_value_gaps:
                if gap['index'] < len(df) - 5:
                    check_bars = df.iloc[gap['index']:gap['index'] + 5]
                    if gap['type'] == 'BULLISH':
                        if check_bars['low'].min() <= gap['low']:
                            gap['filled'] = True
                    else:
                        if check_bars['high'].max() >= gap['high']:
                            gap['filled'] = True
                            
        except Exception as e:
            logger.error(f"Error identifying fair value gaps: {str(e)}")
    
    def _check_market_structure_shift(self, df: pd.DataFrame) -> Dict:
        """Check for market structure shift (MSS) or change of character (CHoCH)"""
        try:
            mss = {
                'occurred': False,
                'type': None,
                'level': None,
                'index': None
            }
            
            if len(self.market_structure_breaks) < 2:
                # Find recent structure breaks
                for i in range(20, len(df)):
                    window = df.iloc[i-20:i]
                    current = df.iloc[i]
                    
                    # Find previous swing high/low
                    prev_swing_high = window[window['swing_high'] == 1]['high'].max()
                    prev_swing_low = window[window['swing_low'] == 1]['low'].min()
                    
                    # Bullish MSS - break above previous high in downtrend
                    if current['close'] > prev_swing_high and window['trend'].iloc[-1] == 'DOWNTREND':
                        mss['occurred'] = True
                        mss['type'] = 'BULLISH_MSS'
                        mss['level'] = prev_swing_high
                        mss['index'] = i
                        self.market_structure_breaks.append(mss.copy())
                    
                    # Bearish MSS - break below previous low in uptrend
                    if current['close'] < prev_swing_low and window['trend'].iloc[-1] == 'UPTREND':
                        mss['occurred'] = True
                        mss['type'] = 'BEARISH_MSS'
                        mss['level'] = prev_swing_low
                        mss['index'] = i
                        self.market_structure_breaks.append(mss.copy())
            
            # Return most recent MSS
            if self.market_structure_breaks:
                return self.market_structure_breaks[-1]
            
            return mss
            
        except Exception as e:
            logger.error(f"Error checking market structure shift: {str(e)}")
            return {'occurred': False}
    
    def _get_current_session(self) -> str:
        """Get current trading session"""
        try:
            current_time = datetime.now().time()
            
            for session, times in self.params['session_times'].items():
                start = datetime.strptime(times['start'], '%H:%M').time()
                end = datetime.strptime(times['end'], '%H:%M').time()
                
                if start <= current_time <= end:
                    return session.upper()
            
            return 'CLOSED'
            
        except Exception as e:
            logger.error(f"Error getting current session: {str(e)}")
            return 'UNKNOWN'
    
    def _find_trade_setups(self, df: pd.DataFrame, market_structure: Dict, 
                          mss: Dict, session: str) -> List[Dict]:
        """Find ICT trade setups"""
        setups = []
        
        try:
            current = df.iloc[-1]
            
            # 1. Liquidity Sweep Setup
            liquidity_setup = self._check_liquidity_sweep_setup(df, market_structure)
            if liquidity_setup:
                setups.append(liquidity_setup)
            
            # 2. Order Block Setup
            ob_setup = self._check_order_block_setup(df, market_structure)
            if ob_setup:
                setups.append(ob_setup)
            
            # 3. Fair Value Gap Setup
            fvg_setup = self._check_fvg_setup(df, market_structure)
            if fvg_setup:
                setups.append(fvg_setup)
            
            # 4. Market Structure Shift Setup
            if mss['occurred']:
                mss_setup = self._check_mss_setup(df, mss, market_structure)
                if mss_setup:
                    setups.append(mss_setup)
            
            # 5. Optimal Trade Entry Setup
            ote_setup = self._check_optimal_trade_entry(df, market_structure)
            if ote_setup:
                setups.append(ote_setup)
            
            # 6. Session-based setups
            if session in ['LONDON', 'NY']:
                session_setup = self._check_session_setup(df, session, market_structure)
                if session_setup:
                    setups.append(session_setup)
            
            return setups
            
        except Exception as e:
            logger.error(f"Error finding trade setups: {str(e)}")
            return []
    
    def _check_liquidity_sweep_setup(self, df: pd.DataFrame, market_structure: Dict) -> Optional[Dict]:
        """Check for liquidity sweep setup"""
        try:
            if not self.liquidity_zones:
                return None
            
            current = df.iloc[-1]
            recent_bars = df.iloc[-5:]
            
            # Find recent liquidity sweeps
            for zone in self.liquidity_zones[-5:]:
                if zone['type'] == 'BUY_SIDE':
                    # Check if buy-side liquidity was swept
                    if recent_bars['high'].max() > zone['price']:
                        # Look for reversal after sweep
                        if current['close'] < zone['price'] and current['close'] < current['open']:
                            return {
                                'type': 'Liquidity Sweep Short',
                                'direction': 'SELL',
                                'entry_price': current['close'],
                                'stop_loss': recent_bars['high'].max() + current['atr'] * 0.5,
                                'take_profit': current['close'] - current['atr'] * self.params['risk_reward_min'],
                                'confidence': 0.75,
                                'reason': 'Buy-side liquidity swept with bearish reversal'
                            }
                
                elif zone['type'] == 'SELL_SIDE':
                    # Check if sell-side liquidity was swept
                    if recent_bars['low'].min() < zone['price']:
                        # Look for reversal after sweep
                        if current['close'] > zone['price'] and current['close'] > current['open']:
                            return {
                                'type': 'Liquidity Sweep Long',
                                'direction': 'BUY',
                                'entry_price': current['close'],
                                'stop_loss': recent_bars['low'].min() - current['atr'] * 0.5,
                                'take_profit': current['close'] + current['atr'] * self.params['risk_reward_min'],
                                'confidence': 0.75,
                                'reason': 'Sell-side liquidity swept with bullish reversal'
                            }
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking liquidity sweep: {str(e)}")
            return None
    
    def _check_order_block_setup(self, df: pd.DataFrame, market_structure: Dict) -> Optional[Dict]:
        """Check for order block setup"""
        try:
            if not self.order_blocks:
                return None
            
            current = df.iloc[-1]
            
            # Find valid order blocks
            for ob in self.order_blocks[-5:]:
                ob_data = df.iloc[ob['index']]
                
                if ob['type'] == 'BULLISH' and market_structure['bias'] == 'LONG':
                    # Price retraced to bullish order block
                    if ob['low'] <= current['low'] <= ob['high']:
                        if current['close'] > current['open']:  # Bullish reaction
                            return {
                                'type': 'Bullish Order Block',
                                'direction': 'BUY',
                                'entry_price': current['close'],
                                'stop_loss': ob['low'] - current['atr'] * 0.3,
                                'take_profit': current['close'] + current['atr'] * self.params['risk_reward_min'],
                                'confidence': 0.8,
                                'reason': 'Price retraced to bullish order block with confirmation'
                            }
                
                elif ob['type'] == 'BEARISH' and market_structure['bias'] == 'SHORT':
                    # Price retraced to bearish order block
                    if ob['low'] <= current['high'] <= ob['high']:
                        if current['close'] < current['open']:  # Bearish reaction
                            return {
                                'type': 'Bearish Order Block',
                                'direction': 'SELL',
                                'entry_price': current['close'],
                                'stop_loss': ob['high'] + current['atr'] * 0.3,
                                'take_profit': current['close'] - current['atr'] * self.params['risk_reward_min'],
                                'confidence': 0.8,
                                'reason': 'Price retraced to bearish order block with confirmation'
                            }
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking order block setup: {str(e)}")
            return None
    
    def _check_fvg_setup(self, df: pd.DataFrame, market_structure: Dict) -> Optional[Dict]:
        """Check for fair value gap setup"""
        try:
            if not self.fair_value_gaps:
                return None
            
            current = df.iloc[-1]
            
            # Find unfilled FVGs
            unfilled_gaps = [gap for gap in self.fair_value_gaps if not gap['filled']]
            
            for gap in unfilled_gaps[-3:]:
                if gap['type'] == 'BULLISH' and market_structure['bias'] == 'LONG':
                    # Price entered bullish FVG
                    if gap['low'] <= current['low'] <= gap['high']:
                        if current['close'] > current['open']:
                            return {
                                'type': 'Bullish FVG',
                                'direction': 'BUY',
                                'entry_price': current['close'],
                                'stop_loss': gap['low'] - current['atr'] * 0.2,
                                'take_profit': current['close'] + gap['size'] * 2,
                                'confidence': 0.7,
                                'reason': 'Price filled into bullish fair value gap'
                            }
                
                elif gap['type'] == 'BEARISH' and market_structure['bias'] == 'SHORT':
                    # Price entered bearish FVG
                    if gap['low'] <= current['high'] <= gap['high']:
                        if current['close'] < current['open']:
                            return {
                                'type': 'Bearish FVG',
                                'direction': 'SELL',
                                'entry_price': current['close'],
                                'stop_loss': gap['high'] + current['atr'] * 0.2,
                                'take_profit': current['close'] - gap['size'] * 2,
                                'confidence': 0.7,
                                'reason': 'Price filled into bearish fair value gap'
                            }
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking FVG setup: {str(e)}")
            return None
    
    def _check_mss_setup(self, df: pd.DataFrame, mss: Dict, market_structure: Dict) -> Optional[Dict]:
        """Check for market structure shift setup"""
        try:
            if not mss['occurred']:
                return None
            
            current = df.iloc[-1]
            mss_bar = df.iloc[mss['index']]
            
            if mss['type'] == 'BULLISH_MSS':
                # Look for pullback after bullish MSS
                if current['low'] > mss['level'] and current['close'] > current['open']:
                    return {
                        'type': 'Bullish MSS',
                        'direction': 'BUY',
                        'entry_price': current['close'],
                        'stop_loss': mss['level'] - current['atr'] * 0.5,
                        'take_profit': current['close'] + current['atr'] * self.params['risk_reward_min'],
                        'confidence': 0.85,
                        'reason': 'Bullish market structure shift confirmed'
                    }
            
            elif mss['type'] == 'BEARISH_MSS':
                # Look for pullback after bearish MSS
                if current['high'] < mss['level'] and current['close'] < current['open']:
                    return {
                        'type': 'Bearish MSS',
                        'direction': 'SELL',
                        'entry_price': current['close'],
                        'stop_loss': mss['level'] + current['atr'] * 0.5,
                        'take_profit': current['close'] - current['atr'] * self.params['risk_reward_min'],
                        'confidence': 0.85,
                        'reason': 'Bearish market structure shift confirmed'
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking MSS setup: {str(e)}")
            return None
    
    def _check_optimal_trade_entry(self, df: pd.DataFrame, market_structure: Dict) -> Optional[Dict]:
        """Check for optimal trade entry (OTE) setup"""
        try:
            if market_structure['trend'] == 'NEUTRAL':
                return None
            
            current = df.iloc[-1]
            
            # Find recent impulse move
            recent_bars = df.iloc[-20:]
            
            if market_structure['trend'] == 'BULLISH':
                # Find bullish impulse
                impulse_low = recent_bars['low'].min()
                impulse_high = recent_bars['high'].max()
                impulse_range = impulse_high - impulse_low
                
                # Check if price is in OTE zone (62-79% retracement)
                retracement = (impulse_high - current['low']) / impulse_range
                
                if 0.62 <= retracement <= 0.79:
                    if current['close'] > current['open']:
                        return {
                            'type': 'Bullish OTE',
                            'direction': 'BUY',
                            'entry_price': current['close'],
                            'stop_loss': impulse_low - current['atr'] * 0.2,
                            'take_profit': impulse_high + impulse_range * 0.5,
                            'confidence': 0.75,
                            'reason': 'Price in optimal trade entry zone (62-79% retracement)'
                        }
            
            elif market_structure['trend'] == 'BEARISH':
                # Find bearish impulse
                impulse_high = recent_bars['high'].max()
                impulse_low = recent_bars['low'].min()
                impulse_range = impulse_high - impulse_low
                
                # Check if price is in OTE zone
                retracement = (current['high'] - impulse_low) / impulse_range
                
                if 0.62 <= retracement <= 0.79:
                    if current['close'] < current['open']:
                        return {
                            'type': 'Bearish OTE',
                            'direction': 'SELL',
                            'entry_price': current['close'],
                            'stop_loss': impulse_high + current['atr'] * 0.2,
                            'take_profit': impulse_low - impulse_range * 0.5,
                            'confidence': 0.75,
                            'reason': 'Price in optimal trade entry zone (62-79% retracement)'
                        }
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking OTE setup: {str(e)}")
            return None
    
    def _check_session_setup(self, df: pd.DataFrame, session: str, market_structure: Dict) -> Optional[Dict]:
        """Check for session-based setup"""
        try:
            current = df.iloc[-1]
            
            if session == 'LONDON':
                # London session characteristics
                if market_structure['bias'] == 'LONG':
                    # Look for London breakout
                    asian_high = df.iloc[-16:-8]['high'].max()  # Asian session high
                    if current['close'] > asian_high and current['close'] > current['open']:
                        return {
                            'type': 'London Breakout Long',
                            'direction': 'BUY',
                            'entry_price': current['close'],
                            'stop_loss': asian_high - current['atr'] * 0.5,
                            'take_profit': current['close'] + current['atr'] * self.params['risk_reward_min'],
                            'confidence': 0.7,
                            'reason': 'London session breakout above Asian high'
                        }
            
            elif session == 'NY':
                # New York session characteristics
                if market_structure['bias'] == 'SHORT':
                    # Look for NY reversal
                    london_low = df.iloc[-8:-4]['low'].min()  # London session low
                    if current['close'] < london_low and current['close'] < current['open']:
                        return {
                            'type': 'NY Reversal Short',
                            'direction': 'SELL',
                            'entry_price': current['close'],
                            'stop_loss': london_low + current['atr'] * 0.5,
                            'take_profit': current['close'] - current['atr'] * self.params['risk_reward_min'],
                            'confidence': 0.7,
                            'reason': 'New York session reversal below London low'
                        }
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking session setup: {str(e)}")
            return None
    
    def _generate_signal(self, setups: List[Dict], market_structure: Dict, session: str) -> Dict:
        """Generate trading signal from ICT analysis"""
        try:
            if not setups:
                return self._neutral_signal("No valid ICT setups found")
            
            # Score and rank setups
            best_setup = None
            best_score = 0
            
            for setup in setups:
                score = self._score_setup(setup, market_structure, session)
                if score > best_score:
                    best_score = score
                    best_setup = setup
            
            if best_setup and best_score > 0.65:
                # Add ICT-specific information
                ict_analysis = {
                    'liquidity_zones': len(self.liquidity_zones),
                    'order_blocks': len(self.order_blocks),
                    'fair_value_gaps': len([g for g in self.fair_value_gaps if not g['filled']]),
                    'market_structure': market_structure,
                    'session': session
                }
                
                return {
                    'signal': best_setup['direction'],
                    'strength': best_score,
                    'entry_price': best_setup['entry_price'],
                    'stop_loss': best_setup['stop_loss'],
                    'take_profit': best_setup['take_profit'],
                    'setup_type': best_setup['type'],
                    'confidence': best_setup['confidence'],
                    'ict_analysis': ict_analysis,
                    'reason': best_setup['reason']
                }
            
            return self._neutral_signal("ICT setup score too low")
            
        except Exception as e:
            logger.error(f"Error generating signal: {str(e)}")
            return self._neutral_signal(f"Signal generation error: {str(e)}")
    
    def _score_setup(self, setup: Dict, market_structure: Dict, session: str) -> float:
        """Score an ICT setup"""
        try:
            base_score = setup['confidence']
            
            # Market structure alignment
            if market_structure['trend'] != 'NEUTRAL':
                if (setup['direction'] == 'BUY' and market_structure['bias'] == 'LONG') or \
                   (setup['direction'] == 'SELL' and market_structure['bias'] == 'SHORT'):
                    base_score += 0.15
                else:
                    base_score -= 0.15
            
            # Session bonus
            if session in ['LONDON', 'NY']:
                base_score += 0.05
            
            # Multiple concept confluence
            concept_count = sum([
                len(self.liquidity_zones) > 0,
                len(self.order_blocks) > 0,
                len([g for g in self.fair_value_gaps if not g['filled']]) > 0,
                len(self.market_structure_breaks) > 0
            ])
            
            if concept_count >= 3:
                base_score += 0.1
            
            return min(1.0, max(0.0, base_score))
            
        except Exception as e:
            logger.error(f"Error scoring setup: {str(e)}")
            return 0.0
    
    def _neutral_signal(self, reason: str) -> Dict:
        """Generate neutral signal"""
        return {
            'signal': 'NEUTRAL',
            'strength': 0.0,
            'confidence': 0.0,
            'ict_analysis': {
                'liquidity_zones': len(self.liquidity_zones),
                'order_blocks': len(self.order_blocks),
                'fair_value_gaps': len(self.fair_value_gaps)
            },
            'reason': reason
        }
    
    def update_parameters(self, params: Dict) -> None:
        """Update strategy parameters"""
        self.params.update(params)
        logger.info(f"Updated ICT strategy parameters: {params}")