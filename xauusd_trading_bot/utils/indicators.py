import logging
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """Custom technical indicators for trading analysis"""
    
    @staticmethod
    def keltner_channels(df: pd.DataFrame, period: int = 20, multiplier: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Keltner Channels"""
        try:
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            middle = typical_price.ewm(span=period, adjust=False).mean()
            
            # ATR for channel width
            atr = TechnicalIndicators.average_true_range(df, period)
            
            upper = middle + (atr * multiplier)
            lower = middle - (atr * multiplier)
            
            return upper, middle, lower
            
        except Exception as e:
            logger.error(f"Error calculating Keltner Channels: {str(e)}")
            return pd.Series(), pd.Series(), pd.Series()
    
    @staticmethod
    def average_true_range(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        try:
            high = df['high']
            low = df['low']
            close = df['close']
            
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.ewm(span=period, adjust=False).mean()
            
            return atr
            
        except Exception as e:
            logger.error(f"Error calculating ATR: {str(e)}")
            return pd.Series()
    
    @staticmethod
    def cumulative_delta(df: pd.DataFrame) -> pd.Series:
        """Calculate cumulative delta (buy vs sell volume)"""
        try:
            # Estimate buy/sell volume
            buy_volume = df['tick_volume'] * ((df['close'] - df['low']) / (df['high'] - df['low']))
            sell_volume = df['tick_volume'] * ((df['high'] - df['close']) / (df['high'] - df['low']))
            
            delta = buy_volume - sell_volume
            cumulative_delta = delta.cumsum()
            
            return cumulative_delta
            
        except Exception as e:
            logger.error(f"Error calculating cumulative delta: {str(e)}")
            return pd.Series()
    
    @staticmethod
    def market_profile_poc(df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Point of Control (POC) from market profile"""
        try:
            poc_values = []
            
            for i in range(period, len(df)):
                window = df.iloc[i-period:i]
                
                # Create price histogram
                price_range = np.linspace(window['low'].min(), window['high'].max(), 50)
                volumes = []
                
                for price in price_range:
                    # Count volume at each price level
                    mask = (window['low'] <= price) & (window['high'] >= price)
                    vol = window.loc[mask, 'tick_volume'].sum()
                    volumes.append(vol)
                
                # Find POC (price with highest volume)
                if volumes:
                    poc_idx = np.argmax(volumes)
                    poc = price_range[poc_idx]
                else:
                    poc = window['close'].iloc[-1]
                
                poc_values.append(poc)
            
            # Pad the beginning
            poc_series = pd.Series([np.nan] * period + poc_values, index=df.index)
            
            return poc_series
            
        except Exception as e:
            logger.error(f"Error calculating POC: {str(e)}")
            return pd.Series()
    
    @staticmethod
    def volume_weighted_average_price(df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate VWAP"""
        try:
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            vwap = (typical_price * df['tick_volume']).rolling(window=period).sum() / df['tick_volume'].rolling(window=period).sum()
            
            return vwap
            
        except Exception as e:
            logger.error(f"Error calculating VWAP: {str(e)}")
            return pd.Series()
    
    @staticmethod
    def squeeze_momentum(df: pd.DataFrame, bb_period: int = 20, kc_period: int = 20, 
                        bb_std: float = 2.0, kc_mult: float = 1.5) -> Dict[str, pd.Series]:
        """Calculate Squeeze Momentum Indicator"""
        try:
            # Bollinger Bands
            bb_middle = df['close'].rolling(window=bb_period).mean()
            bb_std_dev = df['close'].rolling(window=bb_period).std()
            bb_upper = bb_middle + (bb_std_dev * bb_std)
            bb_lower = bb_middle - (bb_std_dev * bb_std)
            
            # Keltner Channels
            kc_upper, kc_middle, kc_lower = TechnicalIndicators.keltner_channels(df, kc_period, kc_mult)
            
            # Squeeze detection
            squeeze_on = (bb_lower > kc_lower) & (bb_upper < kc_upper)
            squeeze_off = (bb_lower < kc_lower) | (bb_upper > kc_upper)
            
            # Momentum calculation
            highest = df['high'].rolling(window=kc_period).max()
            lowest = df['low'].rolling(window=kc_period).min()
            mean_price = (highest + lowest) / 2
            momentum = df['close'] - mean_price
            
            return {
                'squeeze': squeeze_on.astype(int),
                'momentum': momentum,
                'histogram': momentum - momentum.rolling(window=12).mean()
            }
            
        except Exception as e:
            logger.error(f"Error calculating Squeeze Momentum: {str(e)}")
            return {}
    
    @staticmethod
    def order_flow_imbalance(df: pd.DataFrame, levels: int = 10) -> pd.Series:
        """Calculate order flow imbalance"""
        try:
            imbalances = []
            
            for i in range(len(df)):
                bar = df.iloc[i]
                
                # Divide bar into levels
                price_levels = np.linspace(bar['low'], bar['high'], levels)
                
                # Estimate volume at each level
                level_volume = bar['tick_volume'] / levels
                
                # Calculate imbalance (simplified)
                if bar['close'] > bar['open']:
                    # Bullish bar - more buying at higher levels
                    buy_pressure = sum(range(levels//2, levels)) * level_volume
                    sell_pressure = sum(range(levels//2)) * level_volume
                else:
                    # Bearish bar - more selling at lower levels
                    sell_pressure = sum(range(levels//2, levels)) * level_volume
                    buy_pressure = sum(range(levels//2)) * level_volume
                
                imbalance = (buy_pressure - sell_pressure) / bar['tick_volume']
                imbalances.append(imbalance)
            
            return pd.Series(imbalances, index=df.index)
            
        except Exception as e:
            logger.error(f"Error calculating order flow imbalance: {str(e)}")
            return pd.Series()
    
    @staticmethod
    def pivot_points(df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate pivot points and support/resistance levels"""
        try:
            # Standard pivot calculation
            pivot = (df['high'].shift(1) + df['low'].shift(1) + df['close'].shift(1)) / 3
            
            # Support and resistance levels
            r1 = 2 * pivot - df['low'].shift(1)
            s1 = 2 * pivot - df['high'].shift(1)
            
            r2 = pivot + (df['high'].shift(1) - df['low'].shift(1))
            s2 = pivot - (df['high'].shift(1) - df['low'].shift(1))
            
            r3 = df['high'].shift(1) + 2 * (pivot - df['low'].shift(1))
            s3 = df['low'].shift(1) - 2 * (df['high'].shift(1) - pivot)
            
            return {
                'pivot': pivot,
                'r1': r1, 'r2': r2, 'r3': r3,
                's1': s1, 's2': s2, 's3': s3
            }
            
        except Exception as e:
            logger.error(f"Error calculating pivot points: {str(e)}")
            return {}
    
    @staticmethod
    def market_internals(df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate market internal indicators"""
        try:
            # Advance/Decline ratio
            advances = (df['close'] > df['open']).rolling(window=20).sum()
            declines = (df['close'] < df['open']).rolling(window=20).sum()
            ad_ratio = advances / (declines + 1)  # Avoid division by zero
            
            # Up/Down volume ratio
            up_volume = df.loc[df['close'] > df['open'], 'tick_volume'].rolling(window=20).sum()
            down_volume = df.loc[df['close'] < df['open'], 'tick_volume'].rolling(window=20).sum()
            volume_ratio = up_volume / (down_volume + 1)
            
            # McClellan Oscillator (simplified)
            ema_19 = ad_ratio.ewm(span=19, adjust=False).mean()
            ema_39 = ad_ratio.ewm(span=39, adjust=False).mean()
            mcclellan = ema_19 - ema_39
            
            return {
                'ad_ratio': ad_ratio,
                'volume_ratio': volume_ratio,
                'mcclellan': mcclellan
            }
            
        except Exception as e:
            logger.error(f"Error calculating market internals: {str(e)}")
            return {}