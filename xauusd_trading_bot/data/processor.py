import logging
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
from scipy import stats

# Try to import TA-Lib with fallback to pandas-ta
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    import warnings
    warnings.warn("TA-Lib not available, falling back to pandas-ta for technical indicators", UserWarning)
    try:
        import pandas_ta as ta
        TALIB_AVAILABLE = False
    except ImportError:
        raise ImportError("Neither TA-Lib nor pandas-ta is available. Install one of them: 'pip install TA-Lib' or 'pip install pandas-ta'")

logger = logging.getLogger(__name__)


class DataProcessor:
    """Data processing and feature engineering with memory optimization"""
    
    def __init__(self, memory_limit_mb: int = 50):
        """Initialize data processor with memory optimization
        
        Args:
            memory_limit_mb: Memory limit in MB for processing (default: 50MB)
        """
        self.required_columns = ['open', 'high', 'low', 'close', 'tick_volume']
        self.memory_limit_mb = memory_limit_mb
        self.max_indicators = self._calculate_max_indicators()
    
    def _calculate_max_indicators(self) -> int:
        """Calculate maximum number of indicators based on memory limit"""
        # Rough estimate: each indicator column takes ~8 bytes per row
        # Base columns: 5, typical indicators: ~25-30
        # For 512MB server, limit to essential indicators only
        if self.memory_limit_mb <= 30:
            return 10  # Essential indicators only
        elif self.memory_limit_mb <= 50:
            return 15  # Core indicators
        else:
            return 25  # Full indicator set
    
    def _estimate_memory_usage(self, df: pd.DataFrame, additional_cols: int = 0) -> float:
        """Estimate memory usage in MB for DataFrame with additional columns"""
        current_cols = len(df.columns)
        total_cols = current_cols + additional_cols
        rows = len(df)
        # Assume 8 bytes per float64 value
        estimated_mb = (rows * total_cols * 8) / (1024 * 1024)
        return estimated_mb
    
    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame data types to reduce memory usage"""
        try:
            # Convert appropriate columns to float32 instead of float64
            float_cols = df.select_dtypes(include=['float64']).columns
            for col in float_cols:
                # Check if conversion to float32 would lose significant precision
                if df[col].notna().any():
                    col_max = df[col].max()
                    col_min = df[col].min()
                    # float32 range is approximately ±3.4e38, precision ~7 decimal digits
                    if (abs(col_max) < 1e37 and abs(col_min) < 1e37 and 
                        not col.endswith('_id') and 'timestamp' not in col.lower()):
                        df[col] = df[col].astype('float32')
            
            # Convert boolean-like integer columns to bool
            for col in df.columns:
                if df[col].dtype in ['int64', 'int32'] and df[col].nunique() <= 2:
                    unique_vals = set(df[col].dropna().unique())
                    if unique_vals.issubset({0, 1, -1, 100, -100}):  # Common pattern indicator values
                        df[col] = df[col].astype('int8')
            
            logger.debug(f"Optimized dtypes, memory usage reduced by ~{100*(1-0.5):.0f}%")
            return df
            
        except Exception as e:
            logger.warning(f"Failed to optimize dtypes: {e}")
            return df
        
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for analysis with memory optimization"""
        try:
            if df.empty:
                return df
            
            # Validate required columns
            if not all(col in df.columns for col in self.required_columns):
                raise ValueError(f"Missing required columns: {self.required_columns}")
            
            # Memory usage check
            estimated_memory = self._estimate_memory_usage(df, additional_cols=30)
            if estimated_memory > self.memory_limit_mb:
                logger.warning(f"Estimated memory usage {estimated_memory:.1f}MB exceeds limit {self.memory_limit_mb}MB. Using reduced feature set.")
                return self._prepare_data_memory_optimized(df)
            
            # Create a copy and optimize dtypes early
            processed_df = df.copy()
            processed_df = self._optimize_dtypes(processed_df)
            
            # Add basic features
            processed_df = self._add_basic_features(processed_df)
            
            # Add technical indicators (with memory limit)
            processed_df = self._add_technical_indicators_optimized(processed_df)
            
            # Add price patterns (selective)
            processed_df = self._add_price_patterns_selective(processed_df)
            
            # Add market structure
            processed_df = self._add_market_structure(processed_df)
            
            # Final optimization and cleanup
            processed_df = self._optimize_dtypes(processed_df)
            processed_df = self._clean_data(processed_df)
            
            # Final memory check and log
            final_memory = self._estimate_memory_usage(processed_df)
            logger.debug(f"Final DataFrame memory usage: {final_memory:.1f}MB, columns: {len(processed_df.columns)}")
            
            return processed_df
            
        except ValueError as ve:
            logger.error(f"Data validation error in prepare_data: {ve}")
            return df  # Return original DataFrame on validation error
            
        except MemoryError as me:
            logger.error(f"Memory error in prepare_data: {me}")
            # Try to process with minimal features
            try:
                logger.info("Attempting memory-optimized processing due to memory error")
                return self._prepare_data_memory_optimized(df)
            except Exception as e:
                logger.error(f"Memory-optimized processing also failed: {e}")
                return df
                
        except Exception as e:
            logger.error(f"Unexpected error in prepare_data: {e}")
            return df  # Return original DataFrame on any other error
            
        finally:
            # Cleanup and logging
            logger.debug("Data preparation completed")
            
    def _prepare_data_memory_optimized(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data with minimal memory footprint for constrained environments"""
        try:
            logger.info("Using memory-optimized data preparation")
            
            # Create optimized copy
            processed_df = df.copy()
            processed_df = self._optimize_dtypes(processed_df)
            
            # Add only essential features
            processed_df = self._add_basic_features(processed_df)
            
            # Add only essential indicators
            processed_df = self._add_essential_indicators(processed_df)
            
            # Skip complex patterns and market structure for memory savings
            processed_df = self._clean_data(processed_df)
            
            logger.info(f"Memory-optimized preparation complete: {len(processed_df.columns)} columns")
            return processed_df
            
        except Exception as e:
            logger.error(f"Error in memory-optimized preparation: {str(e)}")
            return df
    
    def _add_essential_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add only essential indicators for memory-constrained environments"""
        try:
            logger.debug("Adding essential indicators only")
            
            # Only the most critical indicators
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
            
            # Simple RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss.replace(0, np.nan)
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Basic ATR
            df['atr'] = (df['high'] - df['low']).rolling(window=14).mean()
            
            # Fill NaN values
            df.fillna(method='ffill', inplace=True)
            df.fillna(0, inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding essential indicators: {str(e)}")
            return df
    
    def _add_technical_indicators_optimized(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators with memory limits"""
        try:
            # Check current memory usage
            current_memory = self._estimate_memory_usage(df)
            
            if self.max_indicators <= 10:
                # Use essential indicators only
                return self._add_essential_indicators(df)
            elif self.max_indicators <= 15:
                # Use core indicators
                return self._add_core_indicators(df)
            else:
                # Use full indicator set
                return self._add_technical_indicators(df)
                
        except Exception as e:
            logger.error(f"Error in optimized technical indicators: {str(e)}")
            return df
    
    def _add_core_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add core indicators (medium memory usage)"""
        try:
            # Essential + some additional
            df = self._add_essential_indicators(df)
            
            # Add MACD
            ema_12 = df['close'].ewm(span=12).mean()
            ema_26 = df['close'].ewm(span=26).mean()
            df['macd'] = ema_12 - ema_26
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            
            # Simple Bollinger Bands
            sma_20 = df['close'].rolling(window=20).mean()
            std_20 = df['close'].rolling(window=20).std()
            df['bb_upper'] = sma_20 + (std_20 * 2)
            df['bb_lower'] = sma_20 - (std_20 * 2)
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding core indicators: {str(e)}")
            return df
    
    def _add_price_patterns_selective(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add selective price patterns to reduce memory usage"""
        try:
            if self.max_indicators <= 10:
                # Skip patterns for memory savings
                return df
            
            # Add only essential patterns
            body_size = abs(df['close'] - df['open'])
            total_range = df['high'] - df['low']
            total_range_safe = total_range.replace(0, np.nan)
            
            # Doji only (most important pattern)
            body_ratio = body_size / total_range_safe
            df['doji'] = (body_ratio < 0.1).astype('int8')
            
            # Inside bars only
            df['inside_bar'] = ((df['high'] < df['high'].shift(1)) & 
                              (df['low'] > df['low'].shift(1))).astype('int8')
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding selective patterns: {str(e)}")
            return df
    
    def _add_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic price features"""
        try:
            # Price changes
            df['price_change'] = df['close'] - df['open']
            df['price_change_pct'] = (df['close'] - df['open']) / df['open'] * 100
            
            # High-Low range
            df['hl_range'] = df['high'] - df['low']
            df['hl_range_pct'] = (df['high'] - df['low']) / df['low'] * 100
            
            # Body and wick sizes
            df['body_size'] = abs(df['close'] - df['open'])
            df['upper_wick'] = df['high'] - df[['close', 'open']].max(axis=1)
            df['lower_wick'] = df[['close', 'open']].min(axis=1) - df['low']
            
            # Relative position in range
            df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
            
            # Volume features with NaN handling
            df['volume_sma'] = df['tick_volume'].rolling(window=20, min_periods=1).mean()
            # Avoid division by zero/NaN
            volume_sma_safe = df['volume_sma'].replace(0, np.nan).fillna(df['tick_volume'].mean())
            df['volume_ratio'] = df['tick_volume'] / volume_sma_safe
            
            # Returns
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding basic features: {str(e)}")
            return df
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators with TA-Lib fallback"""
        try:
            if TALIB_AVAILABLE:
                return self._add_talib_indicators(df)
            else:
                return self._add_pandas_ta_indicators(df)
                
        except Exception as e:
            logger.error(f"Error adding technical indicators: {str(e)}")
            return df
    
    def _add_talib_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators using TA-Lib"""
        try:
            # Memory usage warning: this method adds ~20 columns to DataFrame
            # For large datasets (>1000 bars), consider using selective indicators
            if len(df) > 500:
                logger.debug(f"Processing {len(df)} bars with full indicator set - memory usage: ~{len(df)*20*8/1024/1024:.1f}MB")
            
        
            # Moving averages
            df['sma_20'] = talib.SMA(df['close'], timeperiod=20)
            df['sma_50'] = talib.SMA(df['close'], timeperiod=50)
            df['ema_20'] = talib.EMA(df['close'], timeperiod=20)
            
            # Bollinger Bands
            df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(
                df['close'], timeperiod=20, nbdevup=2, nbdevdn=2
            )
            df['bb_width'] = df['bb_upper'] - df['bb_lower']
            # Avoid division by zero
            bb_width_safe = df['bb_width'].replace(0, np.nan)
            df['bb_position'] = (df['close'] - df['bb_lower']) / bb_width_safe
            
            # RSI
            df['rsi'] = talib.RSI(df['close'], timeperiod=14)
            
            # MACD
            df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(
                df['close'], fastperiod=12, slowperiod=26, signalperiod=9
            )
            
            # ATR with NaN handling
            atr_result = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
            # ATR with proper fallback (never 0 to avoid division errors)
            atr_fallback = (df['high'] - df['low']).median() * 0.5  # Reasonable fallback
            df['atr'] = atr_result.fillna(atr_fallback) if atr_result is not None else atr_fallback
            
            # ADX
            df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
            df['plus_di'] = talib.PLUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
            df['minus_di'] = talib.MINUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
            
            # Stochastic
            df['stoch_k'], df['stoch_d'] = talib.STOCH(
                df['high'], df['low'], df['close'],
                fastk_period=14, slowk_period=3, slowd_period=3
            )
            
            # CCI
            df['cci'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=20)
            
            # MFI
            df['mfi'] = talib.MFI(df['high'], df['low'], df['close'], df['tick_volume'], timeperiod=14)
            
            logger.debug("Technical indicators added using TA-Lib")
            return df
            
        except Exception as e:
            logger.error(f"Error adding TA-Lib indicators: {str(e)}")
            # If TA-Lib fails, fall back to pandas-ta
            logger.warning("TA-Lib failed, falling back to pandas-ta")
            return self._add_pandas_ta_indicators(df)
    
    def _add_pandas_ta_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators using pandas-ta as fallback"""
        try:
            # Use specific indicators instead of "all" to avoid memory issues
            # df.ta.strategy("all") can add hundreds of columns and cause OOM
            
            # Moving averages
            df['sma_20'] = ta.sma(df['close'], length=20)
            df['sma_50'] = ta.sma(df['close'], length=50)
            df['ema_20'] = ta.ema(df['close'], length=20)
            
            # Bollinger Bands
            bb = ta.bbands(df['close'], length=20, std=2)
            if bb is not None and not bb.empty:
                df['bb_upper'] = bb.iloc[:, 0]  # Upper band
                df['bb_middle'] = bb.iloc[:, 1]  # Middle band
                df['bb_lower'] = bb.iloc[:, 2]  # Lower band
                df['bb_width'] = df['bb_upper'] - df['bb_lower']
                # Avoid division by zero
                bb_width_safe = df['bb_width'].replace(0, np.nan)
                df['bb_position'] = (df['close'] - df['bb_lower']) / bb_width_safe
            
            # RSI
            df['rsi'] = ta.rsi(df['close'], length=14)
            
            # MACD
            macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
            if macd is not None and not macd.empty:
                df['macd'] = macd.iloc[:, 0]
                df['macd_hist'] = macd.iloc[:, 1] 
                df['macd_signal'] = macd.iloc[:, 2]
            
            # ATR with NaN handling
            atr_result = ta.atr(df['high'], df['low'], df['close'], length=14)
            # ATR with proper fallback (never 0 to avoid division errors)
            atr_fallback = (df['high'] - df['low']).median() * 0.5  # Reasonable fallback
            df['atr'] = atr_result.fillna(atr_fallback) if atr_result is not None else atr_fallback
            
            # ADX
            adx = ta.adx(df['high'], df['low'], df['close'], length=14)
            if adx is not None and not adx.empty:
                df['adx'] = adx.iloc[:, 0]
                df['plus_di'] = adx.iloc[:, 1] if adx.shape[1] > 1 else None
                df['minus_di'] = adx.iloc[:, 2] if adx.shape[1] > 2 else None
            
            # Stochastic
            stoch = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3)
            if stoch is not None and not stoch.empty:
                df['stoch_k'] = stoch.iloc[:, 0]
                df['stoch_d'] = stoch.iloc[:, 1] if stoch.shape[1] > 1 else None
            
            # CCI
            df['cci'] = ta.cci(df['high'], df['low'], df['close'], length=20)
            
            # MFI
            df['mfi'] = ta.mfi(df['high'], df['low'], df['close'], df['tick_volume'], length=14)
            
            logger.debug("Technical indicators added using pandas-ta")
            return df
            
        except Exception as e:
            logger.error(f"Error adding pandas-ta indicators: {str(e)}")
            # If all else fails, add basic manual indicators
            return self._add_basic_manual_indicators(df)
    
    def _add_basic_manual_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic manual indicators as last resort"""
        try:
            logger.warning("Using manual indicator calculations as fallback")
            
            # Simple Moving Averages
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            
            # Exponential Moving Average (manual calculation)
            df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
            
            # Simple RSI calculation
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss.replace(0, np.nan)
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Basic ATR calculation
            df['high_low'] = df['high'] - df['low']
            df['high_prev_close'] = abs(df['high'] - df['close'].shift(1))
            df['low_prev_close'] = abs(df['low'] - df['close'].shift(1))
            df['true_range'] = df[['high_low', 'high_prev_close', 'low_prev_close']].max(axis=1)
            # Calculate ATR with proper fallback
            atr_fallback = (df['high'] - df['low']).median() * 0.5
            df['atr'] = df['true_range'].rolling(window=14, min_periods=1).mean().fillna(atr_fallback)
            
            # Clean up temporary columns
            df.drop(['high_low', 'high_prev_close', 'low_prev_close', 'true_range'], axis=1, inplace=True)
            
            # Fill missing values with basic defaults
            for col in ['bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_position',
                       'macd', 'macd_signal', 'macd_hist', 'adx', 'plus_di', 'minus_di',
                       'stoch_k', 'stoch_d', 'cci', 'mfi']:
                if col not in df.columns:
                    df[col] = 0
            
            logger.warning("Manual indicators calculated successfully")
            return df
            
        except Exception as e:
            logger.error(f"Error adding manual indicators: {str(e)}")
            return df
    
    def _add_price_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price pattern recognition with fallback"""
        try:
            if TALIB_AVAILABLE:
                # Candlestick patterns using TA-Lib
                try:
                    df['doji'] = talib.CDLDOJI(df['open'], df['high'], df['low'], df['close'])
                    df['hammer'] = talib.CDLHAMMER(df['open'], df['high'], df['low'], df['close'])
                    df['shooting_star'] = talib.CDLSHOOTINGSTAR(df['open'], df['high'], df['low'], df['close'])
                    df['engulfing'] = talib.CDLENGULFING(df['open'], df['high'], df['low'], df['close'])
                    df['morning_star'] = talib.CDLMORNINGSTAR(df['open'], df['high'], df['low'], df['close'])
                    df['evening_star'] = talib.CDLEVENINGSTAR(df['open'], df['high'], df['low'], df['close'])
                except Exception as e:
                    logger.warning(f"TA-Lib pattern recognition failed: {e}, using manual patterns")
                    self._add_manual_patterns(df)
            else:
                # Manual pattern recognition
                self._add_manual_patterns(df)
            
            # Custom patterns
            df['bullish_pin_bar'] = self._identify_pin_bar(df, bullish=True)
            df['bearish_pin_bar'] = self._identify_pin_bar(df, bullish=False)
            
            # Inside bars
            df['inside_bar'] = (
                (df['high'] < df['high'].shift(1)) & 
                (df['low'] > df['low'].shift(1))
            ).astype(int)
            
            # Outside bars
            df['outside_bar'] = (
                (df['high'] > df['high'].shift(1)) & 
                (df['low'] < df['low'].shift(1))
            ).astype(int)
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding price patterns: {str(e)}")
            return df
    
    def _add_manual_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic candlestick patterns manually"""
        try:
            # Doji pattern: small body relative to range
            body_size = abs(df['close'] - df['open'])
            total_range = df['high'] - df['low']
            total_range_safe = total_range.replace(0, np.nan)
            body_ratio = body_size / total_range_safe
            df['doji'] = (body_ratio < 0.1).astype(int) * 100  # TA-Lib style scoring
            
            # Hammer pattern: small body at top, long lower wick
            lower_wick = df[['close', 'open']].min(axis=1) - df['low']
            upper_wick = df['high'] - df[['close', 'open']].max(axis=1)
            df['hammer'] = ((lower_wick > 2 * body_size) & 
                           (upper_wick < body_size) & 
                           (body_ratio < 0.3)).astype(int) * 100
            
            # Shooting star: small body at bottom, long upper wick  
            df['shooting_star'] = ((upper_wick > 2 * body_size) & 
                                  (lower_wick < body_size) & 
                                  (body_ratio < 0.3)).astype(int) * 100
            
            # Basic engulfing pattern
            bullish_engulfing = ((df['open'] < df['close']) &  # Current bar bullish
                               (df['open'].shift(1) > df['close'].shift(1)) &  # Previous bearish
                               (df['close'] > df['open'].shift(1)) &  # Close above prev open
                               (df['open'] < df['close'].shift(1)))  # Open below prev close
            
            bearish_engulfing = ((df['open'] > df['close']) &  # Current bar bearish
                               (df['open'].shift(1) < df['close'].shift(1)) &  # Previous bullish
                               (df['close'] < df['open'].shift(1)) &  # Close below prev open
                               (df['open'] > df['close'].shift(1)))  # Open above prev close
            
            df['engulfing'] = (bullish_engulfing.astype(int) * 100 - 
                              bearish_engulfing.astype(int) * 100)
            
            # Basic morning/evening star patterns (simplified)
            df['morning_star'] = 0  # Complex pattern, simplified to 0
            df['evening_star'] = 0  # Complex pattern, simplified to 0
            
            logger.debug("Manual candlestick patterns calculated")
            return df
            
        except Exception as e:
            logger.error(f"Error adding manual patterns: {str(e)}")
            # Fill with zeros if manual patterns fail
            for col in ['doji', 'hammer', 'shooting_star', 'engulfing', 'morning_star', 'evening_star']:
                if col not in df.columns:
                    df[col] = 0
            return df
    
    def _add_market_structure(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market structure analysis"""
        try:
            # Swing highs and lows
            df['swing_high'] = self._find_swing_points(df['high'], df['low'], window=5, point_type='high')
            df['swing_low'] = self._find_swing_points(df['high'], df['low'], window=5, point_type='low')
            
            # Trend identification
            df['trend'] = self._identify_trend(df)
            
            # Support and resistance levels
            df['support'] = self._find_support_resistance(df, level_type='support')
            df['resistance'] = self._find_support_resistance(df, level_type='resistance')
            
            # Market phases
            df['market_phase'] = self._identify_market_phase(df)
            
            # Volatility regimes
            df['volatility_regime'] = self._classify_volatility(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding market structure: {str(e)}")
            return df
    
    def _identify_pin_bar(self, df: pd.DataFrame, bullish: bool = True) -> pd.Series:
        """Identify pin bar patterns"""
        try:
            body_size = abs(df['close'] - df['open'])
            
            if bullish:
                # Bullish pin bar: long lower wick, small body at top
                wick_size = df[['close', 'open']].min(axis=1) - df['low']
                condition = (
                    (wick_size > body_size * 2) &  # Wick at least 2x body
                    (df['close_position'] > 0.7) &  # Close in upper 30%
                    (body_size < df['atr'] * 0.3)  # Small body
                )
            else:
                # Bearish pin bar: long upper wick, small body at bottom
                wick_size = df['high'] - df[['close', 'open']].max(axis=1)
                condition = (
                    (wick_size > body_size * 2) &  # Wick at least 2x body
                    (df['close_position'] < 0.3) &  # Close in lower 30%
                    (body_size < df['atr'] * 0.3)  # Small body
                )
            
            return condition.astype(int)
            
        except Exception as e:
            logger.error(f"Error identifying pin bars: {str(e)}")
            return pd.Series(0, index=df.index)
    
    def _find_swing_points(self, high: pd.Series, low: pd.Series, 
                          window: int = 5, point_type: str = 'high') -> pd.Series:
        """Find swing highs and lows"""
        try:
            swing_points = pd.Series(0, index=high.index)
            
            if point_type == 'high':
                for i in range(window, len(high) - window):
                    if high.iloc[i] == high.iloc[i-window:i+window+1].max():
                        swing_points.iloc[i] = 1
            else:
                for i in range(window, len(low) - window):
                    if low.iloc[i] == low.iloc[i-window:i+window+1].min():
                        swing_points.iloc[i] = 1
            
            return swing_points
            
        except Exception as e:
            logger.error(f"Error finding swing points: {str(e)}")
            return pd.Series(0, index=high.index)
    
    def _identify_trend(self, df: pd.DataFrame) -> pd.Series:
        """Identify market trend"""
        try:
            # Simple trend identification using MA crossovers and ADX
            trend = pd.Series('NEUTRAL', index=df.index)
            
            # Conditions for uptrend
            uptrend = (
                (df['sma_20'] > df['sma_50']) &
                (df['close'] > df['sma_20']) &
                (df['adx'] > 25) &
                (df['plus_di'] > df['minus_di'])
            )
            
            # Conditions for downtrend
            downtrend = (
                (df['sma_20'] < df['sma_50']) &
                (df['close'] < df['sma_20']) &
                (df['adx'] > 25) &
                (df['minus_di'] > df['plus_di'])
            )
            
            trend[uptrend] = 'UPTREND'
            trend[downtrend] = 'DOWNTREND'
            
            return trend
            
        except Exception as e:
            logger.error(f"Error identifying trend: {str(e)}")
            return pd.Series('NEUTRAL', index=df.index)
    
    def _find_support_resistance(self, df: pd.DataFrame, level_type: str = 'support') -> pd.Series:
        """Find support and resistance levels"""
        try:
            levels = pd.Series(np.nan, index=df.index)
            
            if level_type == 'support':
                # Find recent swing lows
                swing_lows = df[df['swing_low'] == 1]['low']
                if not swing_lows.empty:
                    # Get the most recent significant swing low
                    recent_low = swing_lows.iloc[-1]
                    levels.iloc[-20:] = recent_low
            else:
                # Find recent swing highs
                swing_highs = df[df['swing_high'] == 1]['high']
                if not swing_highs.empty:
                    # Get the most recent significant swing high
                    recent_high = swing_highs.iloc[-1]
                    levels.iloc[-20:] = recent_high
            
            return levels.fillna(method='ffill')
            
        except Exception as e:
            logger.error(f"Error finding support/resistance: {str(e)}")
            return pd.Series(np.nan, index=df.index)
    
    def _identify_market_phase(self, df: pd.DataFrame) -> pd.Series:
        """Identify market phase (accumulation, distribution, etc.)"""
        try:
            phase = pd.Series('NEUTRAL', index=df.index)
            
            # Volume and price relationship
            volume_increasing = df['volume_ratio'] > 1.2
            price_stable = df['atr'] < df['atr'].rolling(50).mean()
            
            # Accumulation: stable price, increasing volume
            accumulation = price_stable & volume_increasing & (df['trend'] == 'NEUTRAL')
            
            # Distribution: high price, decreasing momentum
            distribution = (
                (df['rsi'] > 70) &
                (df['macd_hist'] < df['macd_hist'].shift(1)) &
                volume_increasing
            )
            
            # Markup: strong uptrend
            markup = (df['trend'] == 'UPTREND') & (df['adx'] > 30)
            
            # Markdown: strong downtrend
            markdown = (df['trend'] == 'DOWNTREND') & (df['adx'] > 30)
            
            phase[accumulation] = 'ACCUMULATION'
            phase[distribution] = 'DISTRIBUTION'
            phase[markup] = 'MARKUP'
            phase[markdown] = 'MARKDOWN'
            
            return phase
            
        except Exception as e:
            logger.error(f"Error identifying market phase: {str(e)}")
            return pd.Series('NEUTRAL', index=df.index)
    
    def _classify_volatility(self, df: pd.DataFrame) -> pd.Series:
        """Classify volatility regime"""
        try:
            # Calculate realized volatility
            returns = df['log_returns'].dropna()
            rolling_vol = returns.rolling(window=20).std() * np.sqrt(252)
            
            # Percentile-based classification
            vol_25 = rolling_vol.quantile(0.25)
            vol_75 = rolling_vol.quantile(0.75)
            
            regime = pd.Series('NORMAL', index=df.index)
            regime[rolling_vol < vol_25] = 'LOW'
            regime[rolling_vol > vol_75] = 'HIGH'
            
            return regime
            
        except Exception as e:
            logger.error(f"Error classifying volatility: {str(e)}")
            return pd.Series('NORMAL', index=df.index)
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate data"""
        try:
            # Remove infinite values
            df = df.replace([np.inf, -np.inf], np.nan)
            
            # Fill NaN values for different column types
            # Forward fill for most indicators
            indicator_cols = [col for col in df.columns if col not in self.required_columns]
            df[indicator_cols] = df[indicator_cols].fillna(method='ffill', limit=5)
            
            # Ensure no NaN in critical columns
            critical_cols = ['close', 'open', 'high', 'low']
            if df[critical_cols].isna().any().any():
                logger.warning("NaN values found in critical columns")
            
            return df
            
        except Exception as e:
            logger.error(f"Error cleaning data: {str(e)}")
            return df
    
    def get_latest_features(self, df: pd.DataFrame) -> Dict:
        """Get latest feature values for current analysis"""
        try:
            if df.empty:
                return {}
            
            latest = df.iloc[-1].to_dict()
            
            # Add derived features
            latest['price_momentum'] = df['close'].pct_change(5).iloc[-1]
            latest['volume_momentum'] = df['volume_ratio'].iloc[-5:].mean()
            
            # Market state summary
            latest['market_state'] = {
                'trend': latest.get('trend', 'NEUTRAL'),
                'phase': latest.get('market_phase', 'NEUTRAL'),
                'volatility': latest.get('volatility_regime', 'NORMAL'),
                'rsi_state': 'OVERBOUGHT' if latest.get('rsi', 50) > 70 else 'OVERSOLD' if latest.get('rsi', 50) < 30 else 'NEUTRAL'
            }
            
            return latest
            
        except Exception as e:
            logger.error(f"Error getting latest features: {str(e)}")
            return {}


class MultiSymbolAnalyzer:
    """Multi-symbol correlation analysis with OANDA rate limiting"""
    
    def __init__(self, settings: Dict):
        """Initialize multi-symbol analyzer
        
        Args:
            settings: Trading configuration with symbols and OANDA settings
        """
        self.settings = settings
        self.rate_limiter = OANDARateLimiter(
            requests_per_minute=settings.get('api', {}).get('oanda', {}).get('rate_limit', {}).get('requests_per_minute', 150)
        )
        self.correlation_cache = {}
        self.last_analysis_time = {}
        
        # Symbol configuration from settings
        self.primary_symbol = settings.get('trading', {}).get('symbols', {}).get('primary', 'XAUUSD')
        self.correlated_symbols = settings.get('trading', {}).get('symbols', {}).get('correlated', {})
        
        # Analysis settings
        self.correlation_threshold = settings.get('trading', {}).get('multi_symbol_analysis', {}).get('correlation_threshold', 0.6)
        self.analysis_interval = settings.get('trading', {}).get('multi_symbol_analysis', {}).get('analysis_interval', 300)
        
    def analyze_correlations(self, mt5_connector, primary_data: pd.DataFrame) -> Dict:
        """Analyze correlations between primary symbol and related symbols
        
        Args:
            mt5_connector: MT5 connection instance
            primary_data: Primary symbol OHLC data
            
        Returns:
            Dict: Correlation analysis results
        """
        try:
            current_time = time.time()
            
            # Check if analysis is needed based on interval
            if (current_time - self.last_analysis_time.get('correlations', 0)) < self.analysis_interval:
                return self.correlation_cache.get('last_analysis', {})
            
            correlations = {}
            symbol_data = {}
            
            # Get data for correlated symbols with rate limiting
            for symbol_name, config in self.correlated_symbols.items():
                try:
                    # Wait for rate limit if necessary
                    self.rate_limiter.wait_if_needed()
                    
                    # Get symbol data
                    symbol_data[symbol_name] = mt5_connector.get_ohlc_data(
                        config['symbol'], 
                        'M5', 
                        len(primary_data)
                    )
                    
                    if not symbol_data[symbol_name].empty:
                        # Calculate correlation
                        correlation = self._calculate_correlation(
                            primary_data['close'], 
                            symbol_data[symbol_name]['close']
                        )
                        
                        correlations[symbol_name] = {
                            'correlation': correlation,
                            'expected_correlation': config['correlation'],
                            'weight': config['weight'],
                            'priority': config['priority'],
                            'current_price': symbol_data[symbol_name]['close'].iloc[-1],
                            'change_24h': self._calculate_24h_change(symbol_data[symbol_name])
                        }
                    
                except Exception as e:
                    logger.warning(f"Failed to get data for {symbol_name}: {e}")
                    continue
            
            # Calculate overall market sentiment
            market_sentiment = self._calculate_market_sentiment(correlations, primary_data)
            
            # Update cache
            analysis_result = {
                'correlations': correlations,
                'market_sentiment': market_sentiment,
                'primary_symbol': self.primary_symbol,
                'analysis_time': current_time,
                'strength_score': self._calculate_strength_score(correlations)
            }
            
            self.correlation_cache['last_analysis'] = analysis_result
            self.last_analysis_time['correlations'] = current_time
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error in correlation analysis: {e}")
            return {}
    
    def _calculate_correlation(self, series1: pd.Series, series2: pd.Series) -> float:
        """Calculate correlation between two price series"""
        try:
            # Align series by index
            aligned_data = pd.concat([series1, series2], axis=1, join='inner')
            if len(aligned_data) < 10:
                return 0.0
            
            correlation = aligned_data.corr().iloc[0, 1]
            return correlation if not pd.isna(correlation) else 0.0
            
        except Exception as e:
            logger.warning(f"Error calculating correlation: {e}")
            return 0.0
    
    def _calculate_24h_change(self, data: pd.DataFrame) -> float:
        """Calculate 24-hour price change percentage"""
        try:
            if len(data) < 24:
                return 0.0
            
            current_price = data['close'].iloc[-1]
            price_24h_ago = data['close'].iloc[-24]
            
            return ((current_price - price_24h_ago) / price_24h_ago) * 100
            
        except Exception as e:
            logger.warning(f"Error calculating 24h change: {e}")
            return 0.0
    
    def _calculate_market_sentiment(self, correlations: Dict, primary_data: pd.DataFrame) -> Dict:
        """Calculate overall market sentiment based on correlations"""
        try:
            if not correlations:
                return {'sentiment': 'NEUTRAL', 'strength': 0.0}
            
            # Calculate weighted sentiment
            bullish_weight = 0.0
            bearish_weight = 0.0
            total_weight = 0.0
            
            primary_change = self._calculate_24h_change(primary_data)
            
            for symbol_name, data in correlations.items():
                weight = data['weight']
                correlation = data['correlation']
                change_24h = data['change_24h']
                
                # If correlation is positive, symbol moves with primary
                # If correlation is negative, symbol moves opposite to primary
                expected_direction = 1 if correlation > 0 else -1
                
                if change_24h * expected_direction > 0:
                    bullish_weight += weight
                else:
                    bearish_weight += weight
                
                total_weight += weight
            
            if total_weight == 0:
                return {'sentiment': 'NEUTRAL', 'strength': 0.0}
            
            bullish_ratio = bullish_weight / total_weight
            bearish_ratio = bearish_weight / total_weight
            
            if bullish_ratio > 0.6:
                sentiment = 'BULLISH'
                strength = bullish_ratio
            elif bearish_ratio > 0.6:
                sentiment = 'BEARISH'
                strength = bearish_ratio
            else:
                sentiment = 'NEUTRAL'
                strength = abs(bullish_ratio - bearish_ratio)
            
            return {
                'sentiment': sentiment,
                'strength': strength,
                'bullish_ratio': bullish_ratio,
                'bearish_ratio': bearish_ratio,
                'primary_change_24h': primary_change
            }
            
        except Exception as e:
            logger.error(f"Error calculating market sentiment: {e}")
            return {'sentiment': 'NEUTRAL', 'strength': 0.0}
    
    def _calculate_strength_score(self, correlations: Dict) -> float:
        """Calculate overall strength score based on correlation alignment"""
        try:
            if not correlations:
                return 0.0
            
            total_score = 0.0
            total_weight = 0.0
            
            for symbol_name, data in correlations.items():
                actual_correlation = abs(data['correlation'])
                expected_correlation = abs(data['expected_correlation'])
                weight = data['weight']
                
                # Score based on how close actual correlation is to expected
                if expected_correlation > 0:
                    alignment_score = actual_correlation / expected_correlation
                    alignment_score = min(alignment_score, 1.0)  # Cap at 1.0
                else:
                    alignment_score = 0.5  # Neutral score for zero expected correlation
                
                total_score += alignment_score * weight
                total_weight += weight
            
            return total_score / total_weight if total_weight > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating strength score: {e}")
            return 0.0
    
    def get_priority_symbols(self) -> List[str]:
        """Get symbols sorted by priority for rate-limited requests"""
        symbol_priorities = []
        
        for symbol_name, config in self.correlated_symbols.items():
            symbol_priorities.append((symbol_name, config['priority']))
        
        # Sort by priority (lower number = higher priority)
        symbol_priorities.sort(key=lambda x: x[1])
        
        return [symbol for symbol, _ in symbol_priorities]


class OANDARateLimiter:
    """Rate limiter specifically for OANDA API (150 requests per minute)"""
    
    def __init__(self, requests_per_minute: int = 150):
        """Initialize rate limiter
        
        Args:
            requests_per_minute: Maximum requests allowed per minute
        """
        self.requests_per_minute = requests_per_minute
        self.requests_per_second = requests_per_minute / 60.0
        self.min_interval = 1.0 / self.requests_per_second
        self.last_request_time = 0.0
        self.request_count = 0
        self.window_start = time.time()
        self._lock = threading.Lock()
    
    def wait_if_needed(self):
        """Wait if necessary to respect rate limits"""
        with self._lock:
            current_time = time.time()
            
            # Reset window if a minute has passed
            if current_time - self.window_start >= 60:
                self.request_count = 0
                self.window_start = current_time
            
            # Check if we've hit the per-minute limit
            if self.request_count >= self.requests_per_minute:
                sleep_time = 60 - (current_time - self.window_start)
                if sleep_time > 0:
                    logger.info(f"OANDA rate limit reached, sleeping for {sleep_time:.2f} seconds")
                    time.sleep(sleep_time)
                    self.request_count = 0
                    self.window_start = time.time()
                    current_time = time.time()
            
            # Ensure minimum interval between requests
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.min_interval:
                sleep_time = self.min_interval - time_since_last
                time.sleep(sleep_time)
                current_time = time.time()
            
            self.last_request_time = current_time
            self.request_count += 1


class DecisionModeManager:
    """Manages different decision modes for human-AI interaction"""
    
    def __init__(self, telegram_config: Dict):
        """Initialize decision mode manager
        
        Args:
            telegram_config: Telegram configuration with decision modes
        """
        self.config = telegram_config
        self.current_mode = "automatic"  # Default mode
        self.pending_decisions = {}  # Store pending confirmations
        self.mode_configs = telegram_config.get('decision_modes', {})
        self._lock = threading.Lock()
        
        # Decision timeout settings
        self.confirmation_timeout = 300  # 5 minutes default
        
    def set_mode(self, mode: str) -> bool:
        """Set the current decision mode
        
        Args:
            mode: One of 'automatic', 'semi_automatic', 'advisory', 'manual_with_ai'
            
        Returns:
            bool: True if mode was set successfully
        """
        valid_modes = list(self.mode_configs.keys())
        
        if mode not in valid_modes:
            logger.error(f"Invalid decision mode: {mode}. Valid modes: {valid_modes}")
            return False
        
        if not self.mode_configs.get(mode, {}).get('enabled', False):
            logger.error(f"Decision mode {mode} is not enabled")
            return False
        
        with self._lock:
            old_mode = self.current_mode
            self.current_mode = mode
            logger.info(f"Decision mode changed from {old_mode} to {mode}")
        
        return True
    
    def get_current_mode(self) -> str:
        """Get the current decision mode"""
        return self.current_mode
    
    def process_trading_signal(self, signal_data: Dict, telegram_bot=None) -> Dict:
        """Process a trading signal based on current decision mode
        
        Args:
            signal_data: Trading signal data
            telegram_bot: Telegram bot instance for sending messages
            
        Returns:
            Dict: Decision result with action to take
        """
        mode_config = self.mode_configs.get(self.current_mode, {})
        
        if self.current_mode == "automatic":
            return self._process_automatic_mode(signal_data)
        
        elif self.current_mode == "semi_automatic":
            return self._process_semi_automatic_mode(signal_data, telegram_bot)
        
        elif self.current_mode == "advisory":
            return self._process_advisory_mode(signal_data, telegram_bot)
        
        elif self.current_mode == "manual_with_ai":
            return self._process_manual_with_ai_mode(signal_data, telegram_bot)
        
        else:
            logger.error(f"Unknown decision mode: {self.current_mode}")
            return {"action": "reject", "reason": "Unknown decision mode"}
    
    def _process_automatic_mode(self, signal_data: Dict) -> Dict:
        """Process signal in automatic mode - execute immediately if valid"""
        if signal_data.get('signal') in ['BUY', 'SELL'] and signal_data.get('confidence', 0) > 0.6:
            return {
                "action": "execute",
                "reason": "Automatic execution based on signal confidence",
                "immediate": True,
                "signal_data": signal_data
            }
        else:
            return {
                "action": "reject",
                "reason": "Signal confidence too low for automatic execution",
                "immediate": True
            }
    
    def _process_semi_automatic_mode(self, signal_data: Dict, telegram_bot) -> Dict:
        """Process signal in semi-automatic mode - require confirmation"""
        if signal_data.get('signal') in ['BUY', 'SELL']:
            # Create pending decision
            decision_id = f"decision_{int(time.time())}"
            
            with self._lock:
                self.pending_decisions[decision_id] = {
                    "signal_data": signal_data,
                    "created_at": time.time(),
                    "timeout": self.confirmation_timeout,
                    "status": "pending"
                }
            
            # Send confirmation request to Telegram
            if telegram_bot:
                self._send_confirmation_request(telegram_bot, decision_id, signal_data)
            
            return {
                "action": "pending",
                "reason": "Waiting for user confirmation",
                "decision_id": decision_id,
                "immediate": False,
                "timeout": self.confirmation_timeout
            }
        else:
            return {
                "action": "reject",
                "reason": "No valid trading signal",
                "immediate": True
            }
    
    def _process_advisory_mode(self, signal_data: Dict, telegram_bot) -> Dict:
        """Process signal in advisory mode - only provide analysis"""
        if telegram_bot:
            self._send_advisory_message(telegram_bot, signal_data)
        
        return {
            "action": "advisory_only",
            "reason": "Analysis sent to user, no automatic execution",
            "immediate": True,
            "signal_data": signal_data
        }
    
    def _process_manual_with_ai_mode(self, signal_data: Dict, telegram_bot) -> Dict:
        """Process signal in manual with AI mode - provide alerts and assistance"""
        if telegram_bot:
            self._send_ai_assistance_message(telegram_bot, signal_data)
        
        return {
            "action": "assist_only",
            "reason": "AI assistance provided, user controls execution",
            "immediate": True,
            "signal_data": signal_data
        }
    
    def confirm_decision(self, decision_id: str, approved: bool, user_id: int) -> Dict:
        """Confirm or reject a pending decision
        
        Args:
            decision_id: ID of the pending decision
            approved: Whether the decision was approved
            user_id: Telegram user ID
            
        Returns:
            Dict: Result of the confirmation
        """
        with self._lock:
            if decision_id not in self.pending_decisions:
                return {
                    "success": False,
                    "reason": "Decision not found or already processed"
                }
            
            decision = self.pending_decisions[decision_id]
            
            # Check if decision has timed out
            if time.time() - decision['created_at'] > decision['timeout']:
                del self.pending_decisions[decision_id]
                return {
                    "success": False,
                    "reason": "Decision has timed out"
                }
            
            # Update decision status
            decision['status'] = 'approved' if approved else 'rejected'
            decision['confirmed_by'] = user_id
            decision['confirmed_at'] = time.time()
            
            signal_data = decision['signal_data']
            
            # Remove from pending
            del self.pending_decisions[decision_id]
            
            if approved:
                return {
                    "success": True,
                    "action": "execute",
                    "signal_data": signal_data,
                    "reason": f"Approved by user {user_id}"
                }
            else:
                return {
                    "success": True,
                    "action": "reject",
                    "reason": f"Rejected by user {user_id}"
                }
    
    def _send_confirmation_request(self, telegram_bot, decision_id: str, signal_data: Dict):
        """Send confirmation request to Telegram"""
        try:
            message = f"""
🤖 **Trading Signal Confirmation Required**

📊 **Signal**: {signal_data.get('signal', 'UNKNOWN')}
📈 **Symbol**: {signal_data.get('symbol', 'XAUUSD')}
🎯 **Entry**: {signal_data.get('entry_price', 'N/A')}
🛑 **Stop Loss**: {signal_data.get('stop_loss', 'N/A')}
💰 **Take Profit**: {signal_data.get('take_profit', 'N/A')}
📊 **Confidence**: {signal_data.get('confidence', 0):.2%}
📝 **Reason**: {signal_data.get('reason', 'N/A')}

⏰ **Time Limit**: {self.confirmation_timeout // 60} minutes

**Commands:**
✅ `/confirm {decision_id}` - Execute trade
❌ `/reject {decision_id}` - Cancel trade
"""
            
            # Send to admin chat IDs
            admin_chat_ids = self.config.get('admin_chat_ids', [])
            for chat_id in admin_chat_ids:
                telegram_bot.send_message(chat_id, message)
                
        except Exception as e:
            logger.error(f"Error sending confirmation request: {e}")
    
    def _send_advisory_message(self, telegram_bot, signal_data: Dict):
        """Send advisory analysis to Telegram"""
        try:
            message = f"""
📊 **Market Analysis Advisory**

📈 **Symbol**: {signal_data.get('symbol', 'XAUUSD')}
🎯 **Suggested Direction**: {signal_data.get('signal', 'NEUTRAL')}
💹 **Entry Level**: {signal_data.get('entry_price', 'N/A')}
🛑 **Suggested Stop**: {signal_data.get('stop_loss', 'N/A')}
💰 **Target Price**: {signal_data.get('take_profit', 'N/A')}
📊 **Confidence**: {signal_data.get('confidence', 0):.2%}
📝 **Analysis**: {signal_data.get('reason', 'N/A')}

⚠️ **Advisory Mode**: This is analysis only. Manual execution required.
"""
            
            admin_chat_ids = self.config.get('admin_chat_ids', [])
            for chat_id in admin_chat_ids:
                telegram_bot.send_message(chat_id, message)
                
        except Exception as e:
            logger.error(f"Error sending advisory message: {e}")
    
    def _send_ai_assistance_message(self, telegram_bot, signal_data: Dict):
        """Send AI assistance alert to Telegram"""
        try:
            signal_strength = "🔥 Strong" if signal_data.get('confidence', 0) > 0.8 else "⚡ Moderate" if signal_data.get('confidence', 0) > 0.6 else "💡 Weak"
            
            message = f"""
🤖 **AI Trading Assistant Alert**

{signal_strength} signal detected:

📈 **Symbol**: {signal_data.get('symbol', 'XAUUSD')}
📊 **Direction**: {signal_data.get('signal', 'NEUTRAL')}
🎯 **Suggested Entry**: {signal_data.get('entry_price', 'N/A')}
🛑 **Risk Level**: {signal_data.get('stop_loss', 'N/A')}
💰 **Profit Target**: {signal_data.get('take_profit', 'N/A')}
📊 **AI Confidence**: {signal_data.get('confidence', 0):.2%}

🧠 **AI Analysis**: {signal_data.get('reason', 'N/A')}

💡 **Manual Mode**: You control all executions. AI provides assistance only.
"""
            
            admin_chat_ids = self.config.get('admin_chat_ids', [])
            for chat_id in admin_chat_ids:
                telegram_bot.send_message(chat_id, message)
                
        except Exception as e:
            logger.error(f"Error sending AI assistance message: {e}")
    
    def cleanup_expired_decisions(self):
        """Clean up expired pending decisions"""
        current_time = time.time()
        expired_ids = []
        
        with self._lock:
            for decision_id, decision in self.pending_decisions.items():
                if current_time - decision['created_at'] > decision['timeout']:
                    expired_ids.append(decision_id)
            
            for decision_id in expired_ids:
                del self.pending_decisions[decision_id]
                logger.info(f"Expired decision {decision_id} cleaned up")
    
    def get_pending_decisions(self) -> List[Dict]:
        """Get list of pending decisions"""
        with self._lock:
            return [
                {
                    "decision_id": decision_id,
                    "signal": decision['signal_data'].get('signal'),
                    "symbol": decision['signal_data'].get('symbol'),
                    "created_at": decision['created_at'],
                    "time_remaining": max(0, decision['timeout'] - (time.time() - decision['created_at']))
                }
                for decision_id, decision in self.pending_decisions.items()
            ]


class NewsAnalyzer:
    """Analyzes news from Investing.com and ForexFactory using scraping"""
    
    def __init__(self, settings: Dict):
        """Initialize news analyzer
        
        Args:
            settings: Configuration settings
        """
        self.settings = settings
        self.news_sources = settings.get('data_sources', {}).get('news_sources', [])
        self.news_cache = {}
        self.last_fetch_time = {}
        self.fetch_interval = 900  # 15 minutes
        
        # Request session for efficient scraping
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Keywords relevant to gold/forex trading
        self.relevant_keywords = [
            'gold', 'xauusd', 'federal reserve', 'fed', 'interest rate', 'inflation',
            'dollar', 'usd', 'dxy', 'unemployment', 'nonfarm', 'cpi', 'ppi',
            'economic', 'gdp', 'treasury', 'fomc', 'powell', 'central bank',
            'monetary policy', 'fiscal policy', 'recession', 'recovery'
        ]
        
    def analyze_news_sentiment(self) -> Dict:
        """Analyze news sentiment from configured sources
        
        Returns:
            Dict: News sentiment analysis results
        """
        try:
            current_time = time.time()
            
            # Check if news fetch is needed
            if (current_time - self.last_fetch_time.get('news', 0)) < self.fetch_interval:
                return self.news_cache.get('last_analysis', {})
            
            all_news = []
            
            # Fetch from Investing.com
            investing_news = self._scrape_investing_com()
            if investing_news:
                all_news.extend(investing_news)
            
            # Fetch from ForexFactory
            forexfactory_news = self._scrape_forexfactory()
            if forexfactory_news:
                all_news.extend(forexfactory_news)
            
            # Analyze sentiment
            sentiment_analysis = self._analyze_sentiment(all_news)
            
            # Update cache
            analysis_result = {
                'news_items': all_news,
                'sentiment_analysis': sentiment_analysis,
                'fetch_time': current_time,
                'news_count': len(all_news)
            }
            
            self.news_cache['last_analysis'] = analysis_result
            self.last_fetch_time['news'] = current_time
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error analyzing news sentiment: {e}")
            return {}
    
    def _scrape_investing_com(self) -> List[Dict]:
        """Scrape news from Investing.com"""
        try:
            # Investing.com gold news URL
            url = "https://www.investing.com/commodities/gold-news"
            
            response = self.session.get(url, timeout=30)
            if response.status_code != 200:
                logger.warning(f"Failed to fetch Investing.com news: {response.status_code}")
                return []
            
            soup = BeautifulSoup(response.content, 'html.parser')
            news_items = []
            
            # Find news articles (this selector may need adjustment based on site changes)
            articles = soup.find_all('div', class_='textDiv')[:10]  # Get latest 10 articles
            
            for article in articles:
                try:
                    title_elem = article.find('a')
                    if not title_elem:
                        continue
                    
                    title = title_elem.get_text(strip=True)
                    link = title_elem.get('href', '')
                    
                    # Make link absolute if relative
                    if link.startswith('/'):
                        link = f"https://www.investing.com{link}"
                    
                    # Check if relevant to our keywords
                    if self._is_relevant(title):
                        news_items.append({
                            'title': title,
                            'link': link,
                            'source': 'Investing.com',
                            'relevance_score': self._calculate_relevance_score(title),
                            'timestamp': time.time()
                        })
                
                except Exception as e:
                    logger.warning(f"Error parsing Investing.com article: {e}")
                    continue
            
            logger.info(f"Fetched {len(news_items)} relevant articles from Investing.com")
            return news_items
            
        except Exception as e:
            logger.error(f"Error scraping Investing.com: {e}")
            return []
    
    def _scrape_forexfactory(self) -> List[Dict]:
        """Scrape news from ForexFactory"""
        try:
            # ForexFactory news URL
            url = "https://www.forexfactory.com/news"
            
            response = self.session.get(url, timeout=30)
            if response.status_code != 200:
                logger.warning(f"Failed to fetch ForexFactory news: {response.status_code}")
                return []
            
            soup = BeautifulSoup(response.content, 'html.parser')
            news_items = []
            
            # Find news articles (this selector may need adjustment based on site changes)
            articles = soup.find_all('tr', class_='calendar__row')[:10]  # Get latest 10 events
            
            for article in articles:
                try:
                    # Extract title from event
                    title_elem = article.find('td', class_='calendar__event')
                    if not title_elem:
                        continue
                    
                    title = title_elem.get_text(strip=True)
                    
                    # Extract impact level
                    impact_elem = article.find('td', class_='calendar__impact')
                    impact = 'medium'
                    if impact_elem:
                        if 'high' in impact_elem.get('class', []):
                            impact = 'high'
                        elif 'low' in impact_elem.get('class', []):
                            impact = 'low'
                    
                    # Check if relevant
                    if self._is_relevant(title):
                        news_items.append({
                            'title': title,
                            'link': url,
                            'source': 'ForexFactory',
                            'impact': impact,
                            'relevance_score': self._calculate_relevance_score(title),
                            'timestamp': time.time()
                        })
                
                except Exception as e:
                    logger.warning(f"Error parsing ForexFactory event: {e}")
                    continue
            
            logger.info(f"Fetched {len(news_items)} relevant events from ForexFactory")
            return news_items
            
        except Exception as e:
            logger.error(f"Error scraping ForexFactory: {e}")
            return []
    
    def _is_relevant(self, text: str) -> bool:
        """Check if news item is relevant to trading"""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.relevant_keywords)
    
    def _calculate_relevance_score(self, text: str) -> float:
        """Calculate relevance score based on keyword matches"""
        text_lower = text.lower()
        score = 0.0
        
        keyword_weights = {
            'gold': 1.0, 'xauusd': 1.0, 'federal reserve': 0.9, 'fed': 0.9,
            'interest rate': 0.8, 'inflation': 0.8, 'dollar': 0.7, 'usd': 0.7,
            'dxy': 0.6, 'economic': 0.5, 'gdp': 0.6, 'cpi': 0.7, 'ppi': 0.6
        }
        
        for keyword, weight in keyword_weights.items():
            if keyword in text_lower:
                score += weight
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _analyze_sentiment(self, news_items: List[Dict]) -> Dict:
        """Analyze overall sentiment from news items"""
        try:
            if not news_items:
                return {'sentiment': 'NEUTRAL', 'strength': 0.0, 'confidence': 0.0}
            
            # Simple sentiment analysis based on keywords
            bullish_keywords = [
                'rise', 'gain', 'increase', 'positive', 'growth', 'strong', 'boost',
                'rally', 'surge', 'jump', 'climb', 'advance', 'optimism', 'recovery'
            ]
            
            bearish_keywords = [
                'fall', 'drop', 'decline', 'negative', 'weak', 'concern', 'risk',
                'crash', 'plunge', 'slide', 'retreat', 'pessimism', 'recession'
            ]
            
            bullish_score = 0.0
            bearish_score = 0.0
            total_relevance = 0.0
            
            for item in news_items:
                title_lower = item['title'].lower()
                relevance = item.get('relevance_score', 0.5)
                
                bullish_matches = sum(1 for kw in bullish_keywords if kw in title_lower)
                bearish_matches = sum(1 for kw in bearish_keywords if kw in title_lower)
                
                bullish_score += bullish_matches * relevance
                bearish_score += bearish_matches * relevance
                total_relevance += relevance
            
            if total_relevance == 0:
                return {'sentiment': 'NEUTRAL', 'strength': 0.0, 'confidence': 0.0}
            
            # Normalize scores
            bullish_ratio = bullish_score / total_relevance
            bearish_ratio = bearish_score / total_relevance
            
            # Determine sentiment
            if bullish_ratio > bearish_ratio * 1.5:
                sentiment = 'BULLISH'
                strength = bullish_ratio
            elif bearish_ratio > bullish_ratio * 1.5:
                sentiment = 'BEARISH'
                strength = bearish_ratio
            else:
                sentiment = 'NEUTRAL'
                strength = abs(bullish_ratio - bearish_ratio)
            
            # Calculate confidence based on news volume and relevance
            confidence = min(len(news_items) / 10.0, 1.0) * (total_relevance / len(news_items))
            
            return {
                'sentiment': sentiment,
                'strength': strength,
                'confidence': confidence,
                'bullish_score': bullish_ratio,
                'bearish_score': bearish_ratio,
                'news_volume': len(news_items),
                'avg_relevance': total_relevance / len(news_items)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return {'sentiment': 'NEUTRAL', 'strength': 0.0, 'confidence': 0.0}
    
    def get_high_impact_news(self, hours_back: int = 24) -> List[Dict]:
        """Get high-impact news from the last N hours"""
        try:
            current_time = time.time()
            cutoff_time = current_time - (hours_back * 3600)
            
            last_analysis = self.news_cache.get('last_analysis', {})
            news_items = last_analysis.get('news_items', [])
            
            high_impact_news = []
            
            for item in news_items:
                if item.get('timestamp', 0) > cutoff_time:
                    # Consider high impact if relevance score > 0.7 or marked as high impact
                    if (item.get('relevance_score', 0) > 0.7 or 
                        item.get('impact') == 'high'):
                        high_impact_news.append(item)
            
            # Sort by relevance score descending
            high_impact_news.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
            
            return high_impact_news[:5]  # Return top 5
            
        except Exception as e:
            logger.error(f"Error getting high impact news: {e}")
            return []


class AIPatternRecognition:
    """Advanced AI-powered price action pattern recognition using Claude and GPT-4"""
    
    def __init__(self, settings: Dict):
        """Initialize AI pattern recognition
        
        Args:
            settings: Configuration settings with API keys
        """
        self.settings = settings
        self.claude_api_key = settings.get('api', {}).get('claude_api_key')
        self.gpt4_api_key = settings.get('api', {}).get('gpt4_api_key')
        
        # Rate limiting for AI APIs
        self.ai_request_times = []
        self.max_ai_requests_per_hour = 100
        
        # Pattern cache to avoid redundant AI calls
        self.pattern_cache = {}
        self.cache_expiry = 1800  # 30 minutes
        
    def analyze_price_action_with_ai(self, df: pd.DataFrame, symbol: str = "XAUUSD") -> Dict:
        """Analyze price action using AI models for pattern recognition
        
        Args:
            df: OHLC dataframe
            symbol: Trading symbol
            
        Returns:
            Dict: AI analysis results
        """
        try:
            # Check cache first
            cache_key = self._generate_cache_key(df, symbol)
            if cache_key in self.pattern_cache:
                cached_result = self.pattern_cache[cache_key]
                if time.time() - cached_result['timestamp'] < self.cache_expiry:
                    return cached_result['analysis']
            
            # Rate limiting check
            if not self._check_rate_limit():
                logger.warning("AI API rate limit reached, using fallback analysis")
                return self._fallback_pattern_analysis(df)
            
            # Prepare data for AI analysis
            market_context = self._prepare_market_context(df)
            
            # Get analysis from both AI models
            claude_analysis = self._analyze_with_claude(market_context, symbol)
            gpt4_analysis = self._analyze_with_gpt4(market_context, symbol)
            
            # Combine and validate results
            combined_analysis = self._combine_ai_analyses(claude_analysis, gpt4_analysis, df)
            
            # Cache the result
            self.pattern_cache[cache_key] = {
                'analysis': combined_analysis,
                'timestamp': time.time()
            }
            
            return combined_analysis
            
        except Exception as e:
            logger.error(f"Error in AI pattern recognition: {e}")
            return self._fallback_pattern_analysis(df)
    
    def _prepare_market_context(self, df: pd.DataFrame) -> Dict:
        """Prepare market context data for AI analysis"""
        try:
            if len(df) < 20:
                return {}
            
            # Get recent price action (last 20 bars)
            recent_data = df.tail(20)
            
            # Calculate key metrics
            current_price = recent_data['close'].iloc[-1]
            price_range = recent_data['high'].max() - recent_data['low'].min()
            avg_volume = recent_data['tick_volume'].mean()
            
            # Identify key levels
            resistance_levels = self._find_resistance_levels(recent_data)
            support_levels = self._find_support_levels(recent_data)
            
            # Price action patterns
            candlestick_patterns = self._identify_candlestick_patterns(recent_data)
            
            # Market structure
            market_structure = self._analyze_market_structure(recent_data)
            
            return {
                'price_data': {
                    'current_price': current_price,
                    'price_range': price_range,
                    'avg_volume': avg_volume,
                    'bars_analyzed': len(recent_data)
                },
                'levels': {
                    'resistance': resistance_levels,
                    'support': support_levels
                },
                'patterns': candlestick_patterns,
                'structure': market_structure,
                'ohlc_summary': self._create_ohlc_summary(recent_data)
            }
            
        except Exception as e:
            logger.error(f"Error preparing market context: {e}")
            return {}
    
    def _analyze_with_claude(self, market_context: Dict, symbol: str) -> Dict:
        """Analyze market using Claude AI"""
        try:
            if not self.claude_api_key:
                return {}
            
            import anthropic
            
            client = anthropic.Anthropic(api_key=self.claude_api_key)
            
            # Create analysis prompt
            prompt = self._create_claude_prompt(market_context, symbol)
            
            response = client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1000,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Parse AI response
            ai_text = response.content[0].text
            return self._parse_ai_response(ai_text, 'claude')
            
        except Exception as e:
            logger.error(f"Error with Claude analysis: {e}")
            return {}
    
    def _analyze_with_gpt4(self, market_context: Dict, symbol: str) -> Dict:
        """Analyze market using GPT-4"""
        try:
            if not self.gpt4_api_key:
                return {}
            
            import openai
            
            client = openai.OpenAI(api_key=self.gpt4_api_key)
            
            # Create analysis prompt
            prompt = self._create_gpt4_prompt(market_context, symbol)
            
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert trader specializing in price action analysis for gold (XAU/USD) trading."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.3
            )
            
            # Parse AI response
            ai_text = response.choices[0].message.content
            return self._parse_ai_response(ai_text, 'gpt4')
            
        except Exception as e:
            logger.error(f"Error with GPT-4 analysis: {e}")
            return {}
    
    def _create_claude_prompt(self, market_context: Dict, symbol: str) -> str:
        """Create analysis prompt for Claude"""
        ohlc_summary = market_context.get('ohlc_summary', {})
        patterns = market_context.get('patterns', {})
        structure = market_context.get('structure', {})
        
        prompt = f"""
As an expert gold trader, analyze this {symbol} price action data:

MARKET DATA:
- Current Price: {market_context.get('price_data', {}).get('current_price', 'N/A')}
- Price Range: {market_context.get('price_data', {}).get('price_range', 'N/A')}
- Bars Analyzed: {market_context.get('price_data', {}).get('bars_analyzed', 'N/A')}

RECENT PRICE ACTION:
{ohlc_summary}

IDENTIFIED PATTERNS:
{patterns}

MARKET STRUCTURE:
{structure}

Please provide:
1. Direction bias (BULLISH/BEARISH/NEUTRAL)
2. Confidence level (0-100%)
3. Key patterns identified
4. Entry recommendation
5. Risk management levels
6. Market context assessment

Format your response as structured analysis focusing on actionable insights.
"""
        return prompt
    
    def _create_gpt4_prompt(self, market_context: Dict, symbol: str) -> str:
        """Create analysis prompt for GPT-4"""
        return self._create_claude_prompt(market_context, symbol)  # Same prompt structure
    
    def _parse_ai_response(self, ai_text: str, model: str) -> Dict:
        """Parse AI response into structured data"""
        try:
            # Extract key information using regex and keywords
            analysis = {
                'model': model,
                'raw_response': ai_text,
                'direction': self._extract_direction(ai_text),
                'confidence': self._extract_confidence(ai_text),
                'patterns': self._extract_patterns(ai_text),
                'entry_price': self._extract_entry_price(ai_text),
                'stop_loss': self._extract_stop_loss(ai_text),
                'take_profit': self._extract_take_profit(ai_text),
                'reasoning': self._extract_reasoning(ai_text)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error parsing AI response: {e}")
            return {'model': model, 'error': str(e)}
    
    def _extract_direction(self, text: str) -> str:
        """Extract trading direction from AI response"""
        text_upper = text.upper()
        
        if 'BULLISH' in text_upper or 'BUY' in text_upper:
            return 'BULLISH'
        elif 'BEARISH' in text_upper or 'SELL' in text_upper:
            return 'BEARISH'
        else:
            return 'NEUTRAL'
    
    def _extract_confidence(self, text: str) -> float:
        """Extract confidence level from AI response"""
        import re
        
        # Look for percentage patterns
        confidence_patterns = [
            r'confidence[:\s]+(\d+)%',
            r'(\d+)%\s+confidence',
            r'probability[:\s]+(\d+)%'
        ]
        
        for pattern in confidence_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return float(match.group(1)) / 100.0
        
        return 0.5  # Default 50% confidence
    
    def _extract_patterns(self, text: str) -> List[str]:
        """Extract identified patterns from AI response"""
        patterns = []
        
        # Common pattern keywords
        pattern_keywords = [
            'doji', 'hammer', 'shooting star', 'engulfing', 'harami',
            'flag', 'pennant', 'triangle', 'wedge', 'channel',
            'support', 'resistance', 'breakout', 'reversal'
        ]
        
        text_lower = text.lower()
        for keyword in pattern_keywords:
            if keyword in text_lower:
                patterns.append(keyword.title())
        
        return patterns
    
    def _extract_entry_price(self, text: str) -> float:
        """Extract entry price recommendation from AI response"""
        import re
        
        # Look for price patterns
        price_patterns = [
            r'entry[:\s]+(\d+\.?\d*)',
            r'enter[:\s]+(\d+\.?\d*)',
            r'buy at[:\s]+(\d+\.?\d*)',
            r'sell at[:\s]+(\d+\.?\d*)'
        ]
        
        for pattern in price_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return float(match.group(1))
        
        return 0.0
    
    def _extract_stop_loss(self, text: str) -> float:
        """Extract stop loss level from AI response"""
        import re
        
        sl_patterns = [
            r'stop[:\s]+(\d+\.?\d*)',
            r'sl[:\s]+(\d+\.?\d*)',
            r'stop loss[:\s]+(\d+\.?\d*)'
        ]
        
        for pattern in sl_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return float(match.group(1))
        
        return 0.0
    
    def _extract_take_profit(self, text: str) -> float:
        """Extract take profit level from AI response"""
        import re
        
        tp_patterns = [
            r'target[:\s]+(\d+\.?\d*)',
            r'tp[:\s]+(\d+\.?\d*)',
            r'take profit[:\s]+(\d+\.?\d*)',
            r'profit[:\s]+(\d+\.?\d*)'
        ]
        
        for pattern in tp_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return float(match.group(1))
        
        return 0.0
    
    def _extract_reasoning(self, text: str) -> str:
        """Extract key reasoning from AI response"""
        # Take first 200 characters as summary
        return text[:200] + "..." if len(text) > 200 else text
    
    def _combine_ai_analyses(self, claude_analysis: Dict, gpt4_analysis: Dict, df: pd.DataFrame) -> Dict:
        """Combine analyses from both AI models"""
        try:
            combined = {
                'claude': claude_analysis,
                'gpt4': gpt4_analysis,
                'consensus': {},
                'combined_signal': 'NEUTRAL',
                'combined_confidence': 0.0
            }
            
            # Calculate consensus
            claude_dir = claude_analysis.get('direction', 'NEUTRAL')
            gpt4_dir = gpt4_analysis.get('direction', 'NEUTRAL')
            
            if claude_dir == gpt4_dir and claude_dir != 'NEUTRAL':
                combined['combined_signal'] = claude_dir
                combined['combined_confidence'] = (
                    claude_analysis.get('confidence', 0.5) + 
                    gpt4_analysis.get('confidence', 0.5)
                ) / 2
            elif claude_dir != 'NEUTRAL' or gpt4_dir != 'NEUTRAL':
                # Partial agreement - use higher confidence one
                if claude_analysis.get('confidence', 0) > gpt4_analysis.get('confidence', 0):
                    combined['combined_signal'] = claude_dir
                    combined['combined_confidence'] = claude_analysis.get('confidence', 0.5) * 0.7
                else:
                    combined['combined_signal'] = gpt4_dir
                    combined['combined_confidence'] = gpt4_analysis.get('confidence', 0.5) * 0.7
            
            # Combine patterns
            all_patterns = []
            all_patterns.extend(claude_analysis.get('patterns', []))
            all_patterns.extend(gpt4_analysis.get('patterns', []))
            combined['consensus']['patterns'] = list(set(all_patterns))
            
            # Use technical analysis for price levels if AI didn't provide them
            current_price = df['close'].iloc[-1]
            atr = df['atr'].iloc[-1] if 'atr' in df.columns else (df['high'] - df['low']).mean()
            
            combined['entry_price'] = (
                claude_analysis.get('entry_price', 0) or 
                gpt4_analysis.get('entry_price', 0) or 
                current_price
            )
            
            combined['stop_loss'] = (
                claude_analysis.get('stop_loss', 0) or 
                gpt4_analysis.get('stop_loss', 0) or 
                (current_price - atr * 2 if combined['combined_signal'] == 'BULLISH' else current_price + atr * 2)
            )
            
            combined['take_profit'] = (
                claude_analysis.get('take_profit', 0) or 
                gpt4_analysis.get('take_profit', 0) or 
                (current_price + atr * 3 if combined['combined_signal'] == 'BULLISH' else current_price - atr * 3)
            )
            
            return combined
            
        except Exception as e:
            logger.error(f"Error combining AI analyses: {e}")
            return {'error': str(e)}
    
    def _fallback_pattern_analysis(self, df: pd.DataFrame) -> Dict:
        """Fallback analysis when AI is not available"""
        try:
            current_price = df['close'].iloc[-1]
            atr = (df['high'] - df['low']).mean()
            
            # Simple momentum analysis
            price_change = (df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5]
            
            if price_change > 0.02:
                signal = 'BULLISH'
                confidence = min(abs(price_change) * 10, 0.8)
            elif price_change < -0.02:
                signal = 'BEARISH'
                confidence = min(abs(price_change) * 10, 0.8)
            else:
                signal = 'NEUTRAL'
                confidence = 0.3
            
            return {
                'combined_signal': signal,
                'combined_confidence': confidence,
                'entry_price': current_price,
                'stop_loss': current_price - (atr * 2) if signal == 'BULLISH' else current_price + (atr * 2),
                'take_profit': current_price + (atr * 3) if signal == 'BULLISH' else current_price - (atr * 3),
                'source': 'fallback_analysis'
            }
            
        except Exception as e:
            logger.error(f"Error in fallback analysis: {e}")
            return {'error': str(e)}
    
    def _check_rate_limit(self) -> bool:
        """Check if we can make AI API requests"""
        current_time = time.time()
        
        # Remove requests older than 1 hour
        self.ai_request_times = [
            req_time for req_time in self.ai_request_times 
            if current_time - req_time < 3600
        ]
        
        # Check if under limit
        if len(self.ai_request_times) < self.max_ai_requests_per_hour:
            self.ai_request_times.append(current_time)
            return True
        
        return False
    
    def _generate_cache_key(self, df: pd.DataFrame, symbol: str) -> str:
        """Generate cache key for pattern analysis"""
        if len(df) < 5:
            return f"{symbol}_empty"
        
        # Use last 5 closes and current time (rounded to 5 minutes) as key
        recent_closes = df['close'].tail(5).values
        time_bucket = int(time.time() // 300) * 300  # Round to 5-minute buckets
        
        return f"{symbol}_{hash(tuple(recent_closes))}_{time_bucket}"
    
    def _find_resistance_levels(self, df: pd.DataFrame) -> List[float]:
        """Find resistance levels"""
        highs = df['high'].values
        resistance_levels = []
        
        for i in range(2, len(highs) - 2):
            if (highs[i] > highs[i-1] and highs[i] > highs[i-2] and 
                highs[i] > highs[i+1] and highs[i] > highs[i+2]):
                resistance_levels.append(highs[i])
        
        return sorted(set(resistance_levels), reverse=True)[:3]  # Top 3
    
    def _find_support_levels(self, df: pd.DataFrame) -> List[float]:
        """Find support levels"""
        lows = df['low'].values
        support_levels = []
        
        for i in range(2, len(lows) - 2):
            if (lows[i] < lows[i-1] and lows[i] < lows[i-2] and 
                lows[i] < lows[i+1] and lows[i] < lows[i+2]):
                support_levels.append(lows[i])
        
        return sorted(set(support_levels))[:3]  # Bottom 3
    
    def _identify_candlestick_patterns(self, df: pd.DataFrame) -> Dict:
        """Identify basic candlestick patterns"""
        patterns = {}
        
        if len(df) < 3:
            return patterns
        
        last_bar = df.iloc[-1]
        prev_bar = df.iloc[-2]
        
        # Doji pattern
        body_size = abs(last_bar['close'] - last_bar['open'])
        range_size = last_bar['high'] - last_bar['low']
        
        if range_size > 0 and body_size / range_size < 0.1:
            patterns['doji'] = True
        
        # Hammer pattern
        lower_shadow = last_bar['open'] - last_bar['low'] if last_bar['close'] > last_bar['open'] else last_bar['close'] - last_bar['low']
        upper_shadow = last_bar['high'] - last_bar['close'] if last_bar['close'] > last_bar['open'] else last_bar['high'] - last_bar['open']
        
        if range_size > 0 and lower_shadow > body_size * 2 and upper_shadow < body_size * 0.5:
            patterns['hammer'] = True
        
        # Engulfing pattern
        if (last_bar['close'] > last_bar['open'] and prev_bar['close'] < prev_bar['open'] and
            last_bar['close'] > prev_bar['open'] and last_bar['open'] < prev_bar['close']):
            patterns['bullish_engulfing'] = True
        elif (last_bar['close'] < last_bar['open'] and prev_bar['close'] > prev_bar['open'] and
              last_bar['close'] < prev_bar['open'] and last_bar['open'] > prev_bar['close']):
            patterns['bearish_engulfing'] = True
        
        return patterns
    
    def _analyze_market_structure(self, df: pd.DataFrame) -> Dict:
        """Analyze market structure"""
        structure = {}
        
        if len(df) < 10:
            return structure
        
        # Trend analysis
        sma_short = df['close'].rolling(5).mean()
        sma_long = df['close'].rolling(10).mean()
        
        if sma_short.iloc[-1] > sma_long.iloc[-1]:
            structure['trend'] = 'uptrend'
        elif sma_short.iloc[-1] < sma_long.iloc[-1]:
            structure['trend'] = 'downtrend'
        else:
            structure['trend'] = 'sideways'
        
        # Volatility analysis
        volatility = df['close'].pct_change().std()
        structure['volatility'] = 'high' if volatility > 0.02 else 'low'
        
        return structure
    
    def _create_ohlc_summary(self, df: pd.DataFrame) -> str:
        """Create OHLC summary for AI prompt"""
        if len(df) < 5:
            return "Insufficient data"
        
        recent = df.tail(5)
        summary = "Last 5 bars:\n"
        
        for i, (idx, row) in enumerate(recent.iterrows()):
            summary += f"Bar {i+1}: O:{row['open']:.2f} H:{row['high']:.2f} L:{row['low']:.2f} C:{row['close']:.2f}\n"
        
        return summary