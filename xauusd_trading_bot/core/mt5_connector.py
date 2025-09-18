"""
MetaTrader 5 Connector Module
Thread-safe MT5 interactions with circuit breaker protection
"""

import logging
import threading
import time
import os
import dotenv
import yaml
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from functools import wraps
from contextlib import contextmanager

import MetaTrader5 as mt5
import pandas as pd
import numpy as np

from ..utils.circuit_breaker import mt5_circuit_breaker
from ..utils.cache_manager import CacheManager

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not installed, continue without it
    pass

logger = logging.getLogger(__name__)

dotenv.load_dotenv()

class PriceUtils:
    """Utilities for MT5 price precision and volume rounding"""
    
    @staticmethod
    def normalize_price(price: float, symbol: str, mt5_connector) -> float:
        """Normalize price according to symbol digits"""
        try:
            symbol_info = mt5_connector.get_symbol_info(symbol)
            if symbol_info:
                digits = symbol_info.get('digits', 5)
                return round(price, digits)
            return round(price, 5)  # Default to 5 digits for most forex pairs
        except Exception as e:
            logger.warning(f"Price normalization failed for {symbol}: {e}")
            return round(price, 5)
    
    @staticmethod
    def normalize_volume(volume: float, symbol: str, mt5_connector) -> float:
        """Normalize volume according to symbol volume step"""
        try:
            symbol_info = mt5_connector.get_symbol_info(symbol)
            if symbol_info:
                volume_step = symbol_info.get('volume_step', 0.01)
                return round(volume / volume_step) * volume_step
            return round(volume, 2)  # Default to 0.01 lot step
        except Exception as e:
            logger.warning(f"Volume normalization failed for {symbol}: {e}")
            return round(volume, 2)
    
    @staticmethod
    def calculate_deviation_points(symbol: str, price: float, mt5_connector, max_deviation_pips: int = 20) -> int:
        """Calculate deviation in points for slippage control"""
        try:
            symbol_info = mt5_connector.get_symbol_info(symbol)
            if symbol_info:
                point = symbol_info.get('point', 0.00001)
                return int(max_deviation_pips * 10 * point)
            return max_deviation_pips * 10  # Default conversion
        except Exception as e:
            logger.warning(f"Deviation calculation failed for {symbol}: {e}")
            return max_deviation_pips * 10


class MT5Connector:
    """Advanced thread-safe MT5 connection handler with circuit breaker and caching"""
    
    # Timeframe mapping
    TIMEFRAMES = {
        'M1': mt5.TIMEFRAME_M1,
        'M5': mt5.TIMEFRAME_M5,
        'M15': mt5.TIMEFRAME_M15,
        'M30': mt5.TIMEFRAME_M30,
        'H1': mt5.TIMEFRAME_H1,
        'H4': mt5.TIMEFRAME_H4,
        'D1': mt5.TIMEFRAME_D1
    }

    def __init__(self, login: int, password: str, server: str, cache_manager: Optional[CacheManager] = None):
        """Initialize MT5 connector with advanced concurrency control and caching"""
        # Load configuration from YAML file or environment variables
        config = self._load_mt5_config()
        
        raw_login = os.getenv("MT5_LOGIN", "")

        try:
            self.login = int(str(raw_login).strip()) if raw_login else None
        except ValueError:
            self.login = None
        
        self.password = os.getenv("MT5_PASSWORD")
        self.server = os.getenv("MT5_SERVER")
        self.terminal_path = config.get('terminal_path') or os.getenv('MT5_TERMINAL_PATH')
        self.timeout_seconds = config.get('timeout_seconds', 60000)
        self.auto_connect = config.get('auto_connect', True)
        self.symbol_settings = config.get('symbol_settings', {})
        
        self.connected = False
        self._account_info = None
        self.cache_manager = cache_manager
        
        # Improved concurrency control with deadlock prevention
        self._connection_lock = threading.RLock()  # For connection/disconnection only
        
        # Use separate read/write locks to allow concurrent reads while preventing conflicts
        # Multiple reads can happen simultaneously, but writes are exclusive
        self._read_lock = threading.RLock()  # For read operations (positions, symbol_info, etc.)
        self._write_lock = threading.RLock()  # For write operations (trades, orders)
        
        # Separate cache lock to avoid holding operation lock during cache operations
        self._cache_lock = threading.RLock()
        
        # Thread-local storage for MT5 operations
        self._thread_local = threading.local()  # For cache operations only
        
        # Operation tracking for deadlock detection
        self._active_operations = {}
        self._operation_timeout = 45  # Timeout for MT5 operations
        
        # Caching for frequently accessed data
        self._symbol_cache = {}
        self._account_cache = {}
        self._cache_ttl = 30  # Cache TTL in seconds
        self._last_cache_update = {}
        
        # Connection health monitoring
        self._last_health_check = 0
        self._health_check_interval = 60  # Check every 60 seconds
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 3
    
    def _load_mt5_config(self) -> dict:
        """Load MT5 configuration from YAML file"""
        try:
            config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config', 'mt5.yaml')
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                    return config
        except Exception as e:
            logger.debug(f"Could not load MT5 YAML config: {e}")
        return {}
    
    def _check_mt5_availability(self) -> bool:
        """Check if MT5 terminal is available and running"""
        try:
            # Try to get terminal info without initializing
            terminal_info = mt5.terminal_info()
            if terminal_info is None:
                # If terminal_info is None, try to initialize briefly
                if not mt5.initialize():
                    return False
                # Check again after initialization
                terminal_info = mt5.terminal_info()
                if terminal_info is None:
                    return False
            return True
        except Exception as e:
            logger.debug(f"MT5 availability check failed: {e}")
            return False
        
    def connect(self) -> bool:
        """Connect to MT5 terminal with improved error handling"""
        with self._connection_lock:
            try:
               
                # Check if MT5 terminal is available
                if not self._check_mt5_availability():
                    logger.error("MT5 terminal is not available or not running")
                    err = mt5.last_error()
                    logger.error(f"Initialization failed: {err}")
                    return False
                
                # Initialize MT5
                if not mt5.initialize():
                    error_code = mt5.last_error()
                    logger.error(f"MT5 initialization failed. Error code: {error_code}")
                    logger.error("Make sure MetaTrader 5 terminal is running and accessible")
                    return False
                
                logger.info("MT5 initialized successfully")
                
                # Login if credentials provided
                if self.login and self.password and self.server:
                    logger.info(f"Attempting to login with account {self.login} on server {self.server}")
                    authorized = mt5.login(
                        login=self.login,
                        password=self.password,
                        server=self.server
                    )
                    
                    if not authorized:
                        error_code = mt5.last_error()
                        logger.error(f"MT5 login failed. Error code: {error_code}")
                        logger.error("Please check your login credentials and server name")
                        return False
                        
                    logger.info(f"MT5 login successful for account {self.login}")
                else:
                    logger.info("No MT5 credentials provided, using existing terminal connection")
                
                # Verify connection
                account_info = mt5.account_info()
                if account_info is None:
                    logger.error("Failed to get account info after connection")
                    logger.error("Make sure you are logged into MetaTrader 5 terminal")
                    return False
                
                self._account_info = account_info._asdict()
                self.connected = True
                self._reconnect_attempts = 0  # Reset reconnection counter
                
                # Test connection with a simple request
                terminal_info = mt5.terminal_info()
                if terminal_info is None:
                    logger.error("Terminal info unavailable")
                    return False
                
                # Initialize cache
                with self._cache_lock:
                    self._account_cache = self._account_info.copy()
                    self._last_cache_update['account'] = time.time()
                
                logger.info(f"Connected to MT5 - Account: {self._account_info['login']}, "
                           f"Server: {self._account_info.get('server', 'Unknown')}")
                return True
                
            except Exception as e:
                logger.error(f"MT5 connection error: {str(e)}")
                return False
    
    def _health_check(self) -> bool:
        """Perform periodic health check and auto-reconnect if needed"""
        current_time = time.time()
        
        # Skip if health check performed recently
        if (current_time - self._last_health_check) < self._health_check_interval:
            return self.connected
        
        self._last_health_check = current_time
        
        # Quick health check
        try:
            if not self.connected:
                return False
            
            # Test connection with lightweight request
            terminal_info = mt5.terminal_info()
            if terminal_info is None:
                logger.warning("MT5 terminal info unavailable - connection may be lost")
                return self._attempt_reconnect()
            
            # Additional check - try to get account info
            account_info = mt5.account_info()
            if account_info is None:
                logger.warning("MT5 account info unavailable - attempting reconnect")
                return self._attempt_reconnect()
            
            return True
            
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return self._attempt_reconnect()
    
    def _attempt_reconnect(self) -> bool:
        """Attempt to reconnect to MT5"""
        if self._reconnect_attempts >= self._max_reconnect_attempts:
            logger.error(f"Max reconnection attempts ({self._max_reconnect_attempts}) reached")
            return False
        
        self._reconnect_attempts += 1
        logger.info(f"Attempting reconnection ({self._reconnect_attempts}/{self._max_reconnect_attempts})")
        
        # Disconnect first
        try:
            mt5.shutdown()
        except:
            pass
        
        self.connected = False
        
        # Try to reconnect
        return self.connect()
    
    def _clear_cache(self):
        """Clear all cached data"""
        with self._cache_lock:
            self._symbol_cache.clear()
            self._account_cache.clear()
            self._last_cache_update.clear()
    
    def disconnect(self) -> None:
        """Disconnect from MT5"""
        if self.connected:
            try:
                mt5.shutdown()
                self.connected = False
                logger.info("Disconnected from MT5")
            except Exception as e:
                logger.error(f"Error during MT5 disconnect: {str(e)}")
                self.connected = False
    
    def is_connected(self) -> bool:
        """Check if connected to MT5 with health monitoring"""
        if not self.connected:
            return False
        
        # Perform periodic health check
        return self._health_check()
    
    def get_account_info(self) -> Dict:
        """Get current account information with caching and improved concurrency"""
        # Check cache first
        with self._cache_lock:
            cache_key = 'account'
            last_update = self._last_cache_update.get(cache_key, 0)
            
            if (time.time() - last_update) < self._cache_ttl and self._account_cache:
                return self._account_cache.copy()
        
        # Use operation lock with timeout to prevent deadlock
        operation_id = f"account_info_{threading.current_thread().ident}"
        
        try:
            # Track operation start time for deadlock detection
            self._active_operations[operation_id] = time.time()
            
            # Acquire operation lock with timeout
            lock_acquired = False
            try:
                # Try to acquire read lock with timeout (read operation)
                timeout_seconds = 10  # Reduced from 30 seconds
                lock_acquired = self._read_lock.acquire(blocking=True, timeout=timeout_seconds)
                
                if not lock_acquired:
                    logger.warning(f"Failed to acquire read lock for account_info within {timeout_seconds}s")
                    return {}
            except Exception as e:
                logger.error(f"Error acquiring operation lock: {e}")
                return {}
            try:
                if not self.is_connected():
                    return {}
                
                account_info = mt5.account_info()
                if account_info:
                    result = account_info._asdict()
                    
                    # Update cache before returning
                    with self._cache_lock:
                        self._account_cache = result.copy()
                        self._last_cache_update['account'] = time.time()
                    
                    return result
                return {}
                
            except Exception as e:
                logger.error(f"Error getting account info: {str(e)}")
                return {}
            finally:
                if lock_acquired:
                    self._read_lock.release()
                    
        finally:
            # Clean up operation tracking
            self._active_operations.pop(operation_id, None)
    
    def get_ohlc_data(self, symbol: str, timeframe: str, bars: int) -> pd.DataFrame:
        """Get OHLC data for symbol with improved concurrency and deadlock prevention"""
        operation_id = f"ohlc_{symbol}_{threading.current_thread().ident}"
        
        try:
            # Track operation for deadlock detection
            self._active_operations[operation_id] = time.time()
            
            # Acquire operation lock with timeout
            lock_acquired = False
            try:
                timeout_seconds = 10  # Reduced from 30 seconds
                lock_acquired = self._read_lock.acquire(blocking=True, timeout=timeout_seconds)
                
                if not lock_acquired:
                    logger.warning(f"Failed to acquire read lock for OHLC data ({symbol}) within {timeout_seconds}s")
                    return pd.DataFrame()
            except Exception as e:
                logger.error(f"Error acquiring operation lock for OHLC: {e}")
                return pd.DataFrame()
            
            try:
                if not self.is_connected():
                    raise Exception("Not connected to MT5")
                
                # Get timeframe
                tf = self.TIMEFRAMES.get(timeframe, mt5.TIMEFRAME_M5)
                
                # Get rates
                rates = mt5.copy_rates_from_pos(symbol, tf, 0, bars)
                
                if rates is None or len(rates) == 0:
                    raise Exception(f"Failed to get rates for {symbol}")
                
                # Convert to DataFrame
                df = pd.DataFrame(rates)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                df.set_index('time', inplace=True)
                
                # Add additional columns
                df['symbol'] = symbol
                df['timeframe'] = timeframe
                
                return df
                
            except Exception as e:
                logger.error(f"Error getting OHLC data: {str(e)}")
                return pd.DataFrame()
            finally:
                if lock_acquired:
                    self._read_lock.release()
                    
        finally:
            # Clean up operation tracking
            self._active_operations.pop(operation_id, None)
    
    def get_symbol_info(self, symbol: str) -> Dict:
        """Get symbol information with caching and improved concurrency"""
        # Check cache first
        with self._cache_lock:
            cache_key = f'symbol_{symbol}'
            last_update = self._last_cache_update.get(cache_key, 0)
            
            if (time.time() - last_update) < self._cache_ttl and cache_key in self._symbol_cache:
                return self._symbol_cache[cache_key].copy()
        
        # Use operation lock for thread-safe access with timeout
        operation_id = f"symbol_info_{symbol}_{threading.current_thread().ident}"
        
        try:
            # Track operation for deadlock detection
            self._active_operations[operation_id] = time.time()
            
            # Acquire read lock with timeout (allow concurrent reads)
            lock_acquired = False
            try:
                timeout_seconds = 10  # Reduced from 30 seconds
                lock_acquired = self._read_lock.acquire(blocking=True, timeout=timeout_seconds)
                
                if not lock_acquired:
                    logger.warning(f"Failed to acquire read lock for symbol_info ({symbol}) within {timeout_seconds}s")
                    return {}
            except Exception as e:
                logger.error(f"Error acquiring read lock for symbol_info: {e}")
                return {}
            
            try:
                if not self.is_connected():
                    return {}
                
                info = mt5.symbol_info(symbol)
                if info:
                    result = info._asdict()
                    
                    # Update cache
                    with self._cache_lock:
                        cache_key = f'symbol_{symbol}'
                        self._symbol_cache[cache_key] = result.copy()
                        self._last_cache_update[cache_key] = time.time()
                    
                    return result
                return {}
                
            except Exception as e:
                logger.error(f"Error getting symbol info: {str(e)}")
                return {}
            finally:
                if lock_acquired:
                    self._read_lock.release()
                    
        finally:
            # Clean up operation tracking
            self._active_operations.pop(operation_id, None)
    
    def execute_trade(self, trade_signal: Dict) -> Dict:
        """Execute trade based on signal with improved concurrency and deadlock prevention"""
        operation_id = f"trade_{trade_signal.get('symbol', 'unknown')}_{threading.current_thread().ident}"
        
        try:
            # Track operation for deadlock detection
            self._active_operations[operation_id] = time.time()
            
            # Acquire write lock with optimized timeout (trades need exclusive access)
            lock_acquired = False
            timeout_seconds = 15  # Reduced from 45 seconds
            try:
                lock_acquired = self._write_lock.acquire(blocking=True, timeout=timeout_seconds)
                
                if not lock_acquired:
                    logger.error(f"Failed to acquire write lock for trade within {timeout_seconds}s")
                    return {'success': False, 'error': 'Failed to acquire trading lock - system busy'}
            except Exception as e:
                logger.error(f"Error acquiring write lock for trade: {e}")
                return {'success': False, 'error': f'Lock acquisition error: {str(e)}'}
            
            try:
                if not self.is_connected():
                    return {'success': False, 'error': 'Not connected to MT5'}
                
                # Extract trade parameters
                symbol = trade_signal['symbol']
                # Support both 'signal' and 'type' for backward compatibility
                order_type = trade_signal.get('signal', trade_signal.get('type'))  # 'BUY' or 'SELL'
                if not order_type:
                    return {'success': False, 'error': 'Missing signal/type field'}
                
                # Normalize to uppercase for case-insensitive comparison
                order_type = order_type.upper() if isinstance(order_type, str) else order_type
                if order_type not in ['BUY', 'SELL']:
                    return {'success': False, 'error': 'Order type must be BUY or SELL'}
                volume = trade_signal['volume']
                deviation = trade_signal.get('deviation', 20)
                sl = trade_signal.get('stop_loss', 0)
                tp = trade_signal.get('take_profit', 0)
                comment = trade_signal.get('comment', 'AI Trading Bot')
                
                # Get symbol info
                symbol_info = mt5.symbol_info(symbol)
                if symbol_info is None:
                    return {'success': False, 'error': f'Symbol {symbol} not found'}
                
                # Check if symbol is tradeable
                if not symbol_info.visible:
                    mt5.symbol_select(symbol, True)
                
                # Get current price
                tick = mt5.symbol_info_tick(symbol)
                if tick is None:
                    return {'success': False, 'error': 'Failed to get current price'}
                
                price = tick.ask if order_type == 'BUY' else tick.bid
                # Check margin requirements before order execution
                try:
                    required_margin = self.calculate_margin(symbol, volume, order_type)
                    account_info = mt5.account_info()
                    if account_info and account_info.margin_free < required_margin:
                        return {
                            'success': False, 
                            'error': f'Insufficient margin. Required: {required_margin:.2f}, Available: {account_info.margin_free:.2f}'
                        }
                    logger.debug(f"Margin check passed. Required: {required_margin:.2f}, Available: {account_info.margin_free:.2f}")
                except Exception as margin_error:
                    logger.warning(f"Margin calculation failed: {margin_error}. Proceeding with order...")
                
                order_type_mt5 = mt5.ORDER_TYPE_BUY if order_type == 'BUY' else mt5.ORDER_TYPE_SELL
                # Prepare order request
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": volume,
                    "type": order_type_mt5,
                    "price": price,
                    "sl": sl,
                    "tp": tp,
                    "deviation": deviation,
                    "magic": 123456,
                    "comment": comment,
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }
                
                # Send order
                result = mt5.order_send(request)
            
                if result.retcode != mt5.TRADE_RETCODE_DONE:
                    return {
                        'success': False,
                        'error': f'Order failed: {result.comment}',
                        'retcode': result.retcode
                    }
                
                logger.info(f"Trade executed: {result.order}")
                
                return {
                    'success': True,
                    'ticket': result.order,
                    'volume': result.volume,
                    'price': result.price,
                    'comment': result.comment
                }
                    
            except Exception as e:
                logger.error(f"Error executing trade: {str(e)}")
                return {'success': False, 'error': str(e)}
            finally:
                if lock_acquired:
                        self._write_lock.release()
                        
        finally:
            # Clean up operation tracking
            self._active_operations.pop(operation_id, None)
    
    def check_deadlock_status(self) -> Dict:
        """Check for potential deadlocks and long-running operations"""
        try:
            current_time = time.time()
            long_running_ops = {}
            
            for op_id, start_time in self._active_operations.items():
                duration = current_time - start_time
                if duration > self._operation_timeout:
                    long_running_ops[op_id] = {
                        'duration': duration,
                        'started_at': start_time,
                        'timeout_exceeded': True
                    }
            
            return {
                'active_operations': len(self._active_operations),
                'long_running_operations': long_running_ops,
                'potential_deadlock': len(long_running_ops) > 0,
                'operation_timeout': self._operation_timeout
            }
            
        except Exception as e:
            logger.error(f"Error checking deadlock status: {e}")
            return {'error': str(e)}
    
    def get_open_positions(self, symbol: str) -> List[Dict]:
        """Get open positions with improved concurrency"""
        operation_id = f"positions_{symbol or 'all'}_{threading.current_thread().ident}"
        
        try:
            # Track operation for deadlock detection
            self._active_operations[operation_id] = time.time()
            
            # Acquire read lock with timeout (allow concurrent reads)
            lock_acquired = False
            try:
                timeout_seconds = 10  # Reduced from 30 seconds
                lock_acquired = self._read_lock.acquire(blocking=True, timeout=timeout_seconds)
                
                if not lock_acquired:
                    logger.warning(f"Failed to acquire read lock for positions ({symbol}) within {timeout_seconds}s")
                    return []
            except Exception as e:
                logger.error(f"Error acquiring read lock for positions: {e}")
                return []
            
            try:
                if not self.is_connected():
                    return []
                
                if symbol:
                    positions = mt5.positions_get(symbol=symbol)
                else:
                    positions = mt5.positions_get()
                
                if positions is None:
                    return []
                
                return [pos._asdict() for pos in positions]
                
            except Exception as e:
                logger.error(f"Error getting positions: {str(e)}")
                return []
            finally:
                if lock_acquired:
                    self._read_lock.release()
                    
        finally:
            # Clean up operation tracking
            self._active_operations.pop(operation_id, None)
    
    def close_position(self, ticket: int) -> Dict:
        """Close a specific position"""
        try:
            if not self.is_connected():
                return {'success': False, 'error': 'Not connected to MT5'}
            
            # Get position info
            position = mt5.positions_get(ticket=ticket)
            if not position:
                return {'success': False, 'error': 'Position not found'}
            
            position = position[0]
            
            # Prepare close request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "position": ticket,
                "symbol": position.symbol,
                "volume": position.volume,
                "type": mt5.ORDER_TYPE_SELL if position.type == 0 else mt5.ORDER_TYPE_BUY,
                "price": mt5.symbol_info_tick(position.symbol).bid if position.type == 0 else mt5.symbol_info_tick(position.symbol).ask,
                "deviation": 20,
                "magic": 234000,
                "comment": "Close by bot",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Send close order
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                return {
                    'success': False,
                    'error': f'Close failed: {result.comment}',
                    'retcode': result.retcode
                }
            
            return {
                'success': True,
                'ticket': result.order,
                'closed_volume': result.volume,
                'close_price': result.price
            }
            
        except Exception as e:
            logger.error(f"Error closing position: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def modify_position(self, ticket: int, sl: float, tp: float) -> Dict:
        """Modify position SL/TP"""
        try:
            if not self.is_connected():
                return {'success': False, 'error': 'Not connected to MT5'}
            
            # Get position info
            position = mt5.positions_get(ticket=ticket)
            if not position:
                return {'success': False, 'error': 'Position not found'}
            
            position = position[0]
            
            # Prepare modify request
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": ticket,
                "symbol": position.symbol,
                "sl": sl if sl is not None else position.sl,
                "tp": tp if tp is not None else position.tp,
                "magic": 234000,
                "comment": "Modified by bot"
            }
            
            # Send modify order
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                return {
                    'success': False,
                    'error': f'Modify failed: {result.comment}',
                    'retcode': result.retcode
                }
            
            return {
                'success': True,
                'ticket': ticket,
                'new_sl': sl,
                'new_tp': tp
            }
            
        except Exception as e:
            logger.error(f"Error modifying position: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def get_history_orders(self, days: int = 7) -> List[Dict]:
        """Get historical orders"""
        try:
            if not self.is_connected():
                return []
            
            # Calculate time range
            from_date = datetime.now() - timedelta(days=days)
            to_date = datetime.now()
            
            # Get history orders
            orders = mt5.history_orders_get(from_date, to_date)
            
            if orders is None:
                return []
            
            return [order._asdict() for order in orders]
            
        except Exception as e:
            logger.error(f"Error getting history: {str(e)}")
            return []
    
    def calculate_margin(self, symbol: str, volume: float, order_type: str) -> float:
        """Calculate required margin for order"""
        try:
            if not self.is_connected():
                return 0.0
            
            order_type_mt5 = mt5.ORDER_TYPE_BUY if order_type == 'BUY' else mt5.ORDER_TYPE_SELL
            
            # Get current price for accurate margin calculation
            tick = mt5.symbol_info_tick(symbol)
            if not tick:
                logger.warning(f"Could not get tick data for {symbol}, using 0 price")
                price = 0
            else:
                price = tick.ask if order_type == 'BUY' else tick.bid
            
            margin = mt5.order_calc_margin(order_type_mt5, symbol, volume, price)
            
            return margin if margin else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating margin: {str(e)}")
            return 0.0