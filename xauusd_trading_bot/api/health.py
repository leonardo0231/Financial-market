"""
Health Check API Blueprint
Handles system health and status endpoints
"""

import time
from datetime import datetime
from flask import Blueprint, jsonify
import logging

logger = logging.getLogger(__name__)

# Create blueprint
health_bp = Blueprint('health', __name__)

@health_bp.route('/health', methods=['GET'])
def health_check():
    """Main health check endpoint"""
    try:
        # Import here to avoid circular imports
        # Get global instances that are initialized by main.py
        from xauusd_trading_bot import main as main_module
        bot = getattr(main_module, 'bot', None)
        mt5_connector = getattr(main_module, 'mt5_connector', None)  
        db_manager = getattr(main_module, 'get_db_manager', lambda: None)()
        from xauusd_trading_bot._version import VERSION
        
        # Check MT5 connection
        mt5_connected = False
        if mt5_connector:
            mt5_connected = mt5_connector.is_connected()
        
        # Check database connection  
        database_connected = False
        if db_manager:
            database_connected = db_manager.health_check()
        
        # Check bot initialization
        bot_initialized = False
        active_positions = 0
        if bot:
            bot_initialized = getattr(bot, '_initialized', False)
            if mt5_connector and mt5_connected:
                try:
                    positions = mt5_connector.get_open_positions()
                    active_positions = len(positions) if positions else 0
                except:
                    pass
        
        # Determine overall status
        overall_status = 'healthy' if (mt5_connected and database_connected and bot_initialized) else 'unhealthy'
        
        # Prepare response data
        health_data = {
            'status': overall_status,
            'timestamp': datetime.now().isoformat(),
            # Flat structure for n8n compatibility
            'mt5_connected': mt5_connected,
            'database_connected': database_connected,
            'bot_initialized': bot_initialized,
            'active_positions': active_positions,
            'strategies_loaded': True,
            'uptime': time.time() - getattr(bot, 'start_time', time.time()),
            'uptime_seconds': time.time() - getattr(bot, 'start_time', time.time()),
            'version': VERSION,
            # Nested structure for advanced monitoring
            'components': {
                'mt5_connected': mt5_connected,
                'database_connected': database_connected,
                'bot_initialized': bot_initialized
            },
            'metrics': {
                'active_positions': active_positions,
                'uptime_seconds': time.time() - getattr(bot, 'start_time', time.time())
            }
        }
        
        status_code = 200 if overall_status == 'healthy' else 503
        return jsonify(health_data), status_code
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@health_bp.route('/status', methods=['GET'])
def status():
    """Alias for health check for backward compatibility"""
    return health_check()

@health_bp.route('/ping', methods=['GET'])
def ping():
    """Simple ping endpoint"""
    return jsonify({'status': 'pong', 'timestamp': datetime.now().isoformat()})

@health_bp.route('/mt5/status', methods=['GET'])
def mt5_status():
    """Detailed MT5 connection status"""
    try:
        from xauusd_trading_bot import main as main_module
        mt5_connector = getattr(main_module, 'mt5_connector', None)
        
        if not mt5_connector:
            return jsonify({
                'connected': False,
                'error': 'MT5 connector not initialized',
                'timestamp': datetime.now().isoformat()
            }), 503
        
        connected = mt5_connector.is_connected()
        deadlock_status = mt5_connector.check_deadlock_status()
        
        account_info = {}
        if connected:
            try:
                account_info = mt5_connector.get_account_info()
            except Exception as e:
                logger.warning(f"Failed to get account info: {e}")
        
        return jsonify({
            'connected': connected,
            'account_info': account_info,
            'deadlock_status': deadlock_status,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"MT5 status check failed: {str(e)}")
        return jsonify({
            'connected': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@health_bp.route('/database/status', methods=['GET'])
def database_status():
    """Detailed database connection status"""
    try:
        from xauusd_trading_bot import main as main_module
        db_manager = getattr(main_module, 'get_db_manager', lambda: None)()
        
        if not db_manager:
            return jsonify({
                'connected': False,
                'error': 'Database manager not initialized',
                'timestamp': datetime.now().isoformat()
            }), 503
        
        # Test database connection
        try:
            connected = db_manager.health_check()
            
            # Get additional database stats if connected
            stats = {}
            if connected:
                session = db_manager.get_session()
                try:
                    # Count records in main tables
                    from xauusd_trading_bot.database.models import Trade, Signal, Performance
                    stats = {
                        'trades_count': session.query(Trade).count(),
                        'signals_count': session.query(Signal).count(),
                        'performance_records': session.query(Performance).count()
                    }
                except Exception as e:
                    logger.warning(f"Failed to get database stats: {e}")
                    stats = {'error': 'Unable to fetch statistics'}
                finally:
                    session.close()
            
            return jsonify({
                'connected': connected,
                'database_type': 'postgresql' if 'postgresql' in str(db_manager.engine.url) else 'sqlite',
                'stats': stats,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            return jsonify({
                'connected': False,
                'error': f'Database connection test failed: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }), 503
        
    except Exception as e:
        logger.error(f"Database status check failed: {str(e)}")
        return jsonify({
            'connected': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500