"""
Trading Execution API Blueprint
"""

from flask import Blueprint, request, jsonify, abort
import logging
import os

logger = logging.getLogger(__name__)
trading_bp = Blueprint('trading', __name__)

def verify_api_key():
    """Verify API key for sensitive operations"""
    api_key = request.headers.get('X-API-Key')
    expected_key = os.getenv('TRADING_API_KEY')
    
    if not expected_key:
        # If no API key configured, allow (for development)
        return True
    
    if not api_key or api_key != expected_key:
        abort(401, description="Invalid or missing API key")
    
    return True

@trading_bp.route('/execute_trade', methods=['POST'])
def execute_trade():
    """Execute a trade based on signal"""
    # Verify authentication for sensitive operations
    verify_api_key()
    
    try:
        from xauusd_trading_bot import main as main_module
        mt5_connector = getattr(main_module, 'mt5_connector', None)
        risk_calculator = getattr(main_module, 'risk_calculator', None)
        
        if not request.json:
            return jsonify({'error': 'JSON data required'}), 400
        
        trade_data = request.json
        required_fields = ['signal', 'symbol']
        for field in required_fields:
            if field not in trade_data:
                return jsonify({'error': f'Missing field: {field}'}), 400
        
        # Normalize signal to uppercase for case-insensitive comparison
        signal = trade_data['signal'].upper() if isinstance(trade_data['signal'], str) else trade_data['signal']
        trade_data['signal'] = signal
        
        if signal not in ['BUY', 'SELL']:
            return jsonify({'error': 'Signal must be BUY or SELL'}), 400
        
        if not mt5_connector or not mt5_connector.is_connected():
            return jsonify({'error': 'MT5 not connected'}), 503
        
        # Calculate position size if not provided (support both 'volume' and 'size' for backward compatibility)
        volume = trade_data.get('volume') or trade_data.get('size', 0)
        if volume <= 0:
            account_info = mt5_connector.get_account_info()
            if account_info:
                symbol_info = mt5_connector.get_symbol_info(trade_data['symbol'])
                risk_params = risk_calculator.calculate_position_size(
                    signal=trade_data,
                    account_balance=account_info.get('balance', 1000),
                    symbol_info=symbol_info
                )
                volume = risk_params.get('position_size', 0.01)
            else:
                volume = 0.01
        
        # Set volume in trade_data for MT5Connector
        trade_data['volume'] = volume
        
        result = mt5_connector.execute_trade(trade_data)
        
        if result.get('success'):
            return jsonify({
                'success': True,
                'message': 'Trade executed successfully',
                'ticket': result.get('ticket'),
                'volume': result.get('volume'),
                'price': result.get('price')
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Trade execution failed',
                'error': result.get('error', 'Unknown error')
            }), 400
            
    except Exception as e:
        logger.error(f"Trade execution error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@trading_bp.route('/positions', methods=['GET'])
def get_positions():
    """Get current open positions"""
    # Verify authentication for sensitive operations
    verify_api_key()
    
    try:
        from xauusd_trading_bot import main as main_module
        mt5_connector = getattr(main_module, 'mt5_connector', None)
        
        if not mt5_connector or not mt5_connector.is_connected():
            return jsonify({'error': 'MT5 not connected'}), 503
        
        symbol = request.args.get('symbol')
        positions = mt5_connector.get_open_positions(symbol)
        
        return jsonify({
            'success': True,
            'positions': positions,
            'count': len(positions)
        })
        
    except Exception as e:
        logger.error(f"Get positions error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@trading_bp.route('/close_position', methods=['POST'])
def close_position():
    """Close a specific position"""
    # Verify authentication for sensitive operations
    verify_api_key()
    
    try:
        from xauusd_trading_bot import main as main_module
        mt5_connector = getattr(main_module, 'mt5_connector', None)
        
        if not request.json:
            return jsonify({'error': 'JSON data required'}), 400
        
        ticket = request.json.get('ticket')
        if not ticket:
            return jsonify({'error': 'Ticket number required'}), 400
        
        if not mt5_connector or not mt5_connector.is_connected():
            return jsonify({'error': 'MT5 not connected'}), 503
        
        result = mt5_connector.close_position(ticket)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Close position error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@trading_bp.route('/account_info', methods=['GET'])
def get_account_info():
    """Get account information"""
    try:
        from xauusd_trading_bot import main as main_module
        mt5_connector = getattr(main_module, 'mt5_connector', None)
        
        if not mt5_connector or not mt5_connector.is_connected():
            return jsonify({'error': 'MT5 not connected'}), 503
        
        account_info = mt5_connector.get_account_info()
        
        if not account_info:
            return jsonify({'error': 'Failed to get account info'}), 500
        
        return jsonify({'success': True, 'account': account_info})
        
    except Exception as e:
        logger.error(f"Account info error: {str(e)}")
        return jsonify({'error': str(e)}), 500