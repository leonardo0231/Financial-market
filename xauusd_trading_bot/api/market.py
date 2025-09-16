"""
Market Data API Blueprint
"""

from flask import Blueprint, request, jsonify
import logging

logger = logging.getLogger(__name__)
market_bp = Blueprint('market', __name__)

@market_bp.route('/ohlc', methods=['GET'])
def get_ohlc_data():
    """Get OHLC data for a symbol"""
    try:
        from xauusd_trading_bot import main as main_module
        mt5_connector = getattr(main_module, 'mt5_connector', None)
        bot = getattr(main_module, 'bot', None)
        
        symbol = request.args.get('symbol', 'XAUUSD').upper()
        timeframe = request.args.get('timeframe', 'M5').upper()
        bars = request.args.get('bars', '100', type=int)
        
        valid_timeframes = ['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1']
        if timeframe not in valid_timeframes:
            return jsonify({'error': f'Invalid timeframe. Valid: {valid_timeframes}'}), 400
        
        if not (1 <= bars <= 1000):
            return jsonify({'error': 'Bars must be between 1 and 1000'}), 400
        
        if not mt5_connector or not mt5_connector.is_connected():
            return jsonify({'error': 'MT5 not connected'}), 503
        
        ohlc_data = bot.get_market_data(symbol, timeframe, bars) if bot else mt5_connector.get_ohlc_data(symbol, timeframe, bars)
        
        if ohlc_data.empty:
            return jsonify({'success': False, 'error': f'No data for {symbol}'}), 404
        
        data_list = []
        for index, row in ohlc_data.iterrows():
            data_list.append({
                'time': index.isoformat() if hasattr(index, 'isoformat') else str(index),
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'tick_volume': int(row['tick_volume']) if 'tick_volume' in row else 0
            })
        
        return jsonify({
            'success': True,
            'symbol': symbol,
            'timeframe': timeframe,
            'bars_requested': bars,
            'bars_returned': len(data_list),
            'data': data_list
        })
        
    except Exception as e:
        logger.error(f"OHLC error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@market_bp.route('/symbols', methods=['GET'])
def get_symbols():
    """Get available symbols"""
    symbols = [
        {'symbol': 'XAUUSD', 'description': 'Gold vs US Dollar'},
        {'symbol': 'EURUSD', 'description': 'Euro vs US Dollar'},
        {'symbol': 'GBPUSD', 'description': 'British Pound vs US Dollar'},
        {'symbol': 'USDJPY', 'description': 'US Dollar vs Japanese Yen'}
    ]
    return jsonify({'success': True, 'symbols': symbols})