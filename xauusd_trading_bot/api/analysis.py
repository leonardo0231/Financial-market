"""
Market Analysis API Blueprint
"""

from flask import Blueprint, request, jsonify
import logging

logger = logging.getLogger(__name__)
analysis_bp = Blueprint('analysis', __name__)

@analysis_bp.route('/analyze', methods=['POST'])
def analyze_market():
    """Analyze market with selected strategies"""
    try:
        from xauusd_trading_bot import main as main_module
        mt5_connector = getattr(main_module, 'mt5_connector', None)
        data_processor = getattr(main_module, 'data_processor', None)
        strategy_manager = getattr(main_module, 'strategy_manager', None)
        risk_calculator = getattr(main_module, 'risk_calculator', None)
        bot = getattr(main_module, 'bot', None)
        
        if not request.json:
            return jsonify({'error': 'JSON data required'}), 400
            
        data = request.json
        symbol = data.get('symbol', 'XAUUSD').upper()
        strategies = data.get('strategies', ['all'])
        timeframe = data.get('timeframe', 'M5').upper()
        bars = data.get('bars', 100)
        
        if not isinstance(strategies, list):
            return jsonify({'error': 'Strategies must be a list'}), 400
            
        if bars <= 0 or bars > 1000:
            return jsonify({'error': 'Bars must be between 1 and 1000'}), 400
        
        if not mt5_connector or not mt5_connector.is_connected():
            return jsonify({'error': 'MT5 not connected'}), 503
        
        ohlc_data = bot.get_market_data(symbol, timeframe, bars)
        if ohlc_data.empty:
            return jsonify({'error': f'No data available for {symbol}'}), 404
            
        processed_data = data_processor.prepare_data(ohlc_data)
        analysis_result = strategy_manager.analyze_market(processed_data, strategies)
        
        # Calculate risk parameters
        account_info = mt5_connector.get_account_info()
        if account_info and 'balance' in account_info and analysis_result['signal'] in ['BUY', 'SELL']:
            open_positions = len(mt5_connector.get_open_positions(symbol))
            symbol_info = mt5_connector.get_symbol_info(symbol)
            
            risk_params = risk_calculator.calculate_position_size(
                signal=analysis_result,
                account_balance=account_info['balance'],
                symbol_info=symbol_info,
                open_positions=open_positions
            )
            analysis_result['risk_parameters'] = risk_params
        else:
            analysis_result['risk_parameters'] = {
                'position_size': 0.01,
                'risk_amount': 0,
                'risk_reward_ratio': 0,
                'max_loss_amount': 0
            }
        
        return jsonify({
            'success': True,
            'symbol': symbol,
            'timeframe': timeframe,
            'strategies_used': strategies,
            **analysis_result
        })
        
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@analysis_bp.route('/strategies', methods=['GET'])
def get_strategies():
    """Get available strategies"""
    try:
        from xauusd_trading_bot.main import strategy_manager
        if strategy_manager:
            strategies = strategy_manager.get_available_strategies()
            return jsonify({'success': True, 'strategies': strategies})
        return jsonify({'error': 'Strategy manager not available'}), 503
    except Exception as e:
        return jsonify({'error': str(e)}), 500