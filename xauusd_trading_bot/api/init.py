"""
API Blueprints Registration
Modularized Flask API with blueprints
"""

from flask import Flask
from .health import health_bp
from .market import market_bp  
from .analysis import analysis_bp
from .trading import trading_bp

def register_blueprints(app: Flask):
    """Register all API blueprints with the Flask app"""
    
    # Register blueprints with url_prefix
    app.register_blueprint(health_bp, url_prefix='/api')
    app.register_blueprint(market_bp, url_prefix='/api') 
    app.register_blueprint(analysis_bp, url_prefix='/api')
    app.register_blueprint(trading_bp, url_prefix='/api')
    
    # Log registered routes for debugging
    import logging
    logger = logging.getLogger(__name__)
    logger.info("Registered API blueprints:")
    for rule in app.url_map.iter_rules():
        if rule.endpoint.startswith('health') or rule.endpoint.startswith('market') or \
           rule.endpoint.startswith('analysis') or rule.endpoint.startswith('trading'):
            logger.info(f"  {rule.methods} {rule.rule} -> {rule.endpoint}")