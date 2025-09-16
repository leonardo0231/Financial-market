import os
from flask import Flask
from .scheduler import start_scheduler
from .api import orchestrator_bp

def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__)
    
    # Configure logging
    app.logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))
    
    # Register the orchestrator blueprint
    app.register_blueprint(orchestrator_bp)
    
    # Start the scheduler with the app context
    with app.app_context():
        start_scheduler(app)
    
    return app

def main():
    app = create_app()
    
    host = os.getenv("ORCH_HOST", "0.0.0.0")
    port = int(os.getenv("ORCH_PORT", "5678"))
    
    app.logger.info(f"Starting Orchestrator on {host}:{port}")
    app.run(host=host, port=port, debug=False)

if __name__ == "__main__":
    main()
