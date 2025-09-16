import logging
import os
import threading
import yaml
from contextlib import contextmanager
from typing import Optional, Generator

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, scoped_session, Session
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import SQLAlchemyError

from .models import Base

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not installed, continue without it
    pass

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Thread-safe database connection manager with Singleton pattern"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(DatabaseManager, cls).__new__(cls)
                    cls._instance._initialized_singleton = False
        return cls._instance
    
    def __init__(self):
        # Only initialize once
        if hasattr(self, '_initialized_singleton') and self._initialized_singleton:
            return
            
        self.engine: Optional[Engine] = None
        self.session_factory = None
        self.scoped_session_registry = None
        self._initialized = False
        self._initialized_singleton = True
    
    def initialize(self, database_url: Optional[str] = None) -> bool:
        """Initialize database connection"""
        try:
            if not database_url:
                database_url = self._build_database_url()
            
            logger.info(f"Initializing database connection...")
            
            # Create engine with connection pooling
            self.engine = create_engine(
                database_url,
                poolclass=QueuePool,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,
                pool_recycle=3600,
                echo=False,  # Set to True for SQL debugging
                connect_args={
                    "connect_timeout": 10,
                    "application_name": "trading_bot"
                }
            )
            
            # Create session factory
            self.session_factory = sessionmaker(
                bind=self.engine,
                autocommit=False,
                autoflush=False,
                expire_on_commit=False
            )
            
            # Create scoped session for thread safety
            self.scoped_session_registry = scoped_session(self.session_factory)
            
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            logger.info("Database connection initialized successfully")
            self._initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {str(e)}")
            return False
    
    
    def _build_database_url(self) -> str:
        """Build database URL from environment variables or YAML config"""
        environment = os.getenv('ENVIRONMENT', 'development').lower()
        use_sqlite_fallback = os.getenv('USE_SQLITE_FALLBACK', 'false').lower() in ('true', '1', 'yes')
        
        # Try to get database config from YAML file as fallback
        yaml_config = self._load_yaml_config()
        
        if environment in ('production', 'staging'):
            host = os.getenv('POSTGRES_HOST', yaml_config.get('host', 'postgres'))
            port = os.getenv('POSTGRES_PORT', str(yaml_config.get('port', 5432)))
            user = os.getenv('POSTGRES_USER', yaml_config.get('user', 'trading_bot'))
            password = os.getenv('POSTGRES_PASSWORD', yaml_config.get('password'))
            database = os.getenv('POSTGRES_DB', yaml_config.get('database', 'trading_db'))
            
            if not password:
                raise ValueError(
                    f"POSTGRES_PASSWORD environment variable is required for {environment} environment. "
                    "Set USE_SQLITE_FALLBACK=true for development/testing with SQLite."
                )
            return f"postgresql://{user}:{password}@{host}:{port}/{database}"
        
        # Development/testing: try environment variables first, then YAML config
        else:
            host = os.getenv('POSTGRES_HOST', yaml_config.get('host', 'localhost'))
            port = os.getenv('POSTGRES_PORT', str(yaml_config.get('port', 5432)))
            user = os.getenv('POSTGRES_USER', yaml_config.get('user', 'trading_bot'))
            password = os.getenv('POSTGRES_PASSWORD', yaml_config.get('password'))
            database = os.getenv('POSTGRES_DB', yaml_config.get('database', 'trading_db'))
            
            # PostgreSQL is the default and recommended database
            if password:
                logger.info(f"Using PostgreSQL for {environment} environment")
                return f"postgresql://{user}:{password}@{host}:{port}/{database}"
            
            # SQLite fallback only if explicitly requested
            elif use_sqlite_fallback:
                db_path = os.getenv('SQLITE_DB_PATH', './data/trading_bot_dev.db')
                # Ensure directory exists
                os.makedirs(os.path.dirname(db_path), exist_ok=True)
                logger.warning(
                    f"Using SQLite fallback for {environment} environment: {db_path}. "
                    "WARNING: SQLite has limitations with JSON fields and concurrent access. "
                    "PostgreSQL is strongly recommended for production workloads."
                )
                return f"sqlite:///{db_path}"
            
            else:
                raise ValueError(
                    f"No database configuration found for {environment} environment. "
                    "PostgreSQL is required by default. Set POSTGRES_PASSWORD environment variable, "
                    "or use USE_SQLITE_FALLBACK=true for development only."
                )
    
    def _load_yaml_config(self) -> dict:
        """Load database configuration from YAML file"""
        try:
            config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config', 'database.yaml')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    # Extract database config from the YAML structure
                    if 'database' in config:
                        return config['database']
                    return config
        except Exception as e:
            logger.debug(f"Could not load YAML config: {e}")
        return {}
    
    def create_tables(self) -> bool:
        """Create all database tables"""
        try:
            if not self._initialized:
                logger.error("Database not initialized")
                return False
                
            logger.info("Creating database tables...")
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create tables: {str(e)}")
            return False
    
    def get_session(self) -> Session:
        """Get thread-safe database session"""
        if not self._initialized:
            raise RuntimeError("Database not initialized")
        return self.scoped_session_registry()
    
    @contextmanager
    def session_scope(self) -> Generator[Session, None, None]:
        """Provide a transactional scope around operations"""
        session = self.get_session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    def health_check(self) -> bool:
        """Check database connection health"""
        try:
            with self.session_scope() as session:
                session.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {str(e)}")
            return False
    
    def close(self):
        """Close database connections and cleanup resources"""
        try:
            if self.scoped_session_registry:
                self.scoped_session_registry.remove()
                logger.debug("Scoped session registry cleared")
            
            if self.engine:
                self.engine.dispose()
                logger.debug("Database engine disposed")
            
            self._initialized = False
            logger.info("Database connections closed successfully")
            
        except Exception as e:
            logger.error(f"Error closing database connections: {str(e)}")
            self._initialized = False

# Global database manager instance
db_manager = DatabaseManager()

def get_db_session() -> Session:
    """Get database session (convenience function)"""
    return db_manager.get_session()