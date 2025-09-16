#!/usr/bin/env python3
"""
Test script to verify database connection configuration
"""
import os
import sys
import logging

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_database_connection():
    """Test database connection with current configuration"""
    try:
        from xauusd_trading_bot.database.connection import db_manager
        
        logger.info("Testing database connection...")
        
        # Test initialization
        if db_manager.initialize():
            logger.info("✅ Database connection successful!")
            
            # Test health check
            if db_manager.health_check():
                logger.info("✅ Database health check passed!")
                
                # Test table creation
                if db_manager.create_tables():
                    logger.info("✅ Database tables created successfully!")
                    return True
                else:
                    logger.error("❌ Failed to create database tables")
                    return False
            else:
                logger.error("❌ Database health check failed")
                return False
        else:
            logger.error("❌ Failed to initialize database connection")
            return False
            
    except Exception as e:
        logger.error(f"❌ Database connection test failed: {e}")
        return False
    finally:
        # Clean up
        try:
            db_manager.close()
        except:
            pass

def test_yaml_config():
    """Test YAML configuration loading"""
    try:
        from xauusd_trading_bot.database.connection import DatabaseManager
        
        logger.info("Testing YAML configuration loading...")
        db_manager = DatabaseManager()
        yaml_config = db_manager._load_yaml_config()
        
        if yaml_config:
            logger.info(f"✅ YAML config loaded: {yaml_config}")
            return True
        else:
            logger.warning("⚠️ No YAML config found")
            return False
            
    except Exception as e:
        logger.error(f"❌ YAML config test failed: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Testing Database Configuration")
    print("=" * 50)
    
    # Test YAML config first
    yaml_ok = test_yaml_config()
    print()
    
    # Test database connection
    db_ok = test_database_connection()
    print()
    
    if db_ok:
        print("🎉 All tests passed! Database is ready to use.")
        sys.exit(0)
    else:
        print("💥 Some tests failed. Check the logs above for details.")
        sys.exit(1)
