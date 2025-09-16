import os
import glob
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class LogManager:
    """Advanced log management with automatic cleanup"""
    
    def __init__(self, log_dir: str = "logs", max_age_days: int = 7, max_size_mb: int = 100):
        self.log_dir = Path(log_dir)
        self.max_age_days = max_age_days
        self.max_size_mb = max_size_mb
        self.log_dir.mkdir(exist_ok=True)
    
    def cleanup_old_logs(self) -> int:
        """
        Clean up log files older than max_age_days
        Returns number of files deleted
        """
        deleted_count = 0
        
        try:
            cutoff_date = datetime.now() - timedelta(days=self.max_age_days)
            
            # Find all log files - only trading-specific patterns to avoid deleting system files
            log_patterns = ['*.log', '*.log.*', 'trading*.out', 'trading*.err', 'xauusd*.out', 'xauusd*.err']
            
            for pattern in log_patterns:
                log_files = self.log_dir.glob(pattern)
                
                for log_file in log_files:
                    try:
                        # Skip currently active log file
                        if log_file.name == 'trading.log':
                            continue
                            
                        file_stat = log_file.stat()
                        file_date = datetime.fromtimestamp(file_stat.st_mtime)
                        
                        if file_date < cutoff_date:
                            log_file.unlink()
                            logger.info(f"Cleaned up old log file: {log_file.name}")
                            deleted_count += 1
                            
                    except Exception as e:
                        logger.warning(f"Failed to clean up log file {log_file}: {e}")
                        
        except Exception as e:
            logger.error(f"Error during log cleanup: {e}")
            
        return deleted_count
    
    def cleanup_by_size(self) -> int:
        """
        Clean up oldest log files if total size exceeds max_size_mb
        Returns number of files deleted
        """
        deleted_count = 0
        
        try:
            # Calculate total size
            total_size_bytes = 0
            log_files = []
            
            for log_file in self.log_dir.glob('*.log*'):
                if log_file.is_file():
                    size = log_file.stat().st_size
                    mtime = log_file.stat().st_mtime
                    log_files.append((log_file, size, mtime))
                    total_size_bytes += size
            
            max_size_bytes = self.max_size_mb * 1024 * 1024
            
            if total_size_bytes > max_size_bytes:
                # Sort by modification time (oldest first)
                log_files.sort(key=lambda x: x[2])
                
                # Delete oldest files until under size limit
                for log_file, size, _ in log_files:
                    if total_size_bytes <= max_size_bytes:
                        break
                        
                    # Skip currently active log file
                    if log_file.name == 'trading.log':
                        continue
                    
                    try:
                        log_file.unlink()
                        total_size_bytes -= size
                        deleted_count += 1
                        logger.info(f"Cleaned up log file for size limit: {log_file.name}")
                    except Exception as e:
                        logger.warning(f"Failed to delete log file {log_file}: {e}")
                        
        except Exception as e:
            logger.error(f"Error during size-based log cleanup: {e}")
            
        return deleted_count
    
    def get_log_stats(self) -> dict:
        """Get statistics about log files"""
        try:
            log_files = list(self.log_dir.glob('*.log*'))
            total_size = sum(f.stat().st_size for f in log_files if f.is_file())
            
            return {
                'total_files': len(log_files),
                'total_size_mb': round(total_size / (1024 * 1024), 2),
                'oldest_file': min(log_files, key=lambda f: f.stat().st_mtime).name if log_files else None,
                'newest_file': max(log_files, key=lambda f: f.stat().st_mtime).name if log_files else None
            }
        except Exception as e:
            logger.error(f"Error getting log stats: {e}")
            return {}
    
    def compress_old_logs(self, days_old: int = 3) -> int:
        """
        Compress log files older than specified days
        Returns number of files compressed
        """
        compressed_count = 0
        
        try:
            import gzip
            import shutil
            
            cutoff_date = datetime.now() - timedelta(days=days_old)
            
            for log_file in self.log_dir.glob('*.log'):
                # Skip main log file and already compressed files
                if log_file.name in ['trading.log'] or log_file.suffix == '.gz':
                    continue
                
                file_stat = log_file.stat()
                file_date = datetime.fromtimestamp(file_stat.st_mtime)
                
                if file_date < cutoff_date:
                    compressed_file = log_file.with_suffix(log_file.suffix + '.gz')
                    
                    try:
                        with open(log_file, 'rb') as f_in:
                            with gzip.open(compressed_file, 'wb') as f_out:
                                shutil.copyfileobj(f_in, f_out)
                        
                        log_file.unlink()
                        compressed_count += 1
                        logger.info(f"Compressed log file: {log_file.name} -> {compressed_file.name}")
                        
                    except Exception as e:
                        logger.warning(f"Failed to compress {log_file}: {e}")
                        
        except ImportError:
            logger.warning("gzip module not available for log compression")
        except Exception as e:
            logger.error(f"Error during log compression: {e}")
            
        return compressed_count



def setup_log_cleanup_scheduler():
    """Set up automatic log cleanup (call from main application)"""
    import threading
    import time
    
    def cleanup_worker():
        while True:
            try:
                manager = LogManager()
                deleted_age = manager.cleanup_old_logs()
                deleted_size = manager.cleanup_by_size()
                compressed = manager.compress_old_logs()
                
                if deleted_age + deleted_size + compressed > 0:
                    logger.info(f"Log cleanup completed: {deleted_age} old files, "
                              f"{deleted_size} oversized files, {compressed} compressed")
                
                # Sleep for 24 hours
                time.sleep(24 * 60 * 60)
                
            except Exception as e:
                logger.error(f"Log cleanup worker error: {e}")
                time.sleep(60 * 60)  # Retry in 1 hour
    
    # Start cleanup worker in background thread
    cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
    cleanup_thread.start()
    logger.info("Log cleanup scheduler started")