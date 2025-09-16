import logging
import os
from typing import Optional, List, Dict
import requests
import time

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """Enhanced Telegram notification service with robust error handling"""
    
    def __init__(self):
        """Initialize Telegram notifier with monitoring integrations"""
        self.bot_token = os.getenv('TELEGRAM_TOKEN')
        self.chat_ids = self._parse_chat_ids(os.getenv('TELEGRAM_CHAT_IDS', ''))
        self.api_url = f"https://api.telegram.org/bot{self.bot_token}"
        self.max_message_length = 4000
        self.retry_attempts = 3
        self.retry_delay = 1  # seconds
        
        # Initialize monitoring integrations
        self._init_monitoring()
        
        # Fallback logging for critical failures
        self._fallback_logger = self._setup_fallback_logger()
        
    def _init_monitoring(self):
        """Initialize monitoring integrations with proper fallbacks"""
        # Try to initialize Sentry
        self.sentry_available = False
        try:
            import sentry_sdk
            self.sentry_available = True
            logger.info("Sentry integration available for Telegram notifications")
        except ImportError:
            logger.debug("Sentry not available - falling back to file logging")
        
        # Try to initialize Prometheus
        self.prometheus_available = False
        try:
            from prometheus_client import Counter, Histogram
            self.telegram_failures = Counter(
                'telegram_notifications_failed_total', 
                'Total failed Telegram notifications',
                ['chat_id', 'error_type']
            )
            self.telegram_success = Counter(
                'telegram_notifications_sent_total', 
                'Total successful Telegram notifications',
                ['chat_id']
            )
            self.telegram_latency = Histogram(
                'telegram_notification_duration_seconds',
                'Telegram notification response time'
            )
            self.prometheus_available = True
            logger.info("Prometheus metrics available for Telegram notifications")
        except ImportError:
            logger.debug("Prometheus not available - using internal counters")
            # Initialize internal counters as fallback
            self._internal_metrics = {
                'failures': 0,
                'successes': 0,
                'last_failure_time': None,
                'consecutive_failures': 0
            }
    
    def _setup_fallback_logger(self):
        """Setup dedicated logger for Telegram failures"""
        fallback_logger = logging.getLogger('telegram_fallback')
        
        # Only add handler if not already configured
        if not fallback_logger.handlers:
            try:
                # Create dedicated log file for Telegram failures
                log_file = os.path.join(os.getenv('LOG_DIR', './logs'), 'telegram_failures.log')
                os.makedirs(os.path.dirname(log_file), exist_ok=True)
                
                handler = logging.FileHandler(log_file)
                formatter = logging.Formatter(
                    '%(asctime)s - %(levelname)s - %(message)s'
                )
                handler.setFormatter(formatter)
                fallback_logger.addHandler(handler)
                fallback_logger.setLevel(logging.WARNING)
                
            except Exception as e:
                logger.warning(f"Could not setup Telegram fallback logger: {e}")
                return None
        
        return fallback_logger
        
    def _parse_chat_ids(self, chat_ids_str: str) -> List[str]:
        """Parse comma-separated chat IDs"""
        if not chat_ids_str:
            return []
        return [chat_id.strip() for chat_id in chat_ids_str.split(',') if chat_id.strip()]
    
    def is_configured(self) -> bool:
        """Check if Telegram notifier is properly configured"""
        return bool(self.bot_token and self.chat_ids)
    
    def send_message(self, message: str, chat_id: Optional[str] = None, parse_mode: str = 'Markdown') -> bool:
        """
        Send message to Telegram chat
        
        Args:
            message: Message text to send
            chat_id: Specific chat ID (optional, uses all configured if None)
            parse_mode: Message formatting (Markdown or HTML)
            
        Returns:
            bool: True if message sent successfully to at least one chat
        """
        if not self.is_configured():
            logger.warning("Telegram notifier not configured - skipping notification")
            return False
        
        # Truncate message if too long
        if len(message) > self.max_message_length:
            message = message[:self.max_message_length-10] + "...[truncated]"
        
        # Determine target chat IDs
        target_chats = [chat_id] if chat_id else self.chat_ids
        success_count = 0
        
        for chat in target_chats:
            if self._send_to_chat(message, chat, parse_mode):
                success_count += 1
            else:
                # Small delay between failed attempts
                time.sleep(0.5)
        
        return success_count > 0
    
    def _send_to_chat(self, message: str, chat_id: str, parse_mode: str) -> bool:
        """Send message to specific chat with enhanced retry logic and monitoring"""
        url = f"{self.api_url}/sendMessage"
        data = {
            'chat_id': chat_id,
            'text': message,
            'parse_mode': parse_mode
        }
        
        start_time = time.time()
        last_error = None
        
        for attempt in range(self.retry_attempts):
            try:
                response = requests.post(url, data=data, timeout=10)
                
                if response.status_code == 200:
                    # Record success metrics
                    self._record_success(chat_id, time.time() - start_time)
                    logger.debug(f"Message sent to chat {chat_id}")
                    return True
                else:
                    last_error = f"HTTP {response.status_code}: {response.text}"
                    logger.warning(f"Telegram API error for chat {chat_id}: {last_error}")
                    
            except requests.exceptions.RequestException as e:
                last_error = f"Network error: {str(e)}"
                logger.warning(f"Network error sending to chat {chat_id} (attempt {attempt + 1}): {last_error}")
            
            # Wait before retry (except last attempt)
            if attempt < self.retry_attempts - 1:
                time.sleep(self.retry_delay * (attempt + 1))
        
        # Record failure and send to monitoring with comprehensive fallbacks
        self._record_failure(chat_id, last_error, message)
        
        return False
    
    def _record_success(self, chat_id: str, duration: float):
        """Record successful notification with fallback metrics"""
        # Update Prometheus metrics if available
        if self.prometheus_available:
            try:
                self.telegram_success.labels(chat_id=chat_id).inc()
                self.telegram_latency.observe(duration)
            except Exception as e:
                logger.debug(f"Failed to update Prometheus success metrics: {e}")
        
        # Update internal metrics as fallback
        if hasattr(self, '_internal_metrics'):
            self._internal_metrics['successes'] += 1
            self._internal_metrics['consecutive_failures'] = 0
    
    def _record_failure(self, chat_id: str, error: str, message: str):
        """Record failure with comprehensive monitoring fallbacks"""
        critical_msg = f"CRITICAL: Failed to send message to chat {chat_id} after {self.retry_attempts} attempts. Error: {error}"
        logger.critical(critical_msg)
        
        # Update Prometheus metrics if available
        if self.prometheus_available:
            try:
                error_type = self._classify_error(error)
                self.telegram_failures.labels(chat_id=chat_id, error_type=error_type).inc()
            except Exception as e:
                logger.debug(f"Failed to update Prometheus failure metrics: {e}")
        
        # Update internal metrics as fallback
        if hasattr(self, '_internal_metrics'):
            self._internal_metrics['failures'] += 1
            self._internal_metrics['consecutive_failures'] += 1
            self._internal_metrics['last_failure_time'] = time.time()
        
        # Send to Sentry if available
        if self.sentry_available:
            try:
                import sentry_sdk
                sentry_sdk.capture_message(
                    f"Telegram notification failure: {chat_id}",
                    level="error",
                    extra={
                        "chat_id": chat_id,
                        "attempts": self.retry_attempts,
                        "error": error,
                        "message_preview": message[:100] if len(message) > 100 else message,
                        "consecutive_failures": self._internal_metrics.get('consecutive_failures', 0)
                    }
                )
            except Exception as e:
                logger.debug(f"Failed to send to Sentry: {e}")
                self._log_to_fallback("sentry_error", {"error": str(e), "chat_id": chat_id})
        
        # Log to dedicated fallback file
        self._log_to_fallback("telegram_failure", {
            "chat_id": chat_id,
            "error": error,
            "attempts": self.retry_attempts,
            "message_preview": message[:100] if len(message) > 100 else message
        })
        
        # Check for critical failure patterns
        self._check_critical_patterns()
    
    def _classify_error(self, error: str) -> str:
        """Classify error type for metrics"""
        if "Network error" in error:
            return "network"
        elif "HTTP 4" in error:
            return "client_error"
        elif "HTTP 5" in error:
            return "server_error"
        elif "timeout" in error.lower():
            return "timeout"
        else:
            return "unknown"
    
    def _log_to_fallback(self, event_type: str, data: dict):
        """Log to fallback logger with structured data"""
        if self._fallback_logger:
            try:
                log_entry = {
                    "timestamp": time.time(),
                    "event_type": event_type,
                    "data": data
                }
                self._fallback_logger.error(f"TELEGRAM_FAILURE: {log_entry}")
            except Exception as e:
                # Last resort - use main logger
                logger.error(f"Fallback logging failed: {e}, Original data: {data}")
    
    def _check_critical_patterns(self):
        """Check for critical failure patterns that need immediate attention"""
        if not hasattr(self, '_internal_metrics'):
            return
            
        consecutive_failures = self._internal_metrics.get('consecutive_failures', 0)
        
        # Alert on consecutive failures
        if consecutive_failures >= 5:
            alert_msg = f"CRITICAL: {consecutive_failures} consecutive Telegram notification failures"
            logger.critical(alert_msg)
            
            # Try to send alert via email or other channels if configured
            self._send_critical_alert(alert_msg)
    
    def _send_critical_alert(self, message: str):
        """Send critical alert via alternative channels"""
        # This could be expanded to use email, webhooks, etc.
        try:
            # Example: Send to a backup webhook
            backup_webhook = os.getenv('BACKUP_ALERT_WEBHOOK')
            if backup_webhook:
                import requests
                requests.post(
                    backup_webhook, 
                    json={"text": message, "level": "critical"},
                    timeout=5
                )
        except Exception as e:
            logger.debug(f"Failed to send critical alert: {e}")
    
    def get_metrics_summary(self) -> dict:
        """Get summary of notification metrics for monitoring"""
        summary = {
            "prometheus_available": self.prometheus_available,
            "sentry_available": self.sentry_available,
        }
        
        if hasattr(self, '_internal_metrics'):
            summary.update(self._internal_metrics)
            
        return summary
    
    def send_trade_notification(self, trade_data: Dict) -> bool:
        """Send formatted trade notification"""
        try:
            if trade_data.get('success', False):
                message = self._format_success_message(trade_data)
            else:
                message = self._format_error_message(trade_data)
            
            return self.send_message(message)
            
        except Exception as e:
            logger.error(f"Error formatting trade notification: {str(e)}")
            return False
    
    def _format_success_message(self, trade_data: Dict) -> str:
        """Format successful trade message"""
        symbol = trade_data.get('symbol', 'Unknown')
        signal = trade_data.get('signal', 'Unknown')
        entry_price = trade_data.get('price', 'Unknown')
        volume = trade_data.get('volume', 'Unknown')
        ticket = trade_data.get('ticket', 'Unknown')
        
        message = f"""🟢 *Trade Executed Successfully!*

📊 *Symbol:* `{symbol}`
🎯 *Signal:* `{signal}`
💰 *Entry Price:* `{entry_price}`
📈 *Volume:* `{volume}`
🎫 *Ticket:* `{ticket}`
⏰ *Time:* `{trade_data.get('execution_time', 'Unknown')}`"""

        # Add SL/TP if available
        if 'stop_loss' in trade_data:
            message += f"\n🛡️ *Stop Loss:* `{trade_data['stop_loss']}`"
        if 'take_profit' in trade_data:
            message += f"\n🎯 *Take Profit:* `{trade_data['take_profit']}`"
            
        return message
    
    def _format_error_message(self, trade_data: Dict) -> str:
        """Format failed trade message"""
        symbol = trade_data.get('symbol', 'Unknown')
        error = trade_data.get('error', 'Unknown error')
        
        message = f"""🔴 *Trade Execution Failed!*

📊 *Symbol:* `{symbol}`
❌ *Error:* `{error}`
⏰ *Time:* `{trade_data.get('failed_at', 'Unknown')}`"""

        return message
    
    def send_alert(self, alert_type: str, message: str, priority: str = 'normal') -> bool:
        """
        Send system alert
        
        Args:
            alert_type: Type of alert (error, warning, info)
            message: Alert message
            priority: Priority level (low, normal, high, critical)
        """
        emoji_map = {
            'error': '🚨',
            'warning': '⚠️',
            'info': 'ℹ️',
            'success': '✅'
        }
        
        emoji = emoji_map.get(alert_type, '📢')
        
        if priority == 'critical':
            formatted_message = f"{emoji} *CRITICAL ALERT*\n\n{message}"
        elif priority == 'high':
            formatted_message = f"{emoji} *HIGH PRIORITY*\n\n{message}"
        else:
            formatted_message = f"{emoji} *{alert_type.upper()}*\n\n{message}"
        
        return self.send_message(formatted_message)
    
    def send_status_report(self, status_data: Dict) -> bool:
        """Send system status report"""
        try:
            account_balance = status_data.get('account_balance', 'Unknown')
            open_positions = status_data.get('open_positions', 0)
            profit_loss = status_data.get('profit_loss', 'Unknown')
            
            message = f"""📊 *Trading Bot Status*

💰 *Account Balance:* `{account_balance}`
📈 *Open Positions:* `{open_positions}`
💵 *Profit/Loss:* `{profit_loss}`
⏰ *Last Update:* `{status_data.get('timestamp', 'Unknown')}`"""

            if 'daily_trades' in status_data:
                message += f"\n📊 *Daily Trades:* `{status_data['daily_trades']}`"
            
            if 'system_health' in status_data:
                health = status_data['system_health']
                health_emoji = '🟢' if health == 'good' else '🟡' if health == 'warning' else '🔴'
                message += f"\n{health_emoji} *System Health:* `{health}`"
            
            return self.send_message(message)
            
        except Exception as e:
            logger.error(f"Error formatting status report: {str(e)}")
            return False


# Global notifier instance
_notifier = None

def get_notifier() -> TelegramNotifier:
    """Get global notifier instance (singleton pattern)"""
    global _notifier
    if _notifier is None:
        _notifier = TelegramNotifier()
    return _notifier


def send_notification(message: str, alert_type: str = 'info', priority: str = 'normal') -> bool:
    """
    Convenience function for sending notifications
    
    Args:
        message: Message to send
        alert_type: Type of alert (error, warning, info, success)
        priority: Priority level (low, normal, high, critical)
        
    Returns:
        bool: True if sent successfully
    """
    try:
        notifier = get_notifier()
        return notifier.send_alert(alert_type, message, priority)
    except Exception as e:
        logger.error(f"Error in send_notification: {str(e)}")
        return False


def send_trade_alert(trade_data: Dict) -> bool:
    """
    Convenience function for sending trade notifications
    
    Args:
        trade_data: Dictionary containing trade information
        
    Returns:
        bool: True if sent successfully
    """
    try:
        notifier = get_notifier()
        return notifier.send_trade_notification(trade_data)
    except Exception as e:
        logger.error(f"Error in send_trade_alert: {str(e)}")
        return False


def send_status_update(status_data: Dict) -> bool:
    """
    Convenience function for sending status updates
    
    Args:
        status_data: Dictionary containing status information
        
    Returns:
        bool: True if sent successfully
    """
    try:
        notifier = get_notifier()
        return notifier.send_status_report(status_data)
    except Exception as e:
        logger.error(f"Error in send_status_update: {str(e)}")
        return False