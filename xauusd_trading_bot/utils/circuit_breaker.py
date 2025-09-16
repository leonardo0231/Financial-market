import time
import logging
from typing import Callable, Optional, Any, Dict
from functools import wraps
from enum import Enum
import threading

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, reject calls
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """Thread-safe circuit breaker implementation"""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type = Exception,
        name: Optional[str] = None
    ):
        """
        Initialize circuit breaker
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds before attempting recovery
            expected_exception: Exception type to catch
            name: Circuit breaker name for logging
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.name = name or "CircuitBreaker"
        
        self._failure_count = 0
        self._last_failure_time = None
        self._state = CircuitState.CLOSED
        self._lock = threading.Lock()
        
    @property
    def state(self) -> CircuitState:
        """Get current circuit state"""
        with self._lock:
            self._update_state()
            return self._state
    
    def _update_state(self) -> None:
        """Update circuit state based on failures and timeout"""
        if self._state == CircuitState.OPEN:
            if self._last_failure_time and \
               time.time() - self._last_failure_time >= self.recovery_timeout:
                logger.info(f"{self.name}: Attempting recovery (HALF_OPEN)")
                self._state = CircuitState.HALF_OPEN
    
    def _record_success(self) -> None:
        """Record successful call"""
        with self._lock:
            self._failure_count = 0
            if self._state == CircuitState.HALF_OPEN:
                logger.info(f"{self.name}: Recovery successful (CLOSED)")
                self._state = CircuitState.CLOSED
    
    def _record_failure(self) -> None:
        """Record failed call"""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            
            if self._failure_count >= self.failure_threshold:
                logger.warning(f"{self.name}: Opening circuit after {self._failure_count} failures")
                self._state = CircuitState.OPEN
            elif self._state == CircuitState.HALF_OPEN:
                logger.warning(f"{self.name}: Recovery failed, reopening circuit")
                self._state = CircuitState.OPEN
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        if self.state == CircuitState.OPEN:
            raise Exception(f"{self.name}: Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result
        except self.expected_exception as e:
            self._record_failure()
            raise
    
    def decorator(self, func: Callable) -> Callable:
        """Decorator for protecting functions with circuit breaker"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        return wrapper
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics"""
        with self._lock:
            return {
                'name': self.name,
                'state': self._state.value,
                'failure_count': self._failure_count,
                'failure_threshold': self.failure_threshold,
                'recovery_timeout': self.recovery_timeout,
                'last_failure_time': self._last_failure_time
            }


# Global circuit breakers for different services
mt5_circuit_breaker = CircuitBreaker(
    failure_threshold=3,
    recovery_timeout=30,
    name="MT5_API"
)

api_circuit_breaker = CircuitBreaker(
    failure_threshold=5,
    recovery_timeout=60,
    name="External_API"
)

db_circuit_breaker = CircuitBreaker(
    failure_threshold=10,
    recovery_timeout=120,
    name="Database"
)