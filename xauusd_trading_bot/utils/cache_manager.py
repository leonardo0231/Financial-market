import json
import logging
import hashlib
from typing import Any, Optional, Dict, Callable, Union
from datetime import datetime, timedelta
import redis
from functools import wraps

logger = logging.getLogger(__name__)


class CacheManager:
    """Thread-safe cache manager with Redis backend"""
    
    def __init__(self, redis_client: redis.Redis):
        """Initialize cache manager with Redis client"""
        self.redis = redis_client
        self.default_ttl = 300  # 5 minutes default
        self.prefix = "trading_bot:"
    
    def _json_serializer(self, obj: Any) -> Union[str, Dict, None]:
        """Custom JSON serializer for non-serializable objects"""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        elif hasattr(obj, 'to_dict'):
            return obj.to_dict()
        # Return None for non-serializable objects (will be skipped)
        return None
        
    def cache_key(self, namespace: str, *args, **kwargs) -> str:
        """Generate consistent cache key from arguments"""
        key_data = {
            'namespace': namespace,
            'args': args,
            'kwargs': kwargs
        }
        key_str = json.dumps(key_data, sort_keys=True)
        key_hash = hashlib.md5(key_str.encode()).hexdigest()
        return f"{self.prefix}{namespace}:{key_hash}"
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            cached_data = self.redis.get(key)
            if cached_data:
                # Decode and parse JSON safely
                if isinstance(cached_data, bytes):
                    cached_data = cached_data.decode('utf-8')
                return json.loads(cached_data)
            return None
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.warning(f"Cache decode error for key {key}: {e}")
            return None
        except Exception as e:
            logger.warning(f"Cache get error for key {key}: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache with TTL"""
        try:
            ttl = ttl or self.default_ttl
            # Convert to JSON-serializable format
            serialized = json.dumps(value, default=self._json_serializer)
            return self.redis.setex(key, ttl, serialized)
        except (TypeError, ValueError) as e:
            logger.warning(f"Cache serialization error for key {key}: {e}")
            return False
        except Exception as e:
            logger.warning(f"Cache set error for key {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        try:
            return bool(self.redis.delete(key))
        except Exception as e:
            logger.warning(f"Cache delete error for key {key}: {e}")
            return False
    
    def clear_namespace(self, namespace: str) -> int:
        """Clear all keys in namespace"""
        try:
            pattern = f"{self.prefix}{namespace}:*"
            keys = self.redis.keys(pattern)
            if keys:
                return self.redis.delete(*keys)
            return 0
        except Exception as e:
            logger.warning(f"Cache clear error for namespace {namespace}: {e}")
            return 0
    
    def memoize(self, ttl: Optional[int] = None, namespace: Optional[str] = None):
        """Decorator for caching function results"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key
                cache_namespace = namespace or func.__name__
                cache_key = self.cache_key(cache_namespace, *args, **kwargs)
                
                # Try to get from cache
                cached_result = self.get(cache_key)
                if cached_result is not None:
                    logger.debug(f"Cache hit for {cache_namespace}")
                    return cached_result
                
                # Calculate result
                result = func(*args, **kwargs)
                
                # Store in cache
                self.set(cache_key, result, ttl)
                logger.debug(f"Cache miss for {cache_namespace}, stored result")
                
                return result
            return wrapper
        return decorator