"""
Redis cache implementation for ML models.

This module provides Redis-based caching for distributed systems,
with support for model predictions, feature engineering results,
and cross-service data sharing.
"""

import json
import pickle
import time
from typing import Any, Dict, Optional, List
import hashlib
import logging

logger = logging.getLogger(__name__)

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis not available. Install redis-py: pip install redis")


class RedisCache:
    """
    Redis-based cache for ML models with serialization support.
    """
    
    def __init__(
        self,
        host: str = 'localhost',
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        decode_responses: bool = False,
        default_ttl: int = 3600,
        key_prefix: str = 'ml_cache:'
    ):
        """
        Initialize Redis cache.
        
        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            password: Redis password
            decode_responses: Whether to decode responses
            default_ttl: Default time-to-live in seconds
            key_prefix: Prefix for all cache keys
        """
        if not REDIS_AVAILABLE:
            raise ImportError("Redis is not available. Install redis-py: pip install redis")
        
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.default_ttl = default_ttl
        self.key_prefix = key_prefix
        
        # Initialize Redis connection
        self.redis_client = redis.Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            decode_responses=decode_responses,
            socket_connect_timeout=5,
            socket_timeout=5
        )
        
        # Test connection
        try:
            self.redis_client.ping()
            logger.info(f"Connected to Redis at {host}:{port}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
        
        self._stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'errors': 0
        }
    
    def _generate_key(self, key: str) -> str:
        """Generate a prefixed key for Redis."""
        return f"{self.key_prefix}{key}"
    
    def _serialize_value(self, value: Any) -> bytes:
        """Serialize value for Redis storage."""
        try:
            return pickle.dumps(value)
        except Exception as e:
            logger.error(f"Failed to serialize value: {e}")
            raise
    
    def _deserialize_value(self, value: bytes) -> Any:
        """Deserialize value from Redis."""
        try:
            return pickle.loads(value)
        except Exception as e:
            logger.error(f"Failed to deserialize value: {e}")
            raise
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get item from Redis cache.
        
        Args:
            key: Cache key
            default: Default value if key not found
            
        Returns:
            Cached value or default
        """
        try:
            redis_key = self._generate_key(key)
            value = self.redis_client.get(redis_key)
            
            if value is not None:
                self._stats['hits'] += 1
                return self._deserialize_value(value)
            
            self._stats['misses'] += 1
            return default
            
        except Exception as e:
            logger.error(f"Redis GET error for key {key}: {e}")
            self._stats['errors'] += 1
            return default
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set item in Redis cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds
            
        Returns:
            True if successful, False otherwise
        """
        try:
            redis_key = self._generate_key(key)
            serialized_value = self._serialize_value(value)
            ttl = ttl or self.default_ttl
            
            result = self.redis_client.setex(redis_key, ttl, serialized_value)
            
            if result:
                self._stats['sets'] += 1
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Redis SET error for key {key}: {e}")
            self._stats['errors'] += 1
            return False
    
    def delete(self, key: str) -> bool:
        """
        Delete item from Redis cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if item was deleted, False if not found
        """
        try:
            redis_key = self._generate_key(key)
            result = self.redis_client.delete(redis_key)
            return result > 0
            
        except Exception as e:
            logger.error(f"Redis DELETE error for key {key}: {e}")
            self._stats['errors'] += 1
            return False
    
    def exists(self, key: str) -> bool:
        """
        Check if key exists in Redis cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if key exists, False otherwise
        """
        try:
            redis_key = self._generate_key(key)
            return self.redis_client.exists(redis_key) > 0
            
        except Exception as e:
            logger.error(f"Redis EXISTS error for key {key}: {e}")
            self._stats['errors'] += 1
            return False
    
    def clear(self, pattern: Optional[str] = None) -> int:
        """
        Clear items from Redis cache.
        
        Args:
            pattern: Optional pattern to match keys (e.g., 'model:*')
            
        Returns:
            Number of keys deleted
        """
        try:
            if pattern:
                search_pattern = self._generate_key(pattern)
            else:
                search_pattern = f"{self.key_prefix}*"
            
            keys = self.redis_client.keys(search_pattern)
            
            if keys:
                return self.redis_client.delete(*keys)
            
            return 0
            
        except Exception as e:
            logger.error(f"Redis CLEAR error: {e}")
            self._stats['errors'] += 1
            return 0
    
    def get_ttl(self, key: str) -> int:
        """
        Get time-to-live for a key.
        
        Args:
            key: Cache key
            
        Returns:
            TTL in seconds (-1 if no TTL, -2 if key doesn't exist)
        """
        try:
            redis_key = self._generate_key(key)
            return self.redis_client.ttl(redis_key)
            
        except Exception as e:
            logger.error(f"Redis TTL error for key {key}: {e}")
            self._stats['errors'] += 1
            return -2
    
    def extend_ttl(self, key: str, ttl: int) -> bool:
        """
        Extend time-to-live for a key.
        
        Args:
            key: Cache key
            ttl: New TTL in seconds
            
        Returns:
            True if successful, False otherwise
        """
        try:
            redis_key = self._generate_key(key)
            return self.redis_client.expire(redis_key, ttl)
            
        except Exception as e:
            logger.error(f"Redis EXPIRE error for key {key}: {e}")
            self._stats['errors'] += 1
            return False
    
    def get_keys(self, pattern: str = '*') -> List[str]:
        """
        Get all keys matching a pattern.
        
        Args:
            pattern: Key pattern (e.g., 'model:*')
            
        Returns:
            List of matching keys (without prefix)
        """
        try:
            search_pattern = self._generate_key(pattern)
            keys = self.redis_client.keys(search_pattern)
            
            # Remove prefix from keys
            prefix_len = len(self.key_prefix)
            return [key.decode() if isinstance(key, bytes) else key[prefix_len:] for key in keys]
            
        except Exception as e:
            logger.error(f"Redis KEYS error: {e}")
            self._stats['errors'] += 1
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        try:
            redis_info = self.redis_client.info()
            hit_rate = self._stats['hits'] / (self._stats['hits'] + self._stats['misses']) if (self._stats['hits'] + self._stats['misses']) > 0 else 0
            
            return {
                **self._stats,
                'hit_rate': hit_rate,
                'redis_memory_usage': redis_info.get('used_memory_human', 'N/A'),
                'redis_connected_clients': redis_info.get('connected_clients', 0),
                'redis_uptime_seconds': redis_info.get('uptime_in_seconds', 0)
            }
            
        except Exception as e:
            logger.error(f"Redis INFO error: {e}")
            return {**self._stats, 'hit_rate': 0.0}
    
    def ping(self) -> bool:
        """
        Test Redis connection.
        
        Returns:
            True if connection is alive, False otherwise
        """
        try:
            return self.redis_client.ping()
        except Exception as e:
            logger.error(f"Redis PING error: {e}")
            return False
    
    def flush_db(self) -> bool:
        """
        Flush current Redis database.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            return self.redis_client.flushdb()
        except Exception as e:
            logger.error(f"Redis FLUSHDB error: {e}")
            return False


class RedisPredictionCache(RedisCache):
    """
    Specialized Redis cache for model predictions.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize Redis prediction cache.
        
        Args:
            **kwargs: Arguments for RedisCache
        """
        kwargs.setdefault('key_prefix', 'ml_predictions:')
        kwargs.setdefault('default_ttl', 1800)  # 30 minutes
        super().__init__(**kwargs)
    
    def cache_prediction(
        self,
        model_name: str,
        input_hash: str,
        prediction: Dict[str, Any],
        execution_time: float,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Cache a model prediction.
        
        Args:
            model_name: Name of the model
            input_hash: Hash of input data
            prediction: Prediction result
            execution_time: Time taken to generate prediction
            ttl: Time-to-live for cached prediction
            
        Returns:
            True if successful, False otherwise
        """
        cache_key = f"{model_name}:{input_hash}"
        
        prediction_data = {
            'prediction': prediction,
            'execution_time': execution_time,
            'cached_at': time.time(),
            'model_name': model_name
        }
        
        success = self.set(cache_key, prediction_data, ttl)
        
        if success:
            logger.debug(f"Cached prediction for {model_name} with key {input_hash}")
        
        return success
    
    def get_prediction(
        self,
        model_name: str,
        input_hash: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached prediction.
        
        Args:
            model_name: Name of the model
            input_hash: Hash of input data
            
        Returns:
            Cached prediction data or None
        """
        cache_key = f"{model_name}:{input_hash}"
        prediction_data = self.get(cache_key)
        
        if prediction_data:
            logger.debug(f"Retrieved cached prediction for {model_name}")
            return prediction_data
        
        return None
    
    def clear_model_predictions(self, model_name: str) -> int:
        """
        Clear all cached predictions for a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Number of predictions cleared
        """
        pattern = f"{model_name}:*"
        return self.clear(pattern)


class RedisFeatureCache(RedisCache):
    """
    Specialized Redis cache for feature engineering results.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize Redis feature cache.
        
        Args:
            **kwargs: Arguments for RedisCache
        """
        kwargs.setdefault('key_prefix', 'ml_features:')
        kwargs.setdefault('default_ttl', 7200)  # 2 hours
        super().__init__(**kwargs)
    
    def cache_features(
        self,
        data_hash: str,
        processing_config: Dict[str, Any],
        features: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> bool:
        """
        Cache processed features.
        
        Args:
            data_hash: Hash of input data
            processing_config: Feature processing configuration
            features: Processed features
            ttl: Time-to-live for cached features
            
        Returns:
            True if successful, False otherwise
        """
        config_hash = hashlib.md5(str(processing_config).encode()).hexdigest()
        cache_key = f"features:{data_hash}:{config_hash}"
        
        feature_data = {
            'features': features,
            'processing_config': processing_config,
            'cached_at': time.time(),
            'data_hash': data_hash
        }
        
        success = self.set(cache_key, feature_data, ttl)
        
        if success:
            logger.debug(f"Cached features for data hash {data_hash}")
        
        return success
    
    def get_features(
        self,
        data_hash: str,
        processing_config: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached features.
        
        Args:
            data_hash: Hash of input data
            processing_config: Feature processing configuration
            
        Returns:
            Cached feature data or None
        """
        config_hash = hashlib.md5(str(processing_config).encode()).hexdigest()
        cache_key = f"features:{data_hash}:{config_hash}"
        
        feature_data = self.get(cache_key)
        
        if feature_data:
            logger.debug(f"Retrieved cached features for data hash {data_hash}")
            return feature_data
        
        return None
