"""
Cache manager for ML models.

This module provides a unified cache management interface that can work
with multiple cache backends (memory, Redis) and provides intelligent
cache selection and fallback mechanisms.
"""

import time
import hashlib
import logging
from typing import Any, Dict, Optional, List, Union
from enum import Enum

from .memory_cache import MemoryCache, PredictionCache, FeatureCache
from .redis_cache import RedisCache, RedisPredictionCache, RedisFeatureCache, REDIS_AVAILABLE

logger = logging.getLogger(__name__)


class CacheBackend(Enum):
    """Cache backend types."""
    MEMORY = "memory"
    REDIS = "redis"
    HYBRID = "hybrid"


class CacheManager:
    """
    Unified cache manager with multiple backend support.
    """
    
    def __init__(
        self,
        backend: CacheBackend = CacheBackend.MEMORY,
        redis_config: Optional[Dict[str, Any]] = None,
        memory_config: Optional[Dict[str, Any]] = None,
        enable_fallback: bool = True
    ):
        """
        Initialize cache manager.
        
        Args:
            backend: Primary cache backend
            redis_config: Redis configuration
            memory_config: Memory cache configuration
            enable_fallback: Enable fallback to memory cache if Redis fails
        """
        self.backend = backend
        self.enable_fallback = enable_fallback
        
        # Initialize memory cache
        memory_config = memory_config or {}
        self.memory_cache = MemoryCache(**memory_config)
        
        # Initialize Redis cache if available
        self.redis_cache = None
        if (backend in [CacheBackend.REDIS, CacheBackend.HYBRID] and 
            REDIS_AVAILABLE and redis_config):
            try:
                self.redis_cache = RedisCache(**redis_config)
                logger.info("Redis cache initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Redis cache: {e}")
                if not enable_fallback:
                    raise
        
        # Specialized caches
        self.prediction_cache = None
        self.feature_cache = None
        self._initialize_specialized_caches()
    
    def _initialize_specialized_caches(self):
        """Initialize specialized cache instances."""
        if self.backend == CacheBackend.REDIS and self.redis_cache:
            self.prediction_cache = RedisPredictionCache(
                host=self.redis_cache.host,
                port=self.redis_cache.port,
                db=self.redis_cache.db,
                password=self.redis_cache.password
            )
            self.feature_cache = RedisFeatureCache(
                host=self.redis_cache.host,
                port=self.redis_cache.port,
                db=self.redis_cache.db,
                password=self.redis_cache.password
            )
        else:
            self.prediction_cache = PredictionCache()
            self.feature_cache = FeatureCache()
    
    def _get_cache_instance(self, cache_type: str = 'general'):
        """Get appropriate cache instance based on backend and type."""
        if cache_type == 'prediction':
            return self.prediction_cache
        elif cache_type == 'feature':
            return self.feature_cache
        elif self.backend == CacheBackend.REDIS and self.redis_cache:
            return self.redis_cache
        else:
            return self.memory_cache
    
    def get(self, key: str, default: Any = None, cache_type: str = 'general') -> Any:
        """
        Get item from cache.
        
        Args:
            key: Cache key
            default: Default value if key not found
            cache_type: Type of cache ('general', 'prediction', 'feature')
            
        Returns:
            Cached value or default
        """
        cache = self._get_cache_instance(cache_type)
        
        try:
            return cache.get(key, default)
        except Exception as e:
            logger.error(f"Cache GET error: {e}")
            if self.enable_fallback and cache != self.memory_cache:
                logger.info("Falling back to memory cache")
                return self.memory_cache.get(key, default)
            return default
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        cache_type: str = 'general'
    ) -> bool:
        """
        Set item in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds
            cache_type: Type of cache ('general', 'prediction', 'feature')
            
        Returns:
            True if successful, False otherwise
        """
        cache = self._get_cache_instance(cache_type)
        
        try:
            if hasattr(cache, 'set'):
                return cache.set(key, value, ttl)
            else:
                cache.set(key, value, ttl)
                return True
        except Exception as e:
            logger.error(f"Cache SET error: {e}")
            if self.enable_fallback and cache != self.memory_cache:
                logger.info("Falling back to memory cache")
                self.memory_cache.set(key, value, ttl)
                return True
            return False
    
    def delete(self, key: str, cache_type: str = 'general') -> bool:
        """
        Delete item from cache.
        
        Args:
            key: Cache key
            cache_type: Type of cache ('general', 'prediction', 'feature')
            
        Returns:
            True if successful, False otherwise
        """
        cache = self._get_cache_instance(cache_type)
        
        try:
            return cache.delete(key)
        except Exception as e:
            logger.error(f"Cache DELETE error: {e}")
            if self.enable_fallback and cache != self.memory_cache:
                logger.info("Falling back to memory cache")
                return self.memory_cache.delete(key)
            return False
    
    def clear(self, cache_type: str = 'general', pattern: Optional[str] = None) -> int:
        """
        Clear cache.
        
        Args:
            cache_type: Type of cache to clear
            pattern: Optional pattern for Redis cache
            
        Returns:
            Number of items cleared
        """
        cache = self._get_cache_instance(cache_type)
        
        try:
            if hasattr(cache, 'clear'):
                if pattern and hasattr(cache, 'clear') and 'pattern' in cache.clear.__code__.co_varnames:
                    return cache.clear(pattern)
                else:
                    cache.clear()
                    return 1
            return 0
        except Exception as e:
            logger.error(f"Cache CLEAR error: {e}")
            return 0
    
    def exists(self, key: str, cache_type: str = 'general') -> bool:
        """
        Check if key exists in cache.
        
        Args:
            key: Cache key
            cache_type: Type of cache
            
        Returns:
            True if key exists, False otherwise
        """
        cache = self._get_cache_instance(cache_type)
        
        try:
            if hasattr(cache, 'exists'):
                return cache.exists(key)
            elif hasattr(cache, 'contains'):
                return cache.contains(key)
            else:
                return cache.get(key) is not None
        except Exception as e:
            logger.error(f"Cache EXISTS error: {e}")
            return False
    
    def get_stats(self, cache_type: str = 'general') -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Args:
            cache_type: Type of cache
            
        Returns:
            Dictionary with cache statistics
        """
        cache = self._get_cache_instance(cache_type)
        
        try:
            if hasattr(cache, 'get_stats'):
                return cache.get_stats()
            elif hasattr(cache, 'get_prediction_stats'):
                return cache.get_prediction_stats()
            return {}
        except Exception as e:
            logger.error(f"Cache STATS error: {e}")
            return {}
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all cache types.
        
        Returns:
            Dictionary with statistics for each cache type
        """
        stats = {}
        
        # General cache stats
        stats['general'] = self.get_stats('general')
        
        # Prediction cache stats
        if self.prediction_cache:
            stats['prediction'] = self.get_stats('prediction')
        
        # Feature cache stats
        if self.feature_cache:
            stats['feature'] = self.get_stats('feature')
        
        # Memory cache stats (if different from general)
        if self.backend != CacheBackend.MEMORY:
            stats['memory_fallback'] = self.memory_cache.get_stats()
        
        return stats
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on all cache backends.
        
        Returns:
            Dictionary with health status for each backend
        """
        health = {
            'memory_cache': {
                'status': 'healthy',
                'backend': 'memory',
                'stats': self.memory_cache.get_stats()
            }
        }
        
        if self.redis_cache:
            try:
                redis_healthy = self.redis_cache.ping()
                health['redis_cache'] = {
                    'status': 'healthy' if redis_healthy else 'unhealthy',
                    'backend': 'redis',
                    'stats': self.redis_cache.get_stats() if redis_healthy else {}
                }
            except Exception as e:
                health['redis_cache'] = {
                    'status': 'error',
                    'backend': 'redis',
                    'error': str(e)
                }
        
        return health
    
    def generate_cache_key(self, *args, **kwargs) -> str:
        """
        Generate a cache key from arguments.
        
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Generated cache key
        """
        # Create a string representation of all arguments
        key_parts = []
        
        for arg in args:
            if isinstance(arg, (str, int, float, bool)):
                key_parts.append(str(arg))
            else:
                key_parts.append(str(hash(str(arg))))
        
        for key, value in sorted(kwargs.items()):
            if isinstance(value, (str, int, float, bool)):
                key_parts.append(f"{key}:{value}")
            else:
                key_parts.append(f"{key}:{hash(str(value))}")
        
        # Create hash of the combined key
        combined_key = "|".join(key_parts)
        return hashlib.md5(combined_key.encode()).hexdigest()
    
    def cache_prediction(
        self,
        model_name: str,
        input_data: Dict[str, Any],
        prediction: Dict[str, Any],
        execution_time: float,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Cache a model prediction.
        
        Args:
            model_name: Name of the model
            input_data: Input data for prediction
            prediction: Prediction result
            execution_time: Time taken to generate prediction
            ttl: Time-to-live for cached prediction
            
        Returns:
            True if successful, False otherwise
        """
        input_hash = self.generate_cache_key(input_data)
        
        if hasattr(self.prediction_cache, 'cache_prediction'):
            return self.prediction_cache.cache_prediction(
                model_name, input_hash, prediction, execution_time, ttl
            )
        else:
            cache_key = f"{model_name}:{input_hash}"
            prediction_data = {
                'prediction': prediction,
                'execution_time': execution_time,
                'cached_at': time.time(),
                'model_name': model_name
            }
            return self.set(cache_key, prediction_data, ttl, 'prediction')
    
    def get_prediction(
        self,
        model_name: str,
        input_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached prediction.
        
        Args:
            model_name: Name of the model
            input_data: Input data for prediction
            
        Returns:
            Cached prediction data or None
        """
        input_hash = self.generate_cache_key(input_data)
        
        if hasattr(self.prediction_cache, 'get_prediction'):
            return self.prediction_cache.get_prediction(model_name, input_hash)
        else:
            cache_key = f"{model_name}:{input_hash}"
            return self.get(cache_key, cache_type='prediction')
    
    def cache_features(
        self,
        input_data: Dict[str, Any],
        processing_config: Dict[str, Any],
        features: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> bool:
        """
        Cache processed features.
        
        Args:
            input_data: Input data
            processing_config: Feature processing configuration
            features: Processed features
            ttl: Time-to-live for cached features
            
        Returns:
            True if successful, False otherwise
        """
        data_hash = self.generate_cache_key(input_data)
        
        if hasattr(self.feature_cache, 'cache_features'):
            return self.feature_cache.cache_features(
                data_hash, processing_config, features, ttl
            )
        else:
            config_hash = self.generate_cache_key(processing_config)
            cache_key = f"features:{data_hash}:{config_hash}"
            feature_data = {
                'features': features,
                'processing_config': processing_config,
                'cached_at': time.time(),
                'data_hash': data_hash
            }
            return self.set(cache_key, feature_data, ttl, 'feature')
    
    def get_features(
        self,
        input_data: Dict[str, Any],
        processing_config: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached features.
        
        Args:
            input_data: Input data
            processing_config: Feature processing configuration
            
        Returns:
            Cached feature data or None
        """
        data_hash = self.generate_cache_key(input_data)
        
        if hasattr(self.feature_cache, 'get_features'):
            return self.feature_cache.get_features(data_hash, processing_config)
        else:
            config_hash = self.generate_cache_key(processing_config)
            cache_key = f"features:{data_hash}:{config_hash}"
            return self.get(cache_key, cache_type='feature')
