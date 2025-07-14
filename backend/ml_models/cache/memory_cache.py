"""
In-memory cache implementation for ML models.

This module provides a thread-safe in-memory cache for storing
model predictions, feature engineering results, and other temporary data.
"""

import threading
import time
from typing import Any, Dict, Optional, Tuple
from collections import OrderedDict
import pickle
import hashlib
import logging

logger = logging.getLogger(__name__)


class MemoryCache:
    """
    Thread-safe in-memory cache with TTL and LRU eviction.
    """
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        """
        Initialize memory cache.
        
        Args:
            max_size: Maximum number of items to store
            default_ttl: Default time-to-live in seconds
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache = OrderedDict()
        self._lock = threading.RLock()
        self._stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'evictions': 0,
            'size': 0
        }
    
    def _generate_key(self, key: str) -> str:
        """Generate a hash key for the cache."""
        if isinstance(key, str):
            return hashlib.md5(key.encode()).hexdigest()
        return hashlib.md5(str(key).encode()).hexdigest()
    
    def _is_expired(self, timestamp: float, ttl: int) -> bool:
        """Check if an item has expired."""
        return time.time() - timestamp > ttl
    
    def _evict_expired(self) -> None:
        """Remove expired items from cache."""
        current_time = time.time()
        expired_keys = []
        
        for key, (value, timestamp, ttl) in self._cache.items():
            if self._is_expired(timestamp, ttl):
                expired_keys.append(key)
        
        for key in expired_keys:
            del self._cache[key]
            self._stats['evictions'] += 1
    
    def _evict_lru(self) -> None:
        """Remove least recently used items if cache is full."""
        while len(self._cache) >= self.max_size:
            self._cache.popitem(last=False)
            self._stats['evictions'] += 1
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get item from cache.
        
        Args:
            key: Cache key
            default: Default value if key not found
            
        Returns:
            Cached value or default
        """
        with self._lock:
            hashed_key = self._generate_key(key)
            
            # Clean expired items
            self._evict_expired()
            
            if hashed_key in self._cache:
                value, timestamp, ttl = self._cache[hashed_key]
                
                # Check if expired
                if self._is_expired(timestamp, ttl):
                    del self._cache[hashed_key]
                    self._stats['misses'] += 1
                    return default
                
                # Move to end (most recently used)
                self._cache.move_to_end(hashed_key)
                self._stats['hits'] += 1
                return value
            
            self._stats['misses'] += 1
            return default
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set item in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds
        """
        with self._lock:
            hashed_key = self._generate_key(key)
            ttl = ttl or self.default_ttl
            timestamp = time.time()
            
            # Remove expired items
            self._evict_expired()
            
            # Evict LRU items if necessary
            if hashed_key not in self._cache:
                self._evict_lru()
            
            self._cache[hashed_key] = (value, timestamp, ttl)
            self._cache.move_to_end(hashed_key)
            self._stats['sets'] += 1
            self._stats['size'] = len(self._cache)
    
    def delete(self, key: str) -> bool:
        """
        Delete item from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if item was deleted, False if not found
        """
        with self._lock:
            hashed_key = self._generate_key(key)
            if hashed_key in self._cache:
                del self._cache[hashed_key]
                self._stats['size'] = len(self._cache)
                return True
            return False
    
    def clear(self) -> None:
        """Clear all items from cache."""
        with self._lock:
            self._cache.clear()
            self._stats['size'] = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        with self._lock:
            self._stats['size'] = len(self._cache)
            hit_rate = self._stats['hits'] / (self._stats['hits'] + self._stats['misses']) if (self._stats['hits'] + self._stats['misses']) > 0 else 0
            
            return {
                **self._stats,
                'hit_rate': hit_rate,
                'memory_usage_mb': self._estimate_memory_usage()
            }
    
    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage in MB."""
        try:
            total_size = 0
            for key, (value, timestamp, ttl) in self._cache.items():
                total_size += len(pickle.dumps(value))
            return total_size / (1024 * 1024)
        except Exception:
            return 0.0
    
    def get_keys(self) -> list:
        """Get all cache keys."""
        with self._lock:
            return list(self._cache.keys())
    
    def contains(self, key: str) -> bool:
        """Check if key exists in cache."""
        with self._lock:
            hashed_key = self._generate_key(key)
            return hashed_key in self._cache


class PredictionCache(MemoryCache):
    """
    Specialized cache for model predictions.
    """
    
    def __init__(self, max_size: int = 500, default_ttl: int = 1800):
        """
        Initialize prediction cache.
        
        Args:
            max_size: Maximum number of predictions to cache
            default_ttl: Default TTL for predictions (30 minutes)
        """
        super().__init__(max_size, default_ttl)
        self.prediction_stats = {
            'cached_predictions': 0,
            'cache_savings_seconds': 0.0
        }
    
    def cache_prediction(
        self,
        model_name: str,
        input_hash: str,
        prediction: Dict[str, Any],
        execution_time: float,
        ttl: Optional[int] = None
    ) -> None:
        """
        Cache a model prediction.
        
        Args:
            model_name: Name of the model
            input_hash: Hash of input data
            prediction: Prediction result
            execution_time: Time taken to generate prediction
            ttl: Time-to-live for cached prediction
        """
        cache_key = f"{model_name}:{input_hash}"
        
        prediction_data = {
            'prediction': prediction,
            'execution_time': execution_time,
            'cached_at': time.time(),
            'model_name': model_name
        }
        
        self.set(cache_key, prediction_data, ttl)
        self.prediction_stats['cached_predictions'] += 1
        
        logger.debug(f"Cached prediction for {model_name} with key {input_hash}")
    
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
            # Update cache savings
            self.prediction_stats['cache_savings_seconds'] += prediction_data['execution_time']
            logger.debug(f"Retrieved cached prediction for {model_name}")
            return prediction_data
        
        return None
    
    def get_prediction_stats(self) -> Dict[str, Any]:
        """Get prediction-specific cache statistics."""
        base_stats = self.get_stats()
        return {
            **base_stats,
            **self.prediction_stats,
            'avg_execution_time_saved': (
                self.prediction_stats['cache_savings_seconds'] / 
                max(self.prediction_stats['cached_predictions'], 1)
            )
        }


class FeatureCache(MemoryCache):
    """
    Specialized cache for feature engineering results.
    """
    
    def __init__(self, max_size: int = 200, default_ttl: int = 7200):
        """
        Initialize feature cache.
        
        Args:
            max_size: Maximum number of feature sets to cache
            default_ttl: Default TTL for features (2 hours)
        """
        super().__init__(max_size, default_ttl)
    
    def cache_features(
        self,
        data_hash: str,
        processing_config: Dict[str, Any],
        features: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> None:
        """
        Cache processed features.
        
        Args:
            data_hash: Hash of input data
            processing_config: Feature processing configuration
            features: Processed features
            ttl: Time-to-live for cached features
        """
        config_hash = hashlib.md5(str(processing_config).encode()).hexdigest()
        cache_key = f"features:{data_hash}:{config_hash}"
        
        feature_data = {
            'features': features,
            'processing_config': processing_config,
            'cached_at': time.time(),
            'data_hash': data_hash
        }
        
        self.set(cache_key, feature_data, ttl)
        logger.debug(f"Cached features for data hash {data_hash}")
    
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
