"""
Real-Time Spatial Ore Grade Predictor

This module provides real-time prediction capabilities for web APIs and
interactive applications with caching and performance optimization.
"""

import os
import sys
import time
import json
import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import threading
from functools import lru_cache
import hashlib

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from ml_models.inference.predictor import SpatialOreGradePredictor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealTimeSpatialPredictor:
    """
    Real-time spatial ore grade predictor with caching and performance optimization
    """
    
    def __init__(self, model_path: str, element: str = "CU", 
                 cache_size: int = 1000, cache_ttl: int = 3600):
        """
        Initialize real-time predictor
        
        Args:
            model_path: Path to trained model
            element: Element to predict
            cache_size: Maximum number of cached predictions
            cache_ttl: Cache time-to-live in seconds
        """
        self.model_path = model_path
        self.element = element.upper()
        self.cache_size = cache_size
        self.cache_ttl = cache_ttl
        
        # Initialize predictor
        self.predictor = SpatialOreGradePredictor(model_path, element)
        
        # Initialize cache
        self.prediction_cache = {}
        self.cache_timestamps = {}
        self.cache_lock = threading.Lock()
        
        # Performance tracking
        self.performance_stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_prediction_time': 0.0,
            'average_prediction_time': 0.0,
            'start_time': time.time()
        }
        
        logger.info(f"Real-time predictor initialized for {self.element}")
        logger.info(f"Cache size: {self.cache_size}, TTL: {self.cache_ttl}s")
    
    def predict_single_point(self, latitude: float, longitude: float,
                           depth_from: float = 0.0, depth_to: float = 10.0,
                           use_cache: bool = True) -> Dict[str, Any]:
        """
        Predict ore grade at a single point with caching
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            depth_from: Depth from surface
            depth_to: Depth to
            use_cache: Whether to use caching
            
        Returns:
            Prediction result with performance metadata
        """
        try:
            start_time = time.time()
            
            # Generate cache key
            cache_key = self._generate_cache_key(latitude, longitude, depth_from, depth_to)
            
            # Check cache if enabled
            if use_cache:
                cached_result = self._get_from_cache(cache_key)
                if cached_result is not None:
                    # Update performance stats
                    self._update_performance_stats(start_time, cache_hit=True)
                    
                    # Add real-time metadata
                    cached_result['real_time_metadata'] = {
                        'cache_hit': True,
                        'response_time_ms': (time.time() - start_time) * 1000,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    return cached_result
            
            # Make prediction
            prediction = self.predictor.predict_at_point(
                latitude, longitude, depth_from, depth_to
            )
            
            # Store in cache if enabled
            if use_cache:
                self._store_in_cache(cache_key, prediction)
            
            # Update performance stats
            self._update_performance_stats(start_time, cache_hit=False)
            
            # Add real-time metadata
            prediction['real_time_metadata'] = {
                'cache_hit': False,
                'response_time_ms': (time.time() - start_time) * 1000,
                'timestamp': datetime.now().isoformat()
            }
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error in real-time prediction: {str(e)}")
            raise
    
    def predict_multiple_points_streaming(self, coordinates: List[Dict[str, float]],
                                        use_cache: bool = True) -> List[Dict[str, Any]]:
        """
        Predict multiple points with streaming-like processing
        
        Args:
            coordinates: List of coordinate dictionaries
            use_cache: Whether to use caching
            
        Returns:
            List of prediction results
        """
        try:
            results = []
            
            for coord in coordinates:
                result = self.predict_single_point(
                    coord['latitude'],
                    coord['longitude'],
                    coord.get('depth_from', 0.0),
                    coord.get('depth_to', 10.0),
                    use_cache=use_cache
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in streaming prediction: {str(e)}")
            raise
    
    def predict_with_validation(self, latitude: float, longitude: float,
                              depth_from: float = 0.0, depth_to: float = 10.0,
                              validation_rules: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Predict with input validation and error handling
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            depth_from: Depth from surface
            depth_to: Depth to
            validation_rules: Custom validation rules
            
        Returns:
            Prediction result with validation metadata
        """
        try:
            # Default validation rules
            default_rules = {
                'latitude_range': (-90, 90),
                'longitude_range': (-180, 180),
                'depth_range': (0, 1000),
                'depth_order': True  # depth_from <= depth_to
            }
            
            # Merge with custom rules
            if validation_rules:
                default_rules.update(validation_rules)
            
            # Validate inputs
            validation_result = self._validate_inputs(
                latitude, longitude, depth_from, depth_to, default_rules
            )
            
            if not validation_result['valid']:
                return {
                    'error': 'Validation failed',
                    'validation_errors': validation_result['errors'],
                    'timestamp': datetime.now().isoformat()
                }
            
            # Make prediction
            prediction = self.predict_single_point(
                latitude, longitude, depth_from, depth_to
            )
            
            # Add validation metadata
            prediction['validation_metadata'] = {
                'validated': True,
                'validation_rules': default_rules,
                'validation_passed': True
            }
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error in validated prediction: {str(e)}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_prediction_confidence(self, latitude: float, longitude: float,
                                depth_from: float = 0.0, depth_to: float = 10.0) -> Dict[str, Any]:
        """
        Get prediction confidence metrics
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            depth_from: Depth from surface
            depth_to: Depth to
            
        Returns:
            Confidence metrics
        """
        try:
            # Get base prediction
            prediction = self.predict_single_point(
                latitude, longitude, depth_from, depth_to, use_cache=False
            )
            
            # Calculate additional confidence metrics
            confidence_metrics = {
                'prediction_value': prediction['predicted_grade_ppm'],
                'confidence_interval': prediction['confidence_interval'],
                'model_confidence': self._calculate_model_confidence(prediction),
                'spatial_confidence': self._calculate_spatial_confidence(latitude, longitude),
                'overall_confidence': 'high'  # Placeholder
            }
            
            return confidence_metrics
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {str(e)}")
            return {'error': str(e)}
    
    def _generate_cache_key(self, latitude: float, longitude: float,
                          depth_from: float, depth_to: float) -> str:
        """
        Generate cache key for prediction
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            depth_from: Depth from surface
            depth_to: Depth to
            
        Returns:
            Cache key string
        """
        # Create hash of coordinates with precision
        coord_str = f"{latitude:.6f},{longitude:.6f},{depth_from:.2f},{depth_to:.2f},{self.element}"
        return hashlib.md5(coord_str.encode()).hexdigest()
    
    def _get_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Get prediction from cache
        
        Args:
            cache_key: Cache key
            
        Returns:
            Cached prediction or None
        """
        with self.cache_lock:
            if cache_key in self.prediction_cache:
                # Check if cache entry is still valid
                if self._is_cache_valid(cache_key):
                    return self.prediction_cache[cache_key].copy()
                else:
                    # Remove expired entry
                    del self.prediction_cache[cache_key]
                    del self.cache_timestamps[cache_key]
            
            return None
    
    def _store_in_cache(self, cache_key: str, prediction: Dict[str, Any]):
        """
        Store prediction in cache
        
        Args:
            cache_key: Cache key
            prediction: Prediction result
        """
        with self.cache_lock:
            # Clean cache if at capacity
            if len(self.prediction_cache) >= self.cache_size:
                self._clean_cache()
            
            # Store prediction
            self.prediction_cache[cache_key] = prediction.copy()
            self.cache_timestamps[cache_key] = time.time()
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """
        Check if cache entry is still valid
        
        Args:
            cache_key: Cache key
            
        Returns:
            True if valid, False otherwise
        """
        if cache_key not in self.cache_timestamps:
            return False
        
        age = time.time() - self.cache_timestamps[cache_key]
        return age < self.cache_ttl
    
    def _clean_cache(self):
        """Clean expired cache entries"""
        current_time = time.time()
        expired_keys = []
        
        for key, timestamp in self.cache_timestamps.items():
            if current_time - timestamp > self.cache_ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.prediction_cache[key]
            del self.cache_timestamps[key]
        
        # If still at capacity, remove oldest entries
        if len(self.prediction_cache) >= self.cache_size:
            # Sort by timestamp and remove oldest
            sorted_keys = sorted(self.cache_timestamps.keys(), 
                               key=lambda k: self.cache_timestamps[k])
            
            keys_to_remove = sorted_keys[:len(sorted_keys) - self.cache_size + 1]
            for key in keys_to_remove:
                del self.prediction_cache[key]
                del self.cache_timestamps[key]
    
    def _validate_inputs(self, latitude: float, longitude: float,
                        depth_from: float, depth_to: float,
                        rules: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate input parameters
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            depth_from: Depth from surface
            depth_to: Depth to
            rules: Validation rules
            
        Returns:
            Validation result
        """
        errors = []
        
        # Validate latitude
        lat_range = rules.get('latitude_range', (-90, 90))
        if not (lat_range[0] <= latitude <= lat_range[1]):
            errors.append(f"Latitude {latitude} outside valid range {lat_range}")
        
        # Validate longitude
        long_range = rules.get('longitude_range', (-180, 180))
        if not (long_range[0] <= longitude <= long_range[1]):
            errors.append(f"Longitude {longitude} outside valid range {long_range}")
        
        # Validate depth
        depth_range = rules.get('depth_range', (0, 1000))
        if not (depth_range[0] <= depth_from <= depth_range[1]):
            errors.append(f"Depth from {depth_from} outside valid range {depth_range}")
        
        if not (depth_range[0] <= depth_to <= depth_range[1]):
            errors.append(f"Depth to {depth_to} outside valid range {depth_range}")
        
        # Validate depth order
        if rules.get('depth_order', True) and depth_from > depth_to:
            errors.append(f"Depth from {depth_from} must be <= depth to {depth_to}")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors
        }
    
    def _calculate_model_confidence(self, prediction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate model-based confidence metrics
        
        Args:
            prediction: Prediction result
            
        Returns:
            Model confidence metrics
        """
        # This is a placeholder - in practice, you'd implement more sophisticated confidence calculation
        confidence_interval = prediction.get('confidence_interval', {})
        
        # Calculate confidence based on interval width
        interval_width = confidence_interval.get('upper_bound', 0) - confidence_interval.get('lower_bound', 0)
        
        if interval_width < 100:
            confidence_level = 'high'
        elif interval_width < 300:
            confidence_level = 'medium'
        else:
            confidence_level = 'low'
        
        return {
            'confidence_level': confidence_level,
            'interval_width': interval_width,
            'method': 'interval_based'
        }
    
    def _calculate_spatial_confidence(self, latitude: float, longitude: float) -> Dict[str, Any]:
        """
        Calculate spatial confidence based on location
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            
        Returns:
            Spatial confidence metrics
        """
        # This is a placeholder - in practice, you'd check distance to training data
        # For now, assume higher confidence for certain regions
        
        # Example: Higher confidence for certain coordinate ranges
        if -25 <= latitude <= -20 and -50 <= longitude <= -45:
            confidence_level = 'high'
        elif -30 <= latitude <= -15 and -55 <= longitude <= -40:
            confidence_level = 'medium'
        else:
            confidence_level = 'low'
        
        return {
            'confidence_level': confidence_level,
            'method': 'spatial_proximity'
        }
    
    def _update_performance_stats(self, start_time: float, cache_hit: bool):
        """
        Update performance statistics
        
        Args:
            start_time: Request start time
            cache_hit: Whether this was a cache hit
        """
        prediction_time = time.time() - start_time
        
        self.performance_stats['total_requests'] += 1
        self.performance_stats['total_prediction_time'] += prediction_time
        
        if cache_hit:
            self.performance_stats['cache_hits'] += 1
        else:
            self.performance_stats['cache_misses'] += 1
        
        # Update average
        self.performance_stats['average_prediction_time'] = (
            self.performance_stats['total_prediction_time'] / 
            self.performance_stats['total_requests']
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics
        
        Returns:
            Performance statistics
        """
        uptime = time.time() - self.performance_stats['start_time']
        cache_hit_rate = (
            self.performance_stats['cache_hits'] / 
            max(self.performance_stats['total_requests'], 1)
        ) * 100
        
        return {
            'uptime_seconds': uptime,
            'total_requests': self.performance_stats['total_requests'],
            'cache_hit_rate_percent': cache_hit_rate,
            'average_response_time_ms': self.performance_stats['average_prediction_time'] * 1000,
            'cache_size': len(self.prediction_cache),
            'cache_capacity': self.cache_size
        }
    
    def clear_cache(self):
        """Clear all cached predictions"""
        with self.cache_lock:
            self.prediction_cache.clear()
            self.cache_timestamps.clear()
        
        logger.info("Cache cleared")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get cache information
        
        Returns:
            Cache information
        """
        with self.cache_lock:
            return {
                'current_size': len(self.prediction_cache),
                'max_size': self.cache_size,
                'ttl_seconds': self.cache_ttl,
                'hit_rate_percent': (
                    self.performance_stats['cache_hits'] / 
                    max(self.performance_stats['total_requests'], 1)
                ) * 100
            }


# Example usage
if __name__ == "__main__":
    print("Real-Time Spatial Ore Grade Predictor")
    print("=" * 50)
    print("Usage examples:")
    print()
    print("1. Real-time prediction:")
    print("   rt_predictor = RealTimeSpatialPredictor('model.joblib', 'CU')")
    print("   result = rt_predictor.predict_single_point(-23.5505, -46.6333)")
    print()
    print("2. Validated prediction:")
    print("   result = rt_predictor.predict_with_validation(-23.5505, -46.6333)")
    print()
    print("3. Performance monitoring:")
    print("   stats = rt_predictor.get_performance_stats()")
    print("   cache_info = rt_predictor.get_cache_info()")
    print()
    print("Features:")
    print("- Real-time prediction with caching")
    print("- Input validation and error handling")
    print("- Performance monitoring and statistics")
    print("- Confidence metrics calculation")
    print("- Thread-safe operation")
    print("- API-ready interface")
