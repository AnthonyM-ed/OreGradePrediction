"""
Inference Module for Spatial Ore Grade Prediction

This module provides comprehensive inference capabilities for predicting ore grades
at specific spatial coordinates (latitude, longitude) with various processing modes:

- Single point prediction
- Batch processing for large datasets
- Real-time prediction with caching
- Spatial grid generation
- Confidence interval calculation
"""

from .predictor import SpatialOreGradePredictor
from .batch_predictor import BatchSpatialPredictor
from .real_time_predictor import RealTimeSpatialPredictor

__all__ = [
    'SpatialOreGradePredictor',
    'BatchSpatialPredictor',
    'RealTimeSpatialPredictor'
]

__version__ = '1.0.0'
