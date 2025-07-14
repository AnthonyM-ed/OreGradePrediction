"""
ML Models Utilities Module

This module provides utility functions for geological analysis, spatial operations,
statistics, and logging configuration for the ore grade prediction system.
"""

from .geological_utils import (
    calculate_grade_tonnage,
    apply_cutoff_grade,
    calculate_metal_content,
    geological_statistics,
    validate_geological_data
)

from .spatial_analysis import (
    calculate_spatial_distances,
    find_nearest_neighbors,
    spatial_interpolation,
    calculate_spatial_autocorrelation,
    generate_spatial_grid
)

from .statistics import (
    calculate_advanced_metrics,
    bootstrap_confidence_intervals,
    statistical_tests,
    outlier_detection,
    distribution_analysis
)

from .logging_config import (
    setup_logging,
    get_logger,
    log_model_performance,
    log_prediction_results
)

__all__ = [
    # Geological utilities
    'calculate_grade_tonnage',
    'apply_cutoff_grade',
    'calculate_metal_content',
    'geological_statistics',
    'validate_geological_data',
    
    # Spatial analysis
    'calculate_spatial_distances',
    'find_nearest_neighbors',
    'spatial_interpolation',
    'calculate_spatial_autocorrelation',
    'generate_spatial_grid',
    
    # Statistics
    'calculate_advanced_metrics',
    'bootstrap_confidence_intervals',
    'statistical_tests',
    'outlier_detection',
    'distribution_analysis',
    
    # Logging
    'setup_logging',
    'get_logger',
    'log_model_performance',
    'log_prediction_results'
]
