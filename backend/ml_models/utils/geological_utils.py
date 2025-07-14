"""
Geological utilities for ore grade prediction and analysis.

This module provides functions for geological calculations, grade tonnage analysis,
cutoff grade applications, and geological data validation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


def calculate_grade_tonnage(
    grades: np.ndarray,
    volumes: np.ndarray,
    density: float = 2.7
) -> Dict[str, float]:
    """
    Calculate grade-tonnage relationships for ore deposits.
    
    Args:
        grades: Array of grade values (ppm)
        volumes: Array of volume values (m³)
        density: Rock density (tonnes/m³)
        
    Returns:
        Dictionary with grade-tonnage statistics
    """
    if len(grades) != len(volumes):
        raise ValueError("Grades and volumes arrays must have the same length")
    
    # Convert volumes to tonnage
    tonnage = volumes * density
    
    # Calculate metal content
    metal_content = (grades * tonnage) / 1e6  # Convert ppm to tonnes
    
    # Calculate cumulative statistics
    sorted_indices = np.argsort(grades)[::-1]  # Sort descending
    sorted_grades = grades[sorted_indices]
    sorted_tonnage = tonnage[sorted_indices]
    sorted_metal = metal_content[sorted_indices]
    
    cumulative_tonnage = np.cumsum(sorted_tonnage)
    cumulative_metal = np.cumsum(sorted_metal)
    cumulative_grade = cumulative_metal / cumulative_tonnage * 1e6  # Back to ppm
    
    return {
        'total_tonnage': float(np.sum(tonnage)),
        'total_metal_content': float(np.sum(metal_content)),
        'average_grade': float(np.average(grades, weights=tonnage)),
        'grade_variance': float(np.var(grades)),
        'cumulative_tonnage': cumulative_tonnage.tolist(),
        'cumulative_grade': cumulative_grade.tolist(),
        'cumulative_metal': cumulative_metal.tolist()
    }


def apply_cutoff_grade(
    data: pd.DataFrame,
    element: str,
    cutoff_grade: float,
    grade_column: str = 'standardized_grade_ppm'
) -> pd.DataFrame:
    """
    Apply cutoff grade to filter economic ore zones.
    
    Args:
        data: DataFrame with geological data
        element: Element to apply cutoff to
        cutoff_grade: Minimum grade threshold (ppm)
        grade_column: Column name with grade values
        
    Returns:
        Filtered DataFrame with ore above cutoff grade
    """
    element_data = data[data['element'] == element].copy()
    
    if grade_column not in element_data.columns:
        raise ValueError(f"Column '{grade_column}' not found in data")
    
    # Apply cutoff grade filter
    ore_data = element_data[element_data[grade_column] >= cutoff_grade]
    
    logger.info(f"Applied {cutoff_grade} ppm cutoff for {element}: "
                f"{len(ore_data)}/{len(element_data)} samples above cutoff "
                f"({len(ore_data)/len(element_data)*100:.1f}%)")
    
    return ore_data


def calculate_metal_content(
    grade: float,
    tonnage: float,
    recovery: float = 0.85
) -> float:
    """
    Calculate recoverable metal content.
    
    Args:
        grade: Grade in ppm
        tonnage: Tonnage in tonnes
        recovery: Recovery factor (0-1)
        
    Returns:
        Recoverable metal content in tonnes
    """
    metal_content = (grade * tonnage * recovery) / 1e6
    return metal_content


def geological_statistics(
    data: pd.DataFrame,
    element: str,
    grade_column: str = 'standardized_grade_ppm'
) -> Dict[str, float]:
    """
    Calculate geological statistics for an element.
    
    Args:
        data: DataFrame with geological data
        element: Element to analyze
        grade_column: Column name with grade values
        
    Returns:
        Dictionary with geological statistics
    """
    element_data = data[data['element'] == element][grade_column].dropna()
    
    if len(element_data) == 0:
        return {}
    
    # Basic statistics
    stats = {
        'count': len(element_data),
        'mean': float(element_data.mean()),
        'median': float(element_data.median()),
        'std': float(element_data.std()),
        'min': float(element_data.min()),
        'max': float(element_data.max()),
        'q25': float(element_data.quantile(0.25)),
        'q75': float(element_data.quantile(0.75)),
        'skewness': float(element_data.skew()),
        'kurtosis': float(element_data.kurtosis())
    }
    
    # Coefficient of variation
    stats['cv'] = stats['std'] / stats['mean'] if stats['mean'] > 0 else 0
    
    # Grade distribution percentiles
    percentiles = [90, 95, 99]
    for p in percentiles:
        stats[f'p{p}'] = float(element_data.quantile(p/100))
    
    # Log-normal statistics if applicable
    if stats['min'] > 0:
        log_data = np.log(element_data)
        stats['log_mean'] = float(log_data.mean())
        stats['log_std'] = float(log_data.std())
        stats['geometric_mean'] = float(np.exp(log_data.mean()))
    
    return stats


def validate_geological_data(
    data: pd.DataFrame,
    required_columns: List[str] = None
) -> Dict[str, Union[bool, List[str]]]:
    """
    Validate geological data quality and completeness.
    
    Args:
        data: DataFrame to validate
        required_columns: List of required column names
        
    Returns:
        Dictionary with validation results
    """
    if required_columns is None:
        required_columns = [
            'latitude', 'longitude', 'depth_from', 'depth_to',
            'element', 'standardized_grade_ppm'
        ]
    
    validation_results = {
        'is_valid': True,
        'missing_columns': [],
        'empty_columns': [],
        'negative_grades': 0,
        'negative_depths': 0,
        'invalid_coordinates': 0,
        'duplicate_samples': 0,
        'total_samples': len(data)
    }
    
    # Check for missing columns
    missing_cols = [col for col in required_columns if col not in data.columns]
    validation_results['missing_columns'] = missing_cols
    
    if missing_cols:
        validation_results['is_valid'] = False
        return validation_results
    
    # Check for empty columns
    empty_cols = [col for col in required_columns if data[col].isna().all()]
    validation_results['empty_columns'] = empty_cols
    
    # Check for negative grades
    if 'standardized_grade_ppm' in data.columns:
        negative_grades = (data['standardized_grade_ppm'] < 0).sum()
        validation_results['negative_grades'] = int(negative_grades)
    
    # Check for negative depths
    depth_cols = ['depth_from', 'depth_to']
    for col in depth_cols:
        if col in data.columns:
            negative_depths = (data[col] < 0).sum()
            validation_results['negative_depths'] += int(negative_depths)
    
    # Check for invalid coordinates
    if 'latitude' in data.columns and 'longitude' in data.columns:
        invalid_lat = ((data['latitude'] < -90) | (data['latitude'] > 90)).sum()
        invalid_lon = ((data['longitude'] < -180) | (data['longitude'] > 180)).sum()
        validation_results['invalid_coordinates'] = int(invalid_lat + invalid_lon)
    
    # Check for duplicate samples
    if 'sample_id' in data.columns:
        duplicates = data['sample_id'].duplicated().sum()
        validation_results['duplicate_samples'] = int(duplicates)
    
    # Set overall validity
    validation_results['is_valid'] = (
        len(missing_cols) == 0 and
        len(empty_cols) == 0 and
        validation_results['negative_grades'] == 0 and
        validation_results['negative_depths'] == 0 and
        validation_results['invalid_coordinates'] == 0
    )
    
    return validation_results


def calculate_grade_continuity(
    data: pd.DataFrame,
    element: str,
    distance_threshold: float = 100.0
) -> Dict[str, float]:
    """
    Calculate grade continuity statistics for geological interpretation.
    
    Args:
        data: DataFrame with spatial and grade data
        element: Element to analyze
        distance_threshold: Maximum distance for continuity analysis (m)
        
    Returns:
        Dictionary with continuity statistics
    """
    element_data = data[data['element'] == element].copy()
    
    if len(element_data) < 2:
        return {}
    
    # Calculate pairwise distances and grade differences
    from scipy.spatial.distance import pdist, squareform
    
    coords = element_data[['latitude', 'longitude']].values
    grades = element_data['standardized_grade_ppm'].values
    
    # Calculate distance matrix
    distances = squareform(pdist(coords))
    
    # Calculate grade difference matrix
    grade_diffs = np.abs(grades[:, np.newaxis] - grades)
    
    # Filter by distance threshold
    valid_pairs = distances <= distance_threshold
    
    if not np.any(valid_pairs):
        return {}
    
    # Calculate continuity metrics
    continuity_stats = {
        'mean_grade_difference': float(np.mean(grade_diffs[valid_pairs])),
        'max_grade_difference': float(np.max(grade_diffs[valid_pairs])),
        'continuity_coefficient': float(1 - (np.mean(grade_diffs[valid_pairs]) / np.std(grades))),
        'pairs_within_threshold': int(np.sum(valid_pairs)),
        'total_possible_pairs': int(len(element_data) * (len(element_data) - 1) / 2)
    }
    
    return continuity_stats
