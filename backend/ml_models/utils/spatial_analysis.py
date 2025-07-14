"""
Spatial analysis utilities for ore grade prediction.

This module provides functions for spatial operations, interpolation,
neighborhood analysis, and spatial autocorrelation calculations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from scipy.spatial import distance
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors
import logging

logger = logging.getLogger(__name__)


def calculate_spatial_distances(
    coords1: np.ndarray,
    coords2: np.ndarray = None,
    metric: str = 'euclidean'
) -> np.ndarray:
    """
    Calculate spatial distances between coordinate points.
    
    Args:
        coords1: First set of coordinates (lat, lon)
        coords2: Second set of coordinates (if None, uses coords1)
        metric: Distance metric ('euclidean', 'manhattan', 'haversine')
        
    Returns:
        Distance matrix
    """
    if coords2 is None:
        coords2 = coords1
    
    if metric == 'haversine':
        return _haversine_distance(coords1, coords2)
    elif metric == 'euclidean':
        return distance.cdist(coords1, coords2, metric='euclidean')
    elif metric == 'manhattan':
        return distance.cdist(coords1, coords2, metric='manhattan')
    else:
        raise ValueError(f"Unsupported distance metric: {metric}")


def _haversine_distance(coords1: np.ndarray, coords2: np.ndarray) -> np.ndarray:
    """
    Calculate haversine distance between geographic coordinates.
    
    Args:
        coords1: First set of coordinates (lat, lon) in degrees
        coords2: Second set of coordinates (lat, lon) in degrees
        
    Returns:
        Distance matrix in kilometers
    """
    R = 6371  # Earth's radius in kilometers
    
    # Convert to radians
    lat1, lon1 = np.radians(coords1[:, 0]), np.radians(coords1[:, 1])
    lat2, lon2 = np.radians(coords2[:, 0]), np.radians(coords2[:, 1])
    
    # Haversine formula
    dlat = lat2[:, np.newaxis] - lat1
    dlon = lon2[:, np.newaxis] - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2[:, np.newaxis]) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c


def find_nearest_neighbors(
    target_coords: np.ndarray,
    reference_coords: np.ndarray,
    n_neighbors: int = 5,
    max_distance: float = None
) -> Dict[str, np.ndarray]:
    """
    Find nearest neighbors for spatial interpolation.
    
    Args:
        target_coords: Target coordinates for prediction
        reference_coords: Reference coordinates with known values
        n_neighbors: Number of nearest neighbors to find
        max_distance: Maximum search distance (optional)
        
    Returns:
        Dictionary with neighbor indices and distances
    """
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
    nbrs.fit(reference_coords)
    
    distances, indices = nbrs.kneighbors(target_coords)
    
    # Apply distance filter if specified
    if max_distance is not None:
        mask = distances <= max_distance
        distances = np.where(mask, distances, np.nan)
        indices = np.where(mask, indices, -1)
    
    return {
        'indices': indices,
        'distances': distances,
        'n_neighbors_found': np.sum(~np.isnan(distances), axis=1)
    }


def spatial_interpolation(
    target_coords: np.ndarray,
    reference_coords: np.ndarray,
    reference_values: np.ndarray,
    method: str = 'inverse_distance',
    n_neighbors: int = 5,
    power: float = 2.0,
    max_distance: float = None
) -> Dict[str, np.ndarray]:
    """
    Perform spatial interpolation using various methods.
    
    Args:
        target_coords: Target coordinates for prediction
        reference_coords: Reference coordinates with known values
        reference_values: Known values at reference coordinates
        method: Interpolation method ('inverse_distance', 'kriging', 'rbf')
        n_neighbors: Number of nearest neighbors
        power: Power parameter for inverse distance weighting
        max_distance: Maximum search distance
        
    Returns:
        Dictionary with interpolated values and confidence intervals
    """
    neighbors = find_nearest_neighbors(
        target_coords, reference_coords, n_neighbors, max_distance
    )
    
    n_targets = len(target_coords)
    interpolated_values = np.full(n_targets, np.nan)
    confidence_intervals = np.full((n_targets, 2), np.nan)
    
    for i in range(n_targets):
        neighbor_indices = neighbors['indices'][i]
        neighbor_distances = neighbors['distances'][i]
        
        # Skip if no valid neighbors
        valid_mask = (neighbor_indices >= 0) & (~np.isnan(neighbor_distances))
        if not np.any(valid_mask):
            continue
        
        valid_indices = neighbor_indices[valid_mask]
        valid_distances = neighbor_distances[valid_mask]
        neighbor_values = reference_values[valid_indices]
        
        if method == 'inverse_distance':
            # Inverse distance weighting
            if len(valid_distances) == 1:
                interpolated_values[i] = neighbor_values[0]
                confidence_intervals[i] = [neighbor_values[0], neighbor_values[0]]
            else:
                # Avoid division by zero
                weights = 1.0 / np.maximum(valid_distances ** power, 1e-10)
                weights /= np.sum(weights)
                
                interpolated_values[i] = np.sum(weights * neighbor_values)
                
                # Simple confidence interval based on weighted variance
                variance = np.sum(weights * (neighbor_values - interpolated_values[i])**2)
                std_error = np.sqrt(variance)
                confidence_intervals[i] = [
                    interpolated_values[i] - 1.96 * std_error,
                    interpolated_values[i] + 1.96 * std_error
                ]
        
        elif method == 'kriging':
            # Simple kriging implementation
            interpolated_values[i] = np.mean(neighbor_values)
            std_error = np.std(neighbor_values) / np.sqrt(len(neighbor_values))
            confidence_intervals[i] = [
                interpolated_values[i] - 1.96 * std_error,
                interpolated_values[i] + 1.96 * std_error
            ]
    
    return {
        'values': interpolated_values,
        'confidence_intervals': confidence_intervals,
        'n_neighbors_used': neighbors['n_neighbors_found']
    }


def calculate_spatial_autocorrelation(
    coords: np.ndarray,
    values: np.ndarray,
    distance_bands: List[float] = None
) -> Dict[str, float]:
    """
    Calculate spatial autocorrelation (Moran's I) for different distance bands.
    
    Args:
        coords: Coordinate array (lat, lon)
        values: Values at each coordinate
        distance_bands: Distance bands for analysis
        
    Returns:
        Dictionary with autocorrelation statistics
    """
    if distance_bands is None:
        distance_bands = [50, 100, 200, 500, 1000]  # meters
    
    # Calculate distance matrix
    distances = calculate_spatial_distances(coords)
    
    # Calculate Moran's I for each distance band
    autocorr_results = {}
    
    for band in distance_bands:
        # Create spatial weights matrix
        weights = (distances <= band) & (distances > 0)
        
        if np.sum(weights) == 0:
            continue
        
        # Normalize weights
        row_sums = np.sum(weights, axis=1)
        weights = weights / np.maximum(row_sums[:, np.newaxis], 1)
        
        # Calculate Moran's I
        n = len(values)
        mean_val = np.mean(values)
        
        numerator = 0
        denominator = 0
        
        for i in range(n):
            for j in range(n):
                if weights[i, j] > 0:
                    numerator += weights[i, j] * (values[i] - mean_val) * (values[j] - mean_val)
            denominator += (values[i] - mean_val)**2
        
        if denominator > 0:
            morans_i = (n / np.sum(weights)) * (numerator / denominator)
            autocorr_results[f'morans_i_{band}m'] = morans_i
    
    return autocorr_results


def generate_spatial_grid(
    bounds: Tuple[float, float, float, float],
    resolution: float = 100.0
) -> np.ndarray:
    """
    Generate a regular spatial grid for prediction.
    
    Args:
        bounds: Bounding box (min_lat, min_lon, max_lat, max_lon)
        resolution: Grid resolution in meters
        
    Returns:
        Array of grid coordinates
    """
    min_lat, min_lon, max_lat, max_lon = bounds
    
    # Calculate grid dimensions
    lat_range = max_lat - min_lat
    lon_range = max_lon - min_lon
    
    # Convert resolution to degrees (approximate)
    lat_resolution = resolution / 111000  # 1 degree ≈ 111 km
    lon_resolution = resolution / (111000 * np.cos(np.radians((min_lat + max_lat) / 2)))
    
    # Generate grid points
    lat_points = np.arange(min_lat, max_lat + lat_resolution, lat_resolution)
    lon_points = np.arange(min_lon, max_lon + lon_resolution, lon_resolution)
    
    # Create meshgrid
    lat_grid, lon_grid = np.meshgrid(lat_points, lon_points)
    
    # Flatten to coordinate array
    coords = np.column_stack([lat_grid.flatten(), lon_grid.flatten()])
    
    return coords


def calculate_spatial_clustering(
    coords: np.ndarray,
    values: np.ndarray,
    method: str = 'dbscan',
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    Perform spatial clustering of grade values.
    
    Args:
        coords: Coordinate array (lat, lon)
        values: Values at each coordinate
        method: Clustering method ('dbscan', 'kmeans')
        **kwargs: Additional parameters for clustering
        
    Returns:
        Dictionary with cluster labels and statistics
    """
    from sklearn.cluster import DBSCAN, KMeans
    
    # Combine coordinates and values for clustering
    features = np.column_stack([coords, values.reshape(-1, 1)])
    
    if method == 'dbscan':
        eps = kwargs.get('eps', 0.01)
        min_samples = kwargs.get('min_samples', 5)
        clustering = DBSCAN(eps=eps, min_samples=min_samples)
    elif method == 'kmeans':
        n_clusters = kwargs.get('n_clusters', 5)
        clustering = KMeans(n_clusters=n_clusters, random_state=42)
    else:
        raise ValueError(f"Unsupported clustering method: {method}")
    
    labels = clustering.fit_predict(features)
    
    # Calculate cluster statistics
    cluster_stats = {}
    unique_labels = np.unique(labels)
    
    for label in unique_labels:
        if label == -1:  # Noise points in DBSCAN
            continue
        
        cluster_mask = labels == label
        cluster_values = values[cluster_mask]
        cluster_coords = coords[cluster_mask]
        
        cluster_stats[f'cluster_{label}'] = {
            'size': int(np.sum(cluster_mask)),
            'mean_value': float(np.mean(cluster_values)),
            'std_value': float(np.std(cluster_values)),
            'centroid': tuple(np.mean(cluster_coords, axis=0)),
            'value_range': (float(np.min(cluster_values)), float(np.max(cluster_values)))
        }
    
    return {
        'labels': labels,
        'cluster_stats': cluster_stats,
        'n_clusters': len(unique_labels) - (1 if -1 in unique_labels else 0)
    }
