"""
Specialized Data Extractors
===========================

Extracts and prepares geological data for specific ML tasks and analysis.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from .connections import get_db_connection
from .queries import QueryExecutor
from ..data_processing.query_builder import SQLQueryBuilder, QueryFilter

logger = logging.getLogger(__name__)

class GeologicalDataExtractor:
    """Extracts geological data for ML model training and inference"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.query_executor = QueryExecutor(config)
        self.query_builder = SQLQueryBuilder(config)
    
    def extract_training_data(self, elements: List[str], 
                          filters: List[QueryFilter] = None,
                          include_features: bool = True,
                          max_records: int = None) -> pd.DataFrame:
        """Extract training data for ML models"""
        try:
            # Build query using the correct parameter name
            query, params = self.query_builder.build_drilling_query(
                elements=elements,
                filters=filters,
                limit=max_records
            )
            
            # Execute
            with get_db_connection() as conn:
                df = pd.read_sql(query, conn, params=params)
            
            if include_features:
                df = self._add_geological_features(df)
            
            logger.info(f"Extracted {len(df)} training records for elements: {elements}")
            return df
            
        except Exception as e:
            logger.error(f"Error extracting training data: {e}")
            raise
    
    def extract_spatial_data(self, center_lat: float, center_lon: float,
                           radius_km: float = 10.0, 
                           elements: List[str] = None) -> pd.DataFrame:
        """Extract spatial data around a center point"""
        try:
            query, params = self.query_builder.build_spatial_query(
                center_lat=center_lat,
                center_lon=center_lon,
                radius_km=radius_km,
                elements=elements
            )
            
            with get_db_connection() as conn:
                df = pd.read_sql(query, conn, params=params)
            
            logger.info(f"Extracted {len(df)} spatial records within {radius_km}km radius")
            return df
            
        except Exception as e:
            logger.error(f"Error extracting spatial data: {e}")
            raise
    
    def extract_multi_element_data(self, elements: List[str],
                                 filters: List[QueryFilter] = None) -> pd.DataFrame:
        """Extract multi-element data with pivot structure"""
        try:
            query, params = self.query_builder.build_multi_element_pivot_query(
                elements=elements,
                filters=filters
            )
            
            with get_db_connection() as conn:
                df = pd.read_sql(query, conn, params=params)
            
            # Add inter-element features
            df = self._add_multi_element_features(df, elements)
            
            logger.info(f"Extracted {len(df)} multi-element records")
            return df
            
        except Exception as e:
            logger.error(f"Error extracting multi-element data: {e}")
            raise
    
    def extract_time_series_data(self, hole_id: str, 
                               elements: List[str] = None) -> pd.DataFrame:
        """Extract time series data for a specific hole (depth series)"""
        try:
            filters = [QueryFilter('Hole_ID', '=', hole_id, 'G')]
            
            query, params = self.query_builder.build_drilling_query(
                elements=elements,
                filters=filters
            )
            
            with get_db_connection() as conn:
                df = pd.read_sql(query, conn, params=params)
            
            # Sort by depth
            df = df.sort_values(['depth_from', 'Element'])
            
            # Add depth-based features
            df = self._add_depth_features(df)
            
            logger.info(f"Extracted {len(df)} depth series records for hole {hole_id}")
            return df
            
        except Exception as e:
            logger.error(f"Error extracting time series data: {e}")
            raise
    
    def extract_anomaly_data(self, element: str = 'Cu',
                           std_threshold: float = 2.0) -> pd.DataFrame:
        """Extract anomaly data for outlier analysis"""
        try:
            query, params = self.query_builder.build_anomaly_detection_query(
                element=element,
                std_threshold=std_threshold
            )
            
            with get_db_connection() as conn:
                df = pd.read_sql(query, conn, params=params)
            
            logger.info(f"Extracted {len(df)} anomaly records for {element}")
            return df
            
        except Exception as e:
            logger.error(f"Error extracting anomaly data: {e}")
            raise
    
    def extract_correlation_data(self, elements: List[str]) -> pd.DataFrame:
        """Extract data for correlation analysis"""
        try:
            result = self.query_executor.get_element_correlations(elements)
            df = pd.DataFrame(result)
            
            logger.info(f"Extracted correlation data for {len(elements)} elements")
            return df
            
        except Exception as e:
            logger.error(f"Error extracting correlation data: {e}")
            raise
    
    def extract_prediction_grid(self, bounds: Dict[str, float], 
                              grid_size: float = 0.01) -> pd.DataFrame:
        """Extract prediction grid for spatial interpolation"""
        try:
            # Generate grid points
            lat_range = np.arange(bounds['min_lat'], bounds['max_lat'], grid_size)
            lon_range = np.arange(bounds['min_lon'], bounds['max_lon'], grid_size)
            
            # Create grid
            lat_grid, lon_grid = np.meshgrid(lat_range, lon_range)
            
            # Flatten to create prediction points
            prediction_points = pd.DataFrame({
                'latitude': lat_grid.flatten(),
                'longitude': lon_grid.flatten()
            })
            
            # Add nearest hole distance
            prediction_points = self._add_nearest_hole_distance(prediction_points)
            
            logger.info(f"Generated {len(prediction_points)} prediction grid points")
            return prediction_points
            
        except Exception as e:
            logger.error(f"Error generating prediction grid: {e}")
            raise
    
    def _add_geological_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add geological features to the dataset"""
        # Depth features
        df['mid_depth'] = (df['depth_from'] + df['depth_to']) / 2
        df['depth_range'] = df['depth_to'] - df['depth_from']
        
        # Grade features
        df['log_grade'] = np.log1p(df['standardized_grade_ppm'])
        df['grade_intensity'] = df['standardized_grade_ppm'] * df['interval_length']
        
        # Spatial features (if coordinates available)
        if 'latitude' in df.columns and 'longitude' in df.columns:
            df['spatial_hash'] = df['latitude'].round(3).astype(str) + '_' + df['longitude'].round(3).astype(str)
            
            # Distance from center
            center_lat = df['latitude'].median()
            center_lon = df['longitude'].median()
            df['distance_from_center'] = np.sqrt(
                (df['latitude'] - center_lat)**2 + 
                (df['longitude'] - center_lon)**2
            )
        
        return df
    
    def _add_multi_element_features(self, df: pd.DataFrame, elements: List[str]) -> pd.DataFrame:
        """Add multi-element features"""
        # Element ratios
        if len(elements) >= 2:
            for i, elem1 in enumerate(elements):
                for elem2 in elements[i+1:]:
                    if elem1 in df.columns and elem2 in df.columns:
                        # Avoid division by zero
                        df[f'{elem1}_{elem2}_ratio'] = df[elem1] / (df[elem2] + 1e-8)
        
        # Total grade
        element_cols = [col for col in df.columns if col in elements]
        if element_cols:
            df['total_grade'] = df[element_cols].sum(axis=1, skipna=True)
            df['grade_diversity'] = df[element_cols].count(axis=1)
        
        return df
    
    def _add_depth_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add depth-based features for time series analysis"""
        # Depth progression
        df['depth_progression'] = df['depth_from'] / df['depth_from'].max()
        
        # Cumulative features
        df['cumulative_grade'] = df['standardized_grade_ppm'].cumsum()
        df['cumulative_length'] = df['interval_length'].cumsum()
        
        # Moving averages (if enough data)
        if len(df) >= 3:
            df['grade_ma3'] = df['standardized_grade_ppm'].rolling(window=3, center=True).mean()
            df['grade_std3'] = df['standardized_grade_ppm'].rolling(window=3, center=True).std()
        
        return df
    
    def _add_nearest_hole_distance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add distance to nearest drill hole"""
        # This is a placeholder - in practice, you'd query the database
        # for actual drill hole locations
        try:
            # Get drill hole coordinates from the main view
            coord_query = """
                SELECT DISTINCT latitude, longitude 
                FROM vw_HoleSamples_ElementGrades 
                WHERE latitude IS NOT NULL AND longitude IS NOT NULL
            """
            
            with get_db_connection() as conn:
                hole_coords = pd.read_sql(coord_query, conn)
            
            # Calculate minimum distance to any hole
            distances = []
            for _, point in df.iterrows():
                point_distances = np.sqrt(
                    (hole_coords['latitude'] - point['latitude'])**2 + 
                    (hole_coords['longitude'] - point['longitude'])**2
                )
                distances.append(point_distances.min())
            
            df['distance_to_nearest_hole'] = distances
            
        except Exception as e:
            logger.warning(f"Could not calculate nearest hole distance: {e}")
            df['distance_to_nearest_hole'] = 0
        
        return df

class StatisticalDataExtractor:
    """Extracts statistical summaries and aggregations"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.query_executor = QueryExecutor(config)
    
    def extract_element_statistics(self) -> pd.DataFrame:
        """Extract comprehensive element statistics"""
        try:
            result = self.query_executor.get_elements_summary()
            df = pd.DataFrame(result)
            
            # Add derived statistics
            if not df.empty:
                # Convert to float to avoid decimal issues
                df['std_grade'] = df['std_grade'].astype(float)
                df['avg_grade'] = df['avg_grade'].astype(float)
                df['total_records'] = df['total_records'].astype(float)
                df['unique_holes'] = df['unique_holes'].astype(float)
                
                df['coefficient_of_variation'] = df['std_grade'] / df['avg_grade']
                df['records_per_hole'] = df['total_records'] / df['unique_holes']
            
            logger.info(f"Extracted statistics for {len(df)} elements")
            return df
            
        except Exception as e:
            logger.error(f"Error extracting element statistics: {e}")
            raise
    
    def extract_spatial_statistics(self) -> Dict[str, Any]:
        """Extract spatial extent and statistics"""
        try:
            result = self.query_executor.get_spatial_extent()
            
            # Add derived spatial metrics
            if result:
                result['lat_range'] = result['max_lat'] - result['min_lat']
                result['lon_range'] = result['max_lon'] - result['min_lon']
                result['area_degrees'] = result['lat_range'] * result['lon_range']
            
            logger.info("Extracted spatial statistics")
            return result
            
        except Exception as e:
            logger.error(f"Error extracting spatial statistics: {e}")
            raise
    
    def extract_data_quality_summary(self) -> Dict[str, Any]:
        """Extract data quality summary"""
        try:
            result = self.query_executor.get_data_quality_metrics()
            
            logger.info("Extracted data quality summary")
            return result
            
        except Exception as e:
            logger.error(f"Error extracting data quality summary: {e}")
            raise

# Global extractor instances
geological_extractor = GeologicalDataExtractor()
statistical_extractor = StatisticalDataExtractor()
