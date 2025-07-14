"""
Predefined SQL Queries for Geological Data
==========================================

Contains commonly used SQL queries for geological data analysis and ML model training.
"""

import logging
from typing import Dict, List, Any, Optional
from .connections import get_db_connection

logger = logging.getLogger(__name__)

class GeologicalQueries:
    """Predefined queries for geological data operations"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.main_view = self.config.get('main_view', 'vw_HoleSamples_ElementGrades')
        self.main_table = self.config.get('main_table', 'tblHoleGrades_MapData_Standardized_Cache')
        self.coord_table = self.config.get('coordinates_table', 'tblDHColl')
    
    def get_elements_summary(self) -> str:
        """Get summary of all elements in database using the new view"""
        return f"""
            SELECT 
                Element,
                COUNT(*) as total_records,
                COUNT(DISTINCT Hole_ID) as unique_holes,
                COUNT(DISTINCT SampleID) as unique_samples,
                AVG(standardized_grade_ppm) as avg_grade,
                MIN(standardized_grade_ppm) as min_grade,
                MAX(standardized_grade_ppm) as max_grade,
                STDEV(standardized_grade_ppm) as std_grade
            FROM dbo.{self.main_view}
            WHERE standardized_grade_ppm IS NOT NULL AND standardized_grade_ppm > 0
            GROUP BY Element
            ORDER BY total_records DESC
        """
    
    def get_spatial_extent(self) -> str:
        """Get spatial extent of drill holes using the new view"""
        return f"""
            SELECT 
                MIN(latitude) as min_lat,
                MAX(latitude) as max_lat,
                MIN(longitude) as min_lon,
                MAX(longitude) as max_lon,
                COUNT(DISTINCT Hole_ID) as total_holes
            FROM dbo.{self.main_view}
            WHERE latitude IS NOT NULL AND longitude IS NOT NULL
        """
    
    def get_data_quality_metrics(self) -> str:
        """Get comprehensive data quality metrics using the new view"""
        return f"""
            WITH base_stats AS (
                SELECT 
                    COUNT(*) as total_records,
                    COUNT(DISTINCT Hole_ID) as unique_holes,
                    COUNT(DISTINCT SampleID) as unique_samples,
                    COUNT(DISTINCT Element) as unique_elements,
                    COUNT(DISTINCT DataSet) as unique_datasets,
                    COUNT(DISTINCT LabCode) as unique_labs
                FROM dbo.{self.main_view}
            ),
            coord_stats AS (
                SELECT 
                    COUNT(*) as records_with_coords,
                    COUNT(CASE WHEN latitude IS NOT NULL AND longitude IS NOT NULL THEN 1 END) as valid_coords,
                    COUNT(CASE WHEN elevation IS NOT NULL THEN 1 END) as elevation_not_null
                FROM dbo.{self.main_view}
            ),
            grade_stats AS (
                SELECT 
                    COUNT(CASE WHEN standardized_grade_ppm IS NOT NULL THEN 1 END) as non_null_grades,
                    COUNT(CASE WHEN standardized_grade_ppm > 0 THEN 1 END) as positive_grades,
                    COUNT(CASE WHEN standardized_grade_ppm <= 0 THEN 1 END) as zero_negative_grades
                FROM dbo.{self.main_view}
            ),
            depth_stats AS (
                SELECT 
                    COUNT(CASE WHEN Depth_From IS NOT NULL AND Depth_To IS NOT NULL THEN 1 END) as depth_complete,
                    COUNT(CASE WHEN Interval_Length IS NOT NULL THEN 1 END) as interval_not_null
                FROM dbo.{self.main_view}
            )
            SELECT 
                b.total_records,
                b.unique_holes,
                b.unique_samples,
                b.unique_elements,
                b.unique_datasets,
                b.unique_labs,
                c.records_with_coords,
                c.valid_coords,
                c.elevation_not_null,
                g.non_null_grades,
                g.positive_grades,
                g.zero_negative_grades,
                d.depth_complete,
                d.interval_not_null,
                CAST(c.valid_coords * 100.0 / c.records_with_coords as DECIMAL(5,2)) as coord_completeness_pct,
                CAST(g.positive_grades * 100.0 / b.total_records as DECIMAL(5,2)) as grade_validity_pct,
                CAST(d.depth_complete * 100.0 / b.total_records as DECIMAL(5,2)) as depth_completeness_pct
            FROM base_stats b
            CROSS JOIN coord_stats c
            CROSS JOIN grade_stats g
            CROSS JOIN depth_stats d
        """
    
    def get_training_data(self, elements: List[str], datasets: List[str] = None, limit: Optional[int] = None) -> str:
        """Get training data for ML model using the new view"""
        element_placeholders = ', '.join(['?' for _ in elements])
        limit_clause = f"TOP {limit}" if limit else ""
        
        # Add dataset filtering if provided
        dataset_filter = ""
        if datasets:
            dataset_placeholders = ', '.join(['?' for _ in datasets])
            dataset_filter = f"AND DataSet IN ({dataset_placeholders})"
        
        return f"""
            SELECT {limit_clause}
                SampleID,
                Hole_ID,
                Element,
                DataSet,
                standardized_grade_ppm,
                latitude,
                longitude,
                elevation,
                Depth_From,
                Depth_To,
                Interval_Length,
                LabCode,
                -- Derived features
                (Depth_To + Depth_From) / 2.0 as mid_depth,
                standardized_grade_ppm * Interval_Length as grade_tonnage_proxy
            FROM dbo.{self.main_view}
            WHERE Element IN ({element_placeholders})
                {dataset_filter}
                AND latitude IS NOT NULL
                AND longitude IS NOT NULL
                AND standardized_grade_ppm IS NOT NULL
                AND standardized_grade_ppm > 0
                AND Interval_Length > 0
                AND Depth_From IS NOT NULL
                AND Depth_To IS NOT NULL
            ORDER BY Hole_ID, Element, Depth_From
        """
    
    def get_prediction_grid(self, bounds: Dict[str, float], grid_size: float = 0.01) -> str:
        """Generate prediction grid for spatial interpolation"""
        return f"""
            WITH grid_points AS (
                SELECT 
                    lat_point,
                    lon_point
                FROM (
                    SELECT 
                        ? + (ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) - 1) * ? as lat_point
                    FROM sys.objects
                ) lat_seq
                CROSS JOIN (
                    SELECT 
                        ? + (ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) - 1) * ? as lon_point
                    FROM sys.objects
                ) lon_seq
                WHERE lat_point <= ? AND lon_point <= ?
            )
            SELECT 
                lat_point as latitude,
                lon_point as longitude,
                -- Calculate distance to nearest drill hole
                (
                    SELECT MIN(
                        SQRT(POWER(V.latitude - g.lat_point, 2) + POWER(V.longitude - g.lon_point, 2))
                    )
                    FROM dbo.{self.main_view} V
                    WHERE V.latitude IS NOT NULL AND V.longitude IS NOT NULL
                ) as distance_to_nearest_hole
            FROM grid_points g
            ORDER BY lat_point, lon_point
        """
    
    def get_element_correlations(self, elements: List[str]) -> str:
        """Get correlation data between elements using the new view"""
        element_placeholders = ', '.join(['?' for _ in elements])
        
        return f"""
            WITH element_pairs AS (
                SELECT 
                    e1.Hole_ID,
                    e1.SampleID,
                    e1.Element as element1,
                    e1.standardized_grade_ppm as grade1,
                    e2.Element as element2,
                    e2.standardized_grade_ppm as grade2,
                    e1.latitude,
                    e1.longitude,
                    e1.Depth_From,
                    e1.Depth_To
                FROM dbo.{self.main_view} e1
                INNER JOIN dbo.{self.main_view} e2 ON e1.Hole_ID = e2.Hole_ID 
                    AND ABS(e1.Depth_From - e2.Depth_From) < 0.1 
                    AND ABS(e1.Depth_To - e2.Depth_To) < 0.1
                WHERE e1.Element IN ({element_placeholders})
                    AND e2.Element IN ({element_placeholders})
                    AND e1.Element < e2.Element
                    AND e1.standardized_grade_ppm > 0
                    AND e2.standardized_grade_ppm > 0
                    AND e1.latitude IS NOT NULL
                    AND e1.longitude IS NOT NULL
            )
            SELECT 
                element1,
                element2,
                COUNT(*) as sample_count,
                AVG(grade1) as avg_grade1,
                AVG(grade2) as avg_grade2,
                -- Basic correlation coefficient calculation
                (COUNT(*) * SUM(grade1 * grade2) - SUM(grade1) * SUM(grade2)) /
                (SQRT(COUNT(*) * SUM(grade1 * grade1) - SUM(grade1) * SUM(grade1)) *
                 SQRT(COUNT(*) * SUM(grade2 * grade2) - SUM(grade2) * SUM(grade2))) as correlation
            FROM element_pairs
            GROUP BY element1, element2
            ORDER BY ABS(correlation) DESC
        """
    
    def get_lab_summary(self) -> str:
        """Get summary by laboratory code"""
        return f"""
            SELECT 
                LabCode,
                COUNT(*) as total_records,
                COUNT(DISTINCT Element) as unique_elements,
                COUNT(DISTINCT Hole_ID) as unique_holes,
                COUNT(DISTINCT DataSet) as unique_datasets,
                AVG(standardized_grade_ppm) as avg_grade,
                MIN(standardized_grade_ppm) as min_grade,
                MAX(standardized_grade_ppm) as max_grade
            FROM dbo.{self.main_view}
            WHERE standardized_grade_ppm IS NOT NULL AND standardized_grade_ppm > 0
            GROUP BY LabCode
            ORDER BY total_records DESC
        """
    
    def get_dataset_summary(self) -> str:
        """Get summary by dataset"""
        return f"""
            SELECT 
                DataSet,
                COUNT(*) as total_records,
                COUNT(DISTINCT Element) as unique_elements,
                COUNT(DISTINCT Hole_ID) as unique_holes,
                COUNT(DISTINCT LabCode) as unique_labs,
                AVG(standardized_grade_ppm) as avg_grade,
                MIN(standardized_grade_ppm) as min_grade,
                MAX(standardized_grade_ppm) as max_grade,
                AVG(Interval_Length) as avg_interval,
                MIN(Depth_From) as min_depth,
                MAX(Depth_To) as max_depth
            FROM dbo.{self.main_view}
            WHERE standardized_grade_ppm IS NOT NULL AND standardized_grade_ppm > 0
            GROUP BY DataSet
            ORDER BY total_records DESC
        """
    
    def get_depth_statistics(self) -> str:
        """Get depth-related statistics"""
        return f"""
            SELECT 
                COUNT(*) as total_records,
                AVG(Depth_From) as avg_depth_from,
                AVG(Depth_To) as avg_depth_to,
                AVG(Interval_Length) as avg_interval,
                MIN(Depth_From) as min_depth,
                MAX(Depth_To) as max_depth,
                STDEV(Interval_Length) as std_interval,
                COUNT(CASE WHEN Interval_Length > 5.0 THEN 1 END) as long_intervals,
                COUNT(CASE WHEN Interval_Length < 1.0 THEN 1 END) as short_intervals
            FROM dbo.{self.main_view}
            WHERE Depth_From IS NOT NULL 
                AND Depth_To IS NOT NULL 
                AND Interval_Length IS NOT NULL
        """
    
    def get_spatial_data_by_bounds(self, elements: List[str] = None, datasets: List[str] = None) -> str:
        """Get spatial data within specific bounds"""
        element_filter = ""
        dataset_filter = ""
        
        if elements:
            element_placeholders = ', '.join(['?' for _ in elements])
            element_filter = f"AND Element IN ({element_placeholders})"
        
        if datasets:
            dataset_placeholders = ', '.join(['?' for _ in datasets])
            dataset_filter = f"AND DataSet IN ({dataset_placeholders})"
        
        return f"""
            SELECT 
                SampleID,
                Hole_ID,
                Element,
                DataSet,
                standardized_grade_ppm,
                latitude,
                longitude,
                elevation,
                Depth_From,
                Depth_To,
                LabCode
            FROM dbo.{self.main_view}
            WHERE latitude BETWEEN ? AND ?
                AND longitude BETWEEN ? AND ?
                AND standardized_grade_ppm IS NOT NULL
                AND standardized_grade_ppm > 0
                {element_filter}
                {dataset_filter}
            ORDER BY latitude, longitude, Depth_From
        """

class QueryExecutor:
    """Executes predefined queries with parameters"""
    
    def __init__(self, config: Dict = None):
        self.queries = GeologicalQueries(config)
    
    def execute_query(self, query: str, params: List[Any] = None) -> List[Dict]:
        """Execute query and return results as list of dictionaries"""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            try:
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                
                # Get column names
                columns = [column[0] for column in cursor.description]
                
                # Fetch all results and convert to dictionaries
                results = []
                for row in cursor.fetchall():
                    results.append(dict(zip(columns, row)))
                
                logger.info(f"Query executed successfully, returned {len(results)} rows")
                return results
                
            except Exception as e:
                logger.error(f"Query execution failed: {e}")
                raise
            finally:
                cursor.close()
    
    def get_elements_summary(self) -> List[Dict]:
        """Get elements summary"""
        query = self.queries.get_elements_summary()
        return self.execute_query(query)
    
    def get_spatial_extent(self) -> Dict:
        """Get spatial extent"""
        query = self.queries.get_spatial_extent()
        result = self.execute_query(query)
        return result[0] if result else {}
    
    def get_data_quality_metrics(self) -> Dict:
        """Get data quality metrics"""
        query = self.queries.get_data_quality_metrics()
        result = self.execute_query(query)
        return result[0] if result else {}
    
    def get_training_data(self, elements: List[str], limit: Optional[int] = None) -> List[Dict]:
        """Get training data"""
        query = self.queries.get_training_data(elements, limit)
        return self.execute_query(query, elements)
    
    def get_training_data(self, elements: List[str], datasets: List[str] = None, limit: Optional[int] = None) -> List[Dict]:
        """Get training data with optional dataset filtering"""
        params = elements.copy()
        if datasets:
            params.extend(datasets)
        
        query = self.queries.get_training_data(elements, datasets, limit)
        return self.execute_query(query, params)
    
    def get_element_correlations(self, elements: List[str]) -> List[Dict]:
        """Get element correlations"""
        query = self.queries.get_element_correlations(elements)
        return self.execute_query(query, elements * 2)  # Elements used twice in the query
    
    def get_lab_summary(self) -> List[Dict]:
        """Get laboratory summary"""
        query = self.queries.get_lab_summary()
        return self.execute_query(query)
    
    def get_dataset_summary(self) -> List[Dict]:
        """Get dataset summary"""
        query = self.queries.get_dataset_summary()
        return self.execute_query(query)
    
    def get_depth_statistics(self) -> Dict:
        """Get depth statistics"""
        query = self.queries.get_depth_statistics()
        result = self.execute_query(query)
        return result[0] if result else {}
    
    def get_spatial_data_by_bounds(self, min_lat: float, max_lat: float, 
                                 min_lon: float, max_lon: float,
                                 elements: List[str] = None, 
                                 datasets: List[str] = None) -> List[Dict]:
        """Get spatial data within bounds"""
        params = [min_lat, max_lat, min_lon, max_lon]
        
        if elements:
            params.extend(elements)
        if datasets:
            params.extend(datasets)
        
        query = self.queries.get_spatial_data_by_bounds(elements, datasets)
        return self.execute_query(query, params)

# Global query executor instance
query_executor = QueryExecutor()
