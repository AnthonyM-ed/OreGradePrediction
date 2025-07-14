"""
Database Schema Validation
==========================

Validates database schema and data integrity for geological data tables.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from .connections import get_db_connection
from .queries import QueryExecutor

logger = logging.getLogger(__name__)

class SchemaValidator:
    """Validates database schema and data integrity"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.main_view = self.config.get('main_view', 'vw_HoleSamples_ElementGrades')
        self.main_table = self.config.get('main_table', 'tblHoleGrades_MapData_Standardized_Cache')
        self.coord_table = self.config.get('coordinates_table', 'tblDHColl')
        self.query_executor = QueryExecutor(config)
    
    def validate_table_exists(self, table_name: str) -> bool:
        """Check if table exists in database"""
        query = """
            SELECT COUNT(*) as table_count
            FROM INFORMATION_SCHEMA.TABLES 
            WHERE TABLE_NAME = ?
        """
        
        try:
            result = self.query_executor.execute_query(query, [table_name])
            return result[0]['table_count'] > 0
        except Exception as e:
            logger.error(f"Error checking table existence for {table_name}: {e}")
            return False
    
    def validate_column_exists(self, table_name: str, column_name: str) -> bool:
        """Check if column exists in table"""
        query = """
            SELECT COUNT(*) as column_count
            FROM INFORMATION_SCHEMA.COLUMNS 
            WHERE TABLE_NAME = ? AND COLUMN_NAME = ?
        """
        
        try:
            result = self.query_executor.execute_query(query, [table_name, column_name])
            return result[0]['column_count'] > 0
        except Exception as e:
            logger.error(f"Error checking column existence for {table_name}.{column_name}: {e}")
            return False
    
    def get_table_schema(self, table_name: str) -> List[Dict]:
        """Get table schema information"""
        query = """
            SELECT 
                COLUMN_NAME,
                DATA_TYPE,
                IS_NULLABLE,
                COLUMN_DEFAULT,
                CHARACTER_MAXIMUM_LENGTH,
                NUMERIC_PRECISION,
                NUMERIC_SCALE
            FROM INFORMATION_SCHEMA.COLUMNS 
            WHERE TABLE_NAME = ?
            ORDER BY ORDINAL_POSITION
        """
        
        try:
            return self.query_executor.execute_query(query, [table_name])
        except Exception as e:
            logger.error(f"Error getting schema for {table_name}: {e}")
            return []
    
    def validate_required_columns(self, table_name: str, required_columns: List[str]) -> Tuple[bool, List[str]]:
        """Validate that all required columns exist"""
        missing_columns = []
        
        for column in required_columns:
            if not self.validate_column_exists(table_name, column):
                missing_columns.append(column)
        
        is_valid = len(missing_columns) == 0
        return is_valid, missing_columns
    
    def validate_main_table_schema(self) -> Dict[str, Any]:
        """Validate main grades table schema"""
        required_columns = [
            'Hole_ID',
            'Element',
            'DataSet',
            'weighted_grade',
            'From_m',
            'To_m',
            'Interval_m'
        ]
        
        validation_result = {
            'table_exists': self.validate_table_exists(self.main_table),
            'schema_valid': False,
            'missing_columns': [],
            'schema_info': []
        }
        
        if validation_result['table_exists']:
            is_valid, missing_columns = self.validate_required_columns(self.main_table, required_columns)
            validation_result['schema_valid'] = is_valid
            validation_result['missing_columns'] = missing_columns
            validation_result['schema_info'] = self.get_table_schema(self.main_table)
        
        return validation_result
    
    def validate_coord_table_schema(self) -> Dict[str, Any]:
        """Validate coordinates table schema"""
        required_columns = [
            'Hole_ID',
            'LL_Lat',
            'LL_Long',
            'Elevation'
        ]
        
        validation_result = {
            'table_exists': self.validate_table_exists(self.coord_table),
            'schema_valid': False,
            'missing_columns': [],
            'schema_info': []
        }
        
        if validation_result['table_exists']:
            is_valid, missing_columns = self.validate_required_columns(self.coord_table, required_columns)
            validation_result['schema_valid'] = is_valid
            validation_result['missing_columns'] = missing_columns
            validation_result['schema_info'] = self.get_table_schema(self.coord_table)
        
        return validation_result
    
    def validate_view_exists(self, view_name: str) -> bool:
        """Check if view exists in database"""
        query = """
            SELECT COUNT(*) as view_count
            FROM INFORMATION_SCHEMA.VIEWS 
            WHERE TABLE_NAME = ?
        """
        
        try:
            result = self.query_executor.execute_query(query, [view_name])
            return result[0]['view_count'] > 0
        except Exception as e:
            logger.error(f"Error checking view existence for {view_name}: {e}")
            return False
    
    def validate_main_view_schema(self) -> Dict[str, Any]:
        """Validate main view schema"""
        required_columns = [
            'SampleID',
            'Hole_ID',
            'DataSet',
            'Element',
            'standardized_grade_ppm',
            'Depth_From',
            'Depth_To',
            'Interval_Length',
            'latitude',
            'longitude',
            'elevation',
            'LabCode'
        ]
        
        validation_result = {
            'view_exists': self.validate_view_exists(self.main_view),
            'schema_valid': False,
            'missing_columns': [],
            'schema_info': []
        }
        
        if validation_result['view_exists']:
            is_valid, missing_columns = self.validate_required_columns(self.main_view, required_columns)
            validation_result['schema_valid'] = is_valid
            validation_result['missing_columns'] = missing_columns
            validation_result['schema_info'] = self.get_table_schema(self.main_view)
        
        return validation_result
    
    def validate_data_integrity(self) -> Dict[str, Any]:
        """Validate data integrity across tables"""
        integrity_checks = {
            'hole_id_consistency': self._check_hole_id_consistency(),
            'coordinate_completeness': self._check_coordinate_completeness(),
            'grade_validity': self._check_grade_validity(),
            'depth_consistency': self._check_depth_consistency(),
            'element_distribution': self._check_element_distribution()
        }
        
        return integrity_checks
    
    def _check_hole_id_consistency(self) -> Dict[str, Any]:
        """Check if all hole IDs in grades table have coordinates"""
        query = f"""
            SELECT 
                COUNT(DISTINCT G.Hole_ID) as total_holes_in_grades,
                COUNT(DISTINCT C.Hole_ID) as total_holes_in_coords,
                COUNT(DISTINCT G.Hole_ID) - COUNT(DISTINCT CASE WHEN C.Hole_ID IS NOT NULL THEN G.Hole_ID END) as missing_coords
            FROM dbo.{self.main_table} G
            LEFT JOIN dbo.{self.coord_table} C ON G.Hole_ID = C.Hole_ID
        """
        
        try:
            result = self.query_executor.execute_query(query)
            data = result[0]
            
            return {
                'total_holes_in_grades': data['total_holes_in_grades'],
                'total_holes_in_coords': data['total_holes_in_coords'],
                'missing_coords': data['missing_coords'],
                'consistency_percentage': (data['total_holes_in_grades'] - data['missing_coords']) * 100.0 / data['total_holes_in_grades'] if data['total_holes_in_grades'] > 0 else 0
            }
        except Exception as e:
            logger.error(f"Error checking hole ID consistency: {e}")
            return {'error': str(e)}
    
    def _check_coordinate_completeness(self) -> Dict[str, Any]:
        """Check coordinate data completeness"""
        query = f"""
            SELECT 
                COUNT(*) as total_records,
                COUNT(CASE WHEN LL_Lat IS NOT NULL THEN 1 END) as lat_not_null,
                COUNT(CASE WHEN LL_Long IS NOT NULL THEN 1 END) as long_not_null,
                COUNT(CASE WHEN LL_Lat IS NOT NULL AND LL_Long IS NOT NULL THEN 1 END) as both_coords_not_null,
                COUNT(CASE WHEN Elevation IS NOT NULL THEN 1 END) as elevation_not_null
            FROM dbo.{self.coord_table}
        """
        
        try:
            result = self.query_executor.execute_query(query)
            data = result[0]
            
            return {
                'total_records': data['total_records'],
                'lat_completeness': data['lat_not_null'] * 100.0 / data['total_records'] if data['total_records'] > 0 else 0,
                'long_completeness': data['long_not_null'] * 100.0 / data['total_records'] if data['total_records'] > 0 else 0,
                'coord_completeness': data['both_coords_not_null'] * 100.0 / data['total_records'] if data['total_records'] > 0 else 0,
                'elevation_completeness': data['elevation_not_null'] * 100.0 / data['total_records'] if data['total_records'] > 0 else 0
            }
        except Exception as e:
            logger.error(f"Error checking coordinate completeness: {e}")
            return {'error': str(e)}
    
    def _check_grade_validity(self) -> Dict[str, Any]:
        """Check grade data validity"""
        query = f"""
            SELECT 
                COUNT(*) as total_records,
                COUNT(CASE WHEN weighted_grade IS NOT NULL THEN 1 END) as grade_not_null,
                COUNT(CASE WHEN weighted_grade > 0 THEN 1 END) as grade_positive,
                COUNT(CASE WHEN weighted_grade = 0 THEN 1 END) as grade_zero,
                COUNT(CASE WHEN weighted_grade < 0 THEN 1 END) as grade_negative,
                AVG(weighted_grade) as avg_grade,
                MIN(weighted_grade) as min_grade,
                MAX(weighted_grade) as max_grade
            FROM dbo.{self.main_table}
            WHERE weighted_grade IS NOT NULL
        """
        
        try:
            result = self.query_executor.execute_query(query)
            data = result[0]
            
            return {
                'total_records': data['total_records'],
                'grade_completeness': data['grade_not_null'] * 100.0 / data['total_records'] if data['total_records'] > 0 else 0,
                'positive_grades': data['grade_positive'],
                'zero_grades': data['grade_zero'],
                'negative_grades': data['grade_negative'],
                'avg_grade': float(data['avg_grade']) if data['avg_grade'] else 0,
                'min_grade': float(data['min_grade']) if data['min_grade'] else 0,
                'max_grade': float(data['max_grade']) if data['max_grade'] else 0
            }
        except Exception as e:
            logger.error(f"Error checking grade validity: {e}")
            return {'error': str(e)}
    
    def _check_depth_consistency(self) -> Dict[str, Any]:
        """Check depth data consistency"""
        query = f"""
            SELECT 
                COUNT(*) as total_records,
                COUNT(CASE WHEN From_m IS NOT NULL AND To_m IS NOT NULL THEN 1 END) as depth_complete,
                COUNT(CASE WHEN From_m >= To_m THEN 1 END) as invalid_depth_order,
                COUNT(CASE WHEN Interval_m IS NOT NULL AND ABS(Interval_m - (To_m - From_m)) > 0.01 THEN 1 END) as interval_mismatch,
                AVG(To_m - From_m) as avg_interval,
                MIN(From_m) as min_from_depth,
                MAX(To_m) as max_to_depth
            FROM dbo.{self.main_table}
            WHERE From_m IS NOT NULL AND To_m IS NOT NULL
        """
        
        try:
            result = self.query_executor.execute_query(query)
            data = result[0]
            
            return {
                'total_records': data['total_records'],
                'depth_completeness': data['depth_complete'] * 100.0 / data['total_records'] if data['total_records'] > 0 else 0,
                'invalid_depth_order': data['invalid_depth_order'],
                'interval_mismatch': data['interval_mismatch'],
                'avg_interval': float(data['avg_interval']) if data['avg_interval'] else 0,
                'min_from_depth': float(data['min_from_depth']) if data['min_from_depth'] else 0,
                'max_to_depth': float(data['max_to_depth']) if data['max_to_depth'] else 0
            }
        except Exception as e:
            logger.error(f"Error checking depth consistency: {e}")
            return {'error': str(e)}
    
    def _check_element_distribution(self) -> Dict[str, Any]:
        """Check element distribution"""
        query = f"""
            SELECT 
                Element,
                COUNT(*) as record_count,
                COUNT(DISTINCT Hole_ID) as unique_holes,
                AVG(weighted_grade) as avg_grade,
                MIN(weighted_grade) as min_grade,
                MAX(weighted_grade) as max_grade
            FROM dbo.{self.main_table}
            WHERE weighted_grade IS NOT NULL AND weighted_grade > 0
            GROUP BY Element
            ORDER BY record_count DESC
        """
        
        try:
            result = self.query_executor.execute_query(query)
            
            return {
                'element_count': len(result),
                'elements': result
            }
        except Exception as e:
            logger.error(f"Error checking element distribution: {e}")
            return {'error': str(e)}
    
    def run_complete_validation(self) -> Dict[str, Any]:
        """Run complete schema and data validation"""
        import datetime
        
        validation_report = {
            'timestamp': datetime.datetime.now().isoformat(),
            'main_view_validation': self.validate_main_view_schema(),
            'main_table_validation': self.validate_main_table_schema(),
            'coord_table_validation': self.validate_coord_table_schema(),
            'data_integrity': self.validate_data_integrity()
        }
        
        # Overall validation status
        validation_report['overall_status'] = (
            validation_report['main_view_validation']['schema_valid'] and
            validation_report['main_table_validation']['schema_valid'] and
            validation_report['coord_table_validation']['schema_valid']
        )
        
        logger.info(f"Complete validation completed. Overall status: {validation_report['overall_status']}")
        return validation_report

# Global schema validator instance
schema_validator = SchemaValidator()
