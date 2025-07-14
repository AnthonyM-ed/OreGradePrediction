"""
Data Validation for Geological ML Models
========================================

Validates data quality and integrity for geological datasets.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

class DataValidator:
    """Validates geological data quality"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.validation_rules = self._load_validation_rules()
    
    def _load_validation_rules(self) -> Dict[str, Any]:
        """Load validation rules for geological data"""
        return {
            'coordinates': {
                'latitude_range': (-90, 90),
                'longitude_range': (-180, 180),
                'required_fields': ['latitude', 'longitude']
            },
            'depths': {
                'min_depth': 0,
                'max_depth': 10000,
                'required_fields': ['depth_from', 'depth_to']
            },
            'grades': {
                'min_grade': 0,
                'max_grade': 1000000,  # PPM values can be higher
                'required_fields': ['standardized_grade_ppm']
            },
            'intervals': {
                'min_interval': 0.01,
                'max_interval': 1000,
                'required_fields': ['interval_length']
            }
        }
    
    def validate_coordinates(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate coordinate data"""
        validation_result = {
            'passed': True,
            'issues': [],
            'statistics': {}
        }
        
        try:
            coord_rules = self.validation_rules['coordinates']
            
            # Check required fields
            for field in coord_rules['required_fields']:
                if field not in df.columns:
                    validation_result['issues'].append(f"Missing required field: {field}")
                    validation_result['passed'] = False
                    continue
                
                # Check for null values
                null_count = df[field].isnull().sum()
                if null_count > 0:
                    validation_result['issues'].append(
                        f"{field} has {null_count} null values ({null_count/len(df)*100:.1f}%)"
                    )
                
                # Check coordinate ranges
                if field == 'latitude':
                    lat_range = coord_rules['latitude_range']
                    invalid_lat = df[
                        (df[field] < lat_range[0]) | (df[field] > lat_range[1])
                    ]
                    if len(invalid_lat) > 0:
                        validation_result['issues'].append(
                            f"Found {len(invalid_lat)} invalid latitude values"
                        )
                        validation_result['passed'] = False
                
                elif field == 'longitude':
                    lon_range = coord_rules['longitude_range']
                    invalid_lon = df[
                        (df[field] < lon_range[0]) | (df[field] > lon_range[1])
                    ]
                    if len(invalid_lon) > 0:
                        validation_result['issues'].append(
                            f"Found {len(invalid_lon)} invalid longitude values"
                        )
                        validation_result['passed'] = False
                
                # Calculate statistics
                if field in df.columns:
                    validation_result['statistics'][field] = {
                        'count': df[field].count(),
                        'null_count': df[field].isnull().sum(),
                        'min': df[field].min(),
                        'max': df[field].max(),
                        'mean': df[field].mean(),
                        'std': df[field].std()
                    }
            
            logger.info("Coordinate validation completed")
            return validation_result
            
        except Exception as e:
            logger.error(f"Error validating coordinates: {e}")
            validation_result['passed'] = False
            validation_result['issues'].append(f"Validation error: {str(e)}")
            return validation_result
    
    def validate_depths(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate depth data"""
        validation_result = {
            'passed': True,
            'issues': [],
            'statistics': {}
        }
        
        try:
            depth_rules = self.validation_rules['depths']
            
            # Check required fields
            for field in depth_rules['required_fields']:
                if field not in df.columns:
                    validation_result['issues'].append(f"Missing required field: {field}")
                    validation_result['passed'] = False
                    continue
                
                # Check for null values
                null_count = df[field].isnull().sum()
                if null_count > 0:
                    validation_result['issues'].append(
                        f"{field} has {null_count} null values ({null_count/len(df)*100:.1f}%)"
                    )
                
                # Check depth ranges
                invalid_depths = df[
                    (df[field] < depth_rules['min_depth']) | 
                    (df[field] > depth_rules['max_depth'])
                ]
                if len(invalid_depths) > 0:
                    validation_result['issues'].append(
                        f"Found {len(invalid_depths)} invalid {field} values"
                    )
                    validation_result['passed'] = False
                
                # Calculate statistics
                validation_result['statistics'][field] = {
                    'count': df[field].count(),
                    'null_count': df[field].isnull().sum(),
                    'min': df[field].min(),
                    'max': df[field].max(),
                    'mean': df[field].mean(),
                    'std': df[field].std()
                }
            
            # Check depth consistency
            if 'from_depth' in df.columns and 'to_depth' in df.columns:
                invalid_order = df[df['from_depth'] >= df['to_depth']]
                if len(invalid_order) > 0:
                    validation_result['issues'].append(
                        f"Found {len(invalid_order)} records with from_depth >= to_depth"
                    )
                    validation_result['passed'] = False
                
                # Check interval consistency
                if 'interval_length' in df.columns:
                    calculated_interval = df['to_depth'] - df['from_depth']
                    interval_mismatch = abs(df['interval_length'] - calculated_interval) > 0.01
                    mismatch_count = interval_mismatch.sum()
                    
                    if mismatch_count > 0:
                        validation_result['issues'].append(
                            f"Found {mismatch_count} records with interval length mismatch"
                        )
            
            logger.info("Depth validation completed")
            return validation_result
            
        except Exception as e:
            logger.error(f"Error validating depths: {e}")
            validation_result['passed'] = False
            validation_result['issues'].append(f"Validation error: {str(e)}")
            return validation_result
    
    def validate_grades(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate grade data"""
        validation_result = {
            'passed': True,
            'issues': [],
            'statistics': {}
        }
        
        try:
            grade_rules = self.validation_rules['grades']
            
            # Check required fields
            for field in grade_rules['required_fields']:
                if field not in df.columns:
                    validation_result['issues'].append(f"Missing required field: {field}")
                    validation_result['passed'] = False
                    continue
                
                # Check for null values
                null_count = df[field].isnull().sum()
                if null_count > 0:
                    validation_result['issues'].append(
                        f"{field} has {null_count} null values ({null_count/len(df)*100:.1f}%)"
                    )
                
                # Check for negative grades
                negative_grades = df[df[field] < 0]
                if len(negative_grades) > 0:
                    validation_result['issues'].append(
                        f"Found {len(negative_grades)} negative grade values"
                    )
                
                # Check for extremely high grades
                high_grades = df[df[field] > grade_rules['max_grade']]
                if len(high_grades) > 0:
                    validation_result['issues'].append(
                        f"Found {len(high_grades)} extremely high grade values (>{grade_rules['max_grade']})"
                    )
                
                # Check for zero grades
                zero_grades = df[df[field] == 0]
                if len(zero_grades) > 0:
                    validation_result['issues'].append(
                        f"Found {len(zero_grades)} zero grade values ({len(zero_grades)/len(df)*100:.1f}%)"
                    )
                
                # Calculate statistics
                validation_result['statistics'][field] = {
                    'count': df[field].count(),
                    'null_count': df[field].isnull().sum(),
                    'zero_count': (df[field] == 0).sum(),
                    'negative_count': (df[field] < 0).sum(),
                    'min': df[field].min(),
                    'max': df[field].max(),
                    'mean': df[field].mean(),
                    'median': df[field].median(),
                    'std': df[field].std(),
                    'q25': df[field].quantile(0.25),
                    'q75': df[field].quantile(0.75)
                }
            
            logger.info("Grade validation completed")
            return validation_result
            
        except Exception as e:
            logger.error(f"Error validating grades: {e}")
            validation_result['passed'] = False
            validation_result['issues'].append(f"Validation error: {str(e)}")
            return validation_result
    
    def validate_identifiers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate identifier fields"""
        validation_result = {
            'passed': True,
            'issues': [],
            'statistics': {}
        }
        
        try:
            identifier_fields = ['Hole_ID', 'Element', 'DataSet']
            
            for field in identifier_fields:
                if field not in df.columns:
                    validation_result['issues'].append(f"Missing identifier field: {field}")
                    validation_result['passed'] = False
                    continue
                
                # Check for null values
                null_count = df[field].isnull().sum()
                if null_count > 0:
                    validation_result['issues'].append(
                        f"{field} has {null_count} null values ({null_count/len(df)*100:.1f}%)"
                    )
                    validation_result['passed'] = False
                
                # Check for empty strings
                if df[field].dtype == 'object':
                    empty_count = (df[field] == '').sum()
                    if empty_count > 0:
                        validation_result['issues'].append(
                            f"{field} has {empty_count} empty string values"
                        )
                        validation_result['passed'] = False
                
                # Calculate statistics
                validation_result['statistics'][field] = {
                    'count': df[field].count(),
                    'null_count': df[field].isnull().sum(),
                    'unique_count': df[field].nunique(),
                    'most_common': df[field].mode().iloc[0] if len(df[field].mode()) > 0 else None
                }
            
            logger.info("Identifier validation completed")
            return validation_result
            
        except Exception as e:
            logger.error(f"Error validating identifiers: {e}")
            validation_result['passed'] = False
            validation_result['issues'].append(f"Validation error: {str(e)}")
            return validation_result
    
    def validate_data_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate data consistency across records"""
        validation_result = {
            'passed': True,
            'issues': [],
            'statistics': {}
        }
        
        try:
            # Check for duplicate records
            if 'Hole_ID' in df.columns and 'Element' in df.columns:
                duplicates = df.duplicated(subset=['Hole_ID', 'Element', 'from_depth', 'to_depth'])
                duplicate_count = duplicates.sum()
                
                if duplicate_count > 0:
                    validation_result['issues'].append(
                        f"Found {duplicate_count} duplicate records"
                    )
                    validation_result['passed'] = False
                
                validation_result['statistics']['duplicates'] = {
                    'count': duplicate_count,
                    'percentage': duplicate_count / len(df) * 100
                }
            
            # Check for overlapping intervals within holes
            if all(col in df.columns for col in ['Hole_ID', 'from_depth', 'to_depth']):
                overlapping_intervals = self._check_overlapping_intervals(df)
                if overlapping_intervals > 0:
                    validation_result['issues'].append(
                        f"Found {overlapping_intervals} overlapping depth intervals"
                    )
                    validation_result['passed'] = False
                
                validation_result['statistics']['overlapping_intervals'] = overlapping_intervals
            
            # Check for missing coordinate-grade links
            if 'Hole_ID' in df.columns:
                holes_with_grades = set(df['Hole_ID'].unique())
                validation_result['statistics']['data_consistency'] = {
                    'unique_holes': len(holes_with_grades),
                    'total_records': len(df)
                }
            
            logger.info("Data consistency validation completed")
            return validation_result
            
        except Exception as e:
            logger.error(f"Error validating data consistency: {e}")
            validation_result['passed'] = False
            validation_result['issues'].append(f"Validation error: {str(e)}")
            return validation_result
    
    def _check_overlapping_intervals(self, df: pd.DataFrame) -> int:
        """Check for overlapping depth intervals within holes"""
        overlapping_count = 0
        
        for hole_id in df['Hole_ID'].unique():
            hole_data = df[df['Hole_ID'] == hole_id].sort_values('from_depth')
            
            for i in range(len(hole_data) - 1):
                current_to = hole_data.iloc[i]['to_depth']
                next_from = hole_data.iloc[i + 1]['from_depth']
                
                if current_to > next_from:
                    overlapping_count += 1
        
        return overlapping_count
    
    def validate_statistical_properties(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate statistical properties of the data"""
        validation_result = {
            'passed': True,
            'issues': [],
            'statistics': {}
        }
        
        try:
            if 'weighted_grade' in df.columns:
                grades = df['weighted_grade'].dropna()
                
                # Check for statistical anomalies
                q1 = grades.quantile(0.25)
                q3 = grades.quantile(0.75)
                iqr = q3 - q1
                
                # Outliers (beyond 1.5 * IQR)
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                outliers = grades[(grades < lower_bound) | (grades > upper_bound)]
                outlier_count = len(outliers)
                
                if outlier_count > len(grades) * 0.1:  # More than 10% outliers
                    validation_result['issues'].append(
                        f"High number of outliers: {outlier_count} ({outlier_count/len(grades)*100:.1f}%)"
                    )
                
                # Check for extreme skewness
                skewness = grades.skew()
                if abs(skewness) > 3:
                    validation_result['issues'].append(
                        f"Extreme skewness detected: {skewness:.2f}"
                    )
                
                # Check for extremely low variance
                cv = grades.std() / grades.mean()
                if cv < 0.1:
                    validation_result['issues'].append(
                        f"Extremely low coefficient of variation: {cv:.3f}"
                    )
                
                validation_result['statistics']['statistical_properties'] = {
                    'skewness': skewness,
                    'kurtosis': grades.kurtosis(),
                    'coefficient_of_variation': cv,
                    'outlier_count': outlier_count,
                    'outlier_percentage': outlier_count / len(grades) * 100
                }
            
            logger.info("Statistical properties validation completed")
            return validation_result
            
        except Exception as e:
            logger.error(f"Error validating statistical properties: {e}")
            validation_result['passed'] = False
            validation_result['issues'].append(f"Validation error: {str(e)}")
            return validation_result
    
    def run_complete_validation(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run complete data validation suite"""
        validation_report = {
            'timestamp': datetime.now().isoformat(),
            'dataset_info': {
                'rows': len(df),
                'columns': len(df.columns),
                'memory_usage': df.memory_usage(deep=True).sum()
            },
            'validations': {}
        }
        
        # Run all validations
        validation_report['validations']['coordinates'] = self.validate_coordinates(df)
        validation_report['validations']['depths'] = self.validate_depths(df)
        validation_report['validations']['grades'] = self.validate_grades(df)
        validation_report['validations']['identifiers'] = self.validate_identifiers(df)
        validation_report['validations']['consistency'] = self.validate_data_consistency(df)
        validation_report['validations']['statistical'] = self.validate_statistical_properties(df)
        
        # Overall validation status
        overall_passed = all(
            validation['passed'] for validation in validation_report['validations'].values()
        )
        
        validation_report['overall_status'] = 'PASSED' if overall_passed else 'FAILED'
        
        # Count total issues
        total_issues = sum(
            len(validation['issues']) for validation in validation_report['validations'].values()
        )
        validation_report['total_issues'] = total_issues
        
        logger.info(f"Complete validation finished. Status: {validation_report['overall_status']}, Issues: {total_issues}")
        return validation_report

class DataQualityMonitor:
    """Monitors data quality over time"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.validator = DataValidator(config)
        self.quality_history = []
    
    def monitor_quality(self, df: pd.DataFrame, timestamp: str = None) -> Dict[str, Any]:
        """Monitor data quality and store results"""
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        # Run validation
        validation_report = self.validator.run_complete_validation(df)
        validation_report['timestamp'] = timestamp
        
        # Store in history
        self.quality_history.append({
            'timestamp': timestamp,
            'status': validation_report['overall_status'],
            'issues': validation_report['total_issues'],
            'rows': validation_report['dataset_info']['rows']
        })
        
        return validation_report
    
    def get_quality_trends(self) -> Dict[str, Any]:
        """Get quality trends over time"""
        if not self.quality_history:
            return {'message': 'No quality history available'}
        
        df_history = pd.DataFrame(self.quality_history)
        
        return {
            'total_checks': len(self.quality_history),
            'pass_rate': (df_history['status'] == 'PASSED').mean(),
            'avg_issues': df_history['issues'].mean(),
            'latest_status': df_history.iloc[-1]['status'],
            'trend_data': df_history.to_dict('records')
        }

# Global instances
data_validator = DataValidator()
quality_monitor = DataQualityMonitor()
