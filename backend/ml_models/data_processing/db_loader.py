import pandas as pd
import numpy as np
import pyodbc
import logging
from typing import Dict, List, Optional, Tuple
from django.conf import settings
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

logger = logging.getLogger(__name__)

class XGBoostGeologicalDataLoader:
    """Optimized data loader for XGBoost ore grade prediction"""
    
    def __init__(self):
        self.connection = None
        self.config = self._load_ml_config()
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
    
    def load_by_lab_code(self, lab_code: str, elements: List[str] = None) -> pd.DataFrame:
        """
        Load data filtered by lab code
        
        Args:
            lab_code: Laboratory code to filter by
            elements: Optional list of elements to include
            
        Returns:
            DataFrame with lab-specific data
        """
        if elements is None:
            elements = self.config.get('data_sources', {}).get('elements', ['Cu'])
        
        try:
            conn = self._get_connection()
            
            elements_str = "', '".join(elements)
            
            query = f"""
                SELECT 
                    SampleID,
                    Hole_ID,
                    Element,
                    DataSet,
                    standardized_grade_ppm,
                    Depth_From,
                    Depth_To,
                    Interval_Length,
                    latitude,
                    longitude,
                    elevation,
                    LabCode
                FROM dbo.vw_HoleSamples_ElementGrades
                WHERE LabCode = '{lab_code}'
                    AND Element IN ('{elements_str}')
                    AND standardized_grade_ppm IS NOT NULL
                    AND standardized_grade_ppm > 0
                ORDER BY Hole_ID, Element, Depth_From
            """
            
            df = pd.read_sql(query, conn)
            
            logger.info(f"Loaded {len(df)} records for lab code {lab_code}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data by lab code: {e}")
            raise
    
    def _load_ml_config(self) -> Dict:
        """Load ML configuration optimized for XGBoost"""
        try:
            config_path = os.path.join(settings.BASE_DIR, 'config', 'ml_config.json')
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading ML config: {e}")
            return {
                "xgboost_params": {
                    "objective": "reg:squarederror",
                    "max_depth": 6,
                    "learning_rate": 0.1,
                    "n_estimators": 100,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "random_state": 42
                }
            }
    
    def _get_connection(self):
        """Obtener conexión a SQL Server"""
        if self.connection is None:
            try:
                config = settings.MSSQL_CONFIG
                connection_string = (
                    f"DRIVER={{{config['driver']}}};"
                    f"SERVER={config['server']},{config['port']};"
                    f"DATABASE={config['database']};"
                    f"UID={config['username']};"
                    f"PWD={config['password']};"
                    f"TrustServerCertificate={config['trust_server_certificate']};"
                    f"Encrypt={config['encrypt']};"
                )
                self.connection = pyodbc.connect(connection_string)
            except Exception as e:
                logger.error(f"Error conectando a SQL Server: {e}")
                raise
        return self.connection
    
    def load_xgboost_training_data(self, elements: List[str] = None, datasets: List[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load and prepare training data optimized for XGBoost from the new view
        
        Args:
            elements: List of chemical elements to include
            datasets: List of datasets to include (Drilling_INF, Drilling_BF, Drilling_OP)
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        if elements is None:
            elements = self.config.get('data_sources', {}).get('elements', ['Cu'])
        
        if datasets is None:
            datasets = self.config.get('data_sources', {}).get('datasets', ['Drilling_INF', 'Drilling_BF', 'Drilling_OP'])
        
        try:
            conn = self._get_connection()
            
            # Enhanced query using the new view with standardized grade in ppm
            query = """
                SELECT 
                    SampleID,
                    Hole_ID,
                    Element,
                    DataSet,
                    standardized_grade_ppm,
                    Depth_From,
                    Depth_To,
                    Interval_Length,
                    latitude,
                    longitude,
                    elevation,
                    LabCode,
                    -- Additional features for XGBoost
                    (Depth_From + Depth_To) / 2.0 as mid_depth,
                    standardized_grade_ppm * Interval_Length as grade_tonnage,
                    CASE 
                        WHEN standardized_grade_ppm > 1000 THEN 'high'  -- >1000 ppm
                        WHEN standardized_grade_ppm > 500 THEN 'medium'  -- 500-1000 ppm
                        ELSE 'low'  -- <500 ppm
                    END as grade_category
                FROM dbo.vw_HoleSamples_ElementGrades
                WHERE Element IN ({})
                    AND DataSet IN ({})
                    AND standardized_grade_ppm IS NOT NULL
                    AND standardized_grade_ppm > 0
                    AND Depth_From IS NOT NULL
                    AND Depth_To IS NOT NULL
                    AND Interval_Length > 0
                    AND latitude IS NOT NULL
                    AND longitude IS NOT NULL
                ORDER BY Hole_ID, Element, Depth_From
            """.format(
                "'" + "', '".join(elements) + "'",
                "'" + "', '".join(datasets) + "'"
            )
            
            df = pd.read_sql(query, conn)
            
            # Feature engineering for XGBoost
            df = self._engineer_xgboost_features(df)
            
            # Prepare features and target
            X, y = self._prepare_xgboost_features(df)
            
            # Split data
            test_size = self.config.get('model_config', {}).get('test_size', 0.2)
            random_state = self.config.get('model_config', {}).get('random_state', 42)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            logger.info(f"XGBoost training data loaded: {len(X_train)} train, {len(X_test)} test samples")
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logger.error(f"Error loading XGBoost training data: {e}")
            raise
    
    def _engineer_xgboost_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features specifically optimized for XGBoost"""
        try:
            # Spatial features
            df['lat_lon_ratio'] = df['latitude'] / (df['longitude'] + 1e-8)
            df['distance_from_center'] = np.sqrt(
                (df['latitude'] - df['latitude'].mean())**2 + 
                (df['longitude'] - df['longitude'].mean())**2
            )
            
            # Depth features
            df['depth_ratio'] = df['mid_depth'] / (df['mid_depth'].max() + 1e-8)
            df['interval_ratio'] = df['Interval_Length'] / (df['Interval_Length'].max() + 1e-8)
            
            # Grade features (using standardized_grade_ppm instead of weighted_grade)
            df['log_grade'] = np.log1p(df['standardized_grade_ppm'])
            df['grade_per_meter'] = df['standardized_grade_ppm'] / (df['Interval_Length'] + 1e-8)
            
            # Categorical features (XGBoost handles these well)
            df['element_encoded'] = self._encode_categorical(df['Element'], 'Element')
            df['dataset_encoded'] = self._encode_categorical(df['DataSet'], 'DataSet')
            df['labcode_encoded'] = self._encode_categorical(df['LabCode'], 'LabCode')
            
            # Sample-level features
            sample_stats = df.groupby('SampleID')['standardized_grade_ppm'].agg([
                'mean', 'std', 'min', 'max', 'count'
            ]).add_prefix('sample_')
            df = df.merge(sample_stats, left_on='SampleID', right_index=True)
            
            # Hole-level statistics
            hole_stats = df.groupby('Hole_ID')['standardized_grade_ppm'].agg([
                'mean', 'std', 'min', 'max', 'count'
            ]).add_prefix('hole_')
            df = df.merge(hole_stats, left_on='Hole_ID', right_index=True)
            
            # Element-level statistics
            element_stats = df.groupby('Element')['standardized_grade_ppm'].agg([
                'mean', 'std', 'count'
            ]).add_prefix('element_')
            df = df.merge(element_stats, left_on='Element', right_index=True)
            
            # Dataset-level statistics
            dataset_stats = df.groupby('DataSet')['standardized_grade_ppm'].agg([
                'mean', 'std', 'count'
            ]).add_prefix('dataset_')
            df = df.merge(dataset_stats, left_on='DataSet', right_index=True)
            
            # Depth-related features
            df['depth_normalized'] = (df['mid_depth'] - df['mid_depth'].min()) / (df['mid_depth'].max() - df['mid_depth'].min() + 1e-8)
            df['elevation_normalized'] = (df['elevation'] - df['elevation'].min()) / (df['elevation'].max() - df['elevation'].min() + 1e-8)
            
            return df
            
        except Exception as e:
            logger.error(f"Error engineering XGBoost features: {e}")
            raise
    
    def _encode_categorical(self, series: pd.Series, column_name: str) -> pd.Series:
        """Encode categorical variables for XGBoost"""
        if column_name not in self.label_encoders:
            self.label_encoders[column_name] = LabelEncoder()
            return self.label_encoders[column_name].fit_transform(series)
        else:
            return self.label_encoders[column_name].transform(series)
    
    def _prepare_xgboost_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target for XGBoost"""
        try:
            # Define feature columns (exclude target and identifiers)
            feature_columns = [
                'latitude', 'longitude', 'elevation', 'mid_depth', 'Interval_Length',
                'grade_tonnage', 'lat_lon_ratio', 'distance_from_center',
                'depth_ratio', 'interval_ratio', 'log_grade', 'grade_per_meter',
                'element_encoded', 'dataset_encoded', 'labcode_encoded',
                'sample_mean', 'sample_std', 'sample_min', 'sample_max', 'sample_count',
                'hole_mean', 'hole_std', 'hole_min', 'hole_max', 'hole_count',
                'element_mean', 'element_std', 'element_count',
                'dataset_mean', 'dataset_std', 'dataset_count',
                'depth_normalized', 'elevation_normalized',
                'Depth_From', 'Depth_To'  # Include original depth columns
            ]
            
            # Filter existing columns
            available_features = [col for col in feature_columns if col in df.columns]
            
            # Features
            X = df[available_features].copy()
            
            # Handle missing values (XGBoost can handle them, but let's fill for consistency)
            X = X.fillna(X.median())
            
            # Target (using standardized_grade_ppm instead of weighted_grade)
            y = df['standardized_grade_ppm']
            
            # Store feature names for later use
            self.feature_names = X.columns.tolist()
            
            logger.info(f"Prepared {len(available_features)} features for XGBoost")
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing XGBoost features: {e}")
            raise
    
    def load_prediction_data(self, elements: List[str] = None, datasets: List[str] = None,
                           spatial_bounds: Dict = None, depth_range: Dict = None) -> pd.DataFrame:
        """Load data for making predictions with trained XGBoost model"""
        if elements is None:
            elements = self.config.get('data_sources', {}).get('elements', ['Cu'])
        
        if datasets is None:
            datasets = self.config.get('data_sources', {}).get('datasets', ['Drilling_INF', 'Drilling_BF', 'Drilling_OP'])
        
        try:
            conn = self._get_connection()
            
            # Base query for prediction using the new view
            query = """
                SELECT 
                    SampleID,
                    Hole_ID,
                    Element,
                    DataSet,
                    Depth_From,
                    Depth_To,
                    Interval_Length,
                    latitude,
                    longitude,
                    elevation,
                    LabCode,
                    (Depth_From + Depth_To) / 2.0 as mid_depth
                FROM dbo.vw_HoleSamples_ElementGrades
                WHERE Element IN ({})
                    AND DataSet IN ({})
                    AND Depth_From IS NOT NULL
                    AND Depth_To IS NOT NULL
                    AND Interval_Length > 0
                    AND latitude IS NOT NULL
                    AND longitude IS NOT NULL
            """.format(
                "'" + "', '".join(elements) + "'",
                "'" + "', '".join(datasets) + "'"
            )
            
            # Add spatial bounds if provided
            if spatial_bounds:
                query += f"""
                    AND latitude BETWEEN {spatial_bounds['min_lat']} AND {spatial_bounds['max_lat']}
                    AND longitude BETWEEN {spatial_bounds['min_lon']} AND {spatial_bounds['max_lon']}
                """
            
            # Add depth range if provided
            if depth_range:
                query += f"""
                    AND Depth_From >= {depth_range['min_depth']}
                    AND Depth_To <= {depth_range['max_depth']}
                """
            
            query += " ORDER BY Hole_ID, Element, Depth_From"
            
            df = pd.read_sql(query, conn)
            
            # Apply same feature engineering as training
            df = self._engineer_prediction_features(df)
            
            logger.info(f"Loaded {len(df)} records for prediction")
            return df
            
        except Exception as e:
            logger.error(f"Error loading prediction data: {e}")
            raise
    
    def _engineer_prediction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for prediction (without target variable)"""
        try:
            # Same spatial features as training
            df['lat_lon_ratio'] = df['latitude'] / (df['longitude'] + 1e-8)
            df['distance_from_center'] = np.sqrt(
                (df['latitude'] - df['latitude'].mean())**2 + 
                (df['longitude'] - df['longitude'].mean())**2
            )
            
            # Depth features
            df['depth_ratio'] = df['mid_depth'] / (df['mid_depth'].max() + 1e-8)
            df['interval_ratio'] = df['Interval_Length'] / (df['Interval_Length'].max() + 1e-8)
            
            # Categorical encoding (using existing encoders)
            if 'Element' in self.label_encoders:
                df['element_encoded'] = self.label_encoders['Element'].transform(df['Element'])
            if 'DataSet' in self.label_encoders:
                df['dataset_encoded'] = self.label_encoders['DataSet'].transform(df['DataSet'])
            if 'LabCode' in self.label_encoders:
                df['labcode_encoded'] = self.label_encoders['LabCode'].transform(df['LabCode'])
            
            # Depth-related features
            df['depth_normalized'] = (df['mid_depth'] - df['mid_depth'].min()) / (df['mid_depth'].max() - df['mid_depth'].min() + 1e-8)
            df['elevation_normalized'] = (df['elevation'] - df['elevation'].min()) / (df['elevation'].max() - df['elevation'].min() + 1e-8)
            
            # Note: For prediction, we can't use grade-based features
            # We'll add placeholders for features that depend on historical data
            
            return df
            
        except Exception as e:
            logger.error(f"Error engineering prediction features: {e}")
            raise
    
    def get_xgboost_params(self) -> Dict:
        """Get XGBoost parameters from configuration"""
        return self.config.get('xgboost_params', {
            'objective': 'reg:squarederror',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        })
    def load_drilling_data(self, element: str = 'Cu', dataset: str = None) -> pd.DataFrame:
        """
        Load drilling data for a specific element (legacy method updated to use new view)
        
        Args:
            element: Chemical element to query (Cu, Au, etc.)
            dataset: Dataset to filter by (optional)
            
        Returns:
            DataFrame with drilling data
        """
        try:
            conn = self._get_connection()
            
            query = """
                SELECT 
                    SampleID,
                    Hole_ID,
                    Element,
                    DataSet,
                    standardized_grade_ppm,
                    Depth_From,
                    Depth_To,
                    Interval_Length,
                    latitude,
                    longitude,
                    elevation,
                    LabCode
                FROM dbo.vw_HoleSamples_ElementGrades
                WHERE Element = ?
                    AND standardized_grade_ppm IS NOT NULL
                    AND standardized_grade_ppm > 0
            """
            
            params = [element]
            
            if dataset:
                query += " AND DataSet = ?"
                params.append(dataset)
            
            query += " ORDER BY Hole_ID, Depth_From"
            
            df = pd.read_sql(query, conn, params=params)
            
            logger.info(f"Loaded {len(df)} records for element {element}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading drilling data: {e}")
            raise
    
    def load_multi_element_data(self, elements: List[str] = None, datasets: List[str] = None) -> pd.DataFrame:
        """
        Load multi-element data optimized for XGBoost using the new view
        
        Args:
            elements: List of chemical elements
            datasets: List of datasets to include
            
        Returns:
            DataFrame with multi-element data
        """
        if elements is None:
            elements = self.config.get('data_sources', {}).get('elements', ['Cu'])
        
        if datasets is None:
            datasets = self.config.get('data_sources', {}).get('datasets', ['Drilling_INF', 'Drilling_BF', 'Drilling_OP'])
        
        try:
            conn = self._get_connection()
            
            elements_str = "', '".join(elements)
            datasets_str = "', '".join(datasets)
            
            query = f"""
                SELECT 
                    SampleID,
                    Hole_ID,
                    Element,
                    DataSet,
                    standardized_grade_ppm,
                    Depth_From,
                    Depth_To,
                    Interval_Length,
                    latitude,
                    longitude,
                    elevation,
                    LabCode,
                    (Depth_From + Depth_To) / 2.0 as mid_depth
                FROM dbo.vw_HoleSamples_ElementGrades
                WHERE Element IN ('{elements_str}')
                    AND DataSet IN ('{datasets_str}')
                    AND standardized_grade_ppm IS NOT NULL
                    AND standardized_grade_ppm > 0
                ORDER BY Hole_ID, Element, Depth_From
            """
            
            df = pd.read_sql(query, conn)
            
            logger.info(f"Loaded {len(df)} records for {len(elements)} elements across {len(datasets)} datasets")
            return df
            
        except Exception as e:
            logger.error(f"Error loading multi-element data: {e}")
            raise
    
    def load_spatial_context(self, center_lat: float, center_lon: float, 
                           radius_km: float = 10.0, elements: List[str] = None, 
                           datasets: List[str] = None) -> pd.DataFrame:
        """
        Load data within a specific radius from a central point using the new view
        
        Args:
            center_lat: Central latitude
            center_lon: Central longitude
            radius_km: Radius in kilometers
            elements: List of elements to include
            datasets: List of datasets to include
            
        Returns:
            DataFrame with data in the specified area
        """
        if elements is None:
            elements = self.config.get('data_sources', {}).get('elements', ['Cu'])
        
        if datasets is None:
            datasets = self.config.get('data_sources', {}).get('datasets', ['Drilling_INF', 'Drilling_BF', 'Drilling_OP'])
        
        try:
            conn = self._get_connection()
            
            # Approximate conversion from km to degrees (1 degree ≈ 111 km)
            radius_deg = radius_km / 111.0
            
            elements_str = "', '".join(elements)
            datasets_str = "', '".join(datasets)
            
            query = f"""
                SELECT 
                    SampleID,
                    Hole_ID,
                    Element,
                    DataSet,
                    standardized_grade_ppm,
                    Depth_From,
                    Depth_To,
                    Interval_Length,
                    latitude,
                    longitude,
                    elevation,
                    LabCode,
                    (Depth_From + Depth_To) / 2.0 as mid_depth,
                    -- Calculate distance from center
                    SQRT(POWER(latitude - {center_lat}, 2) + POWER(longitude - {center_lon}, 2)) as distance_from_center
                FROM dbo.vw_HoleSamples_ElementGrades
                WHERE Element IN ('{elements_str}')
                    AND DataSet IN ('{datasets_str}')
                    AND standardized_grade_ppm IS NOT NULL
                    AND standardized_grade_ppm > 0
                    AND latitude BETWEEN {center_lat - radius_deg} AND {center_lat + radius_deg}
                    AND longitude BETWEEN {center_lon - radius_deg} AND {center_lon + radius_deg}
                    AND SQRT(POWER(latitude - {center_lat}, 2) + POWER(longitude - {center_lon}, 2)) <= {radius_deg}
                ORDER BY distance_from_center, Hole_ID, Element, Depth_From
            """
            
            df = pd.read_sql(query, conn)
            
            logger.info(f"Loaded {len(df)} records within {radius_km}km of ({center_lat}, {center_lon})")
            return df
            
        except Exception as e:
            logger.error(f"Error loading spatial context data: {e}")
            raise
    
    def get_data_summary(self) -> Dict:
        """
        Get summary of available data using the new view
        
        Returns:
            Dictionary with data statistics
        """
        try:
            conn = self._get_connection()
            
            # Count records by element
            query_elements = """
                SELECT 
                    Element,
                    COUNT(*) as count,
                    AVG(standardized_grade_ppm) as avg_grade,
                    MIN(standardized_grade_ppm) as min_grade,
                    MAX(standardized_grade_ppm) as max_grade,
                    STDEV(standardized_grade_ppm) as std_grade
                FROM dbo.vw_HoleSamples_ElementGrades
                WHERE standardized_grade_ppm IS NOT NULL
                GROUP BY Element
                ORDER BY count DESC
            """
            
            df_elements = pd.read_sql(query_elements, conn)
            
            # Count unique holes and samples
            query_stats = """
                SELECT 
                    COUNT(DISTINCT Hole_ID) as unique_holes,
                    COUNT(DISTINCT SampleID) as unique_samples,
                    COUNT(*) as total_records
                FROM dbo.vw_HoleSamples_ElementGrades
                WHERE standardized_grade_ppm IS NOT NULL
            """
            
            stats = pd.read_sql(query_stats, conn).iloc[0]
            
            # Count records by dataset
            query_datasets = """
                SELECT 
                    DataSet,
                    COUNT(*) as count,
                    COUNT(DISTINCT Hole_ID) as unique_holes,
                    COUNT(DISTINCT Element) as unique_elements
                FROM dbo.vw_HoleSamples_ElementGrades
                WHERE standardized_grade_ppm IS NOT NULL
                GROUP BY DataSet
                ORDER BY count DESC
            """
            
            df_datasets = pd.read_sql(query_datasets, conn)
            
            # Count by lab code
            query_labs = """
                SELECT 
                    LabCode,
                    COUNT(*) as count,
                    COUNT(DISTINCT Element) as unique_elements
                FROM dbo.vw_HoleSamples_ElementGrades
                WHERE standardized_grade_ppm IS NOT NULL
                GROUP BY LabCode
                ORDER BY count DESC
            """
            
            df_labs = pd.read_sql(query_labs, conn)
            
            summary = {
                'total_records': int(stats['total_records']),
                'unique_holes': int(stats['unique_holes']),
                'unique_samples': int(stats['unique_samples']),
                'elements': df_elements.to_dict('records'),
                'datasets': df_datasets.to_dict('records'),
                'lab_codes': df_labs.to_dict('records')
            }
            
            logger.info(f"Data summary: {summary['total_records']} records, {summary['unique_holes']} holes, {summary['unique_samples']} samples")
            return summary
            
        except Exception as e:
            logger.error(f"Error getting data summary: {e}")
            raise
    
    def close_connection(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            self.connection = None
    
    def __del__(self):
        """Destructor to close connection"""
        self.close_connection()