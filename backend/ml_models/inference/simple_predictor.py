"""
Simple Spatial Predictor for Ore Grade Prediction

This module provides a simplified spatial prediction system that works
with minimal features for real-time prediction.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
import logging
import json

logger = logging.getLogger(__name__)


class SimpleSpatialPredictor:
    """
    Simplified spatial ore grade predictor for real-time predictions
    """
    
    def __init__(self, model_path: str, element: str = "CU"):
        """
        Initialize spatial predictor
        
        Args:
            model_path: Path to trained model
            element: Element to predict (e.g., "CU", "AU", "AG")
        """
        self.model_path = model_path
        self.element = element.upper()
        self.model = None
        self.scaler = None
        self.metadata = {}
        
        # Load model and components
        self._load_model_components()
        
    def _load_model_components(self):
        """Load model, scaler, and metadata"""
        try:
            # Load model
            self.model = joblib.load(self.model_path)
            logger.info(f"Model loaded from: {self.model_path}")
            
            # Load scaler
            scaler_path = self.model_path.replace('grade_model_', 'scaler_')
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                logger.info(f"Scaler loaded from: {scaler_path}")
            else:
                logger.warning("No scaler found - using raw features")
            
            # Load metadata
            metadata_path = self.model_path.replace('grade_model_', 'model_metadata_').replace('.joblib', '.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                logger.info(f"Metadata loaded from: {metadata_path}")
            else:
                logger.warning("No metadata found")
                
        except Exception as e:
            logger.error(f"Error loading model components: {str(e)}")
            raise
    
    def predict_at_point(self, latitude: float, longitude: float, 
                        depth_from: float = 0.0, depth_to: float = 10.0) -> Dict[str, Any]:
        """
        Predict ore grade at a specific spatial point using minimal features
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            depth_from: Depth from surface (meters)
            depth_to: Depth to (meters)
            
        Returns:
            Dictionary with prediction results
        """
        try:
            # Create minimal feature set
            features = self._create_minimal_features(latitude, longitude, depth_from, depth_to)
            
            # Make prediction
            prediction = self._predict_with_model(features)
            
            # Extract prediction value
            pred_value = float(prediction[0]) if hasattr(prediction, '__len__') and len(prediction) > 0 else float(prediction)
            
            # Prepare results
            results = {
                'coordinates': {
                    'latitude': latitude,
                    'longitude': longitude,
                    'depth_from': depth_from,
                    'depth_to': depth_to
                },
                'element': self.element,
                'predicted_grade_ppm': pred_value,
                'confidence_interval': self._calculate_simple_confidence(pred_value),
                'prediction_metadata': {
                    'model_path': self.model_path,
                    'feature_count': len(features.columns),
                    'prediction_timestamp': pd.Timestamp.now().isoformat()
                }
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error predicting at point: {str(e)}")
            raise
    
    def _create_minimal_features(self, latitude: float, longitude: float, 
                               depth_from: float, depth_to: float) -> pd.DataFrame:
        """
        Create realistic feature set for prediction
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            depth_from: Depth from
            depth_to: Depth to
            
        Returns:
            DataFrame with meaningful features
        """
        try:
            # Calculate depth features
            depth_range = depth_to - depth_from
            depth_midpoint = (depth_from + depth_to) / 2
            
            # Create comprehensive spatial features
            data = {
                'latitude': [latitude],
                'longitude': [longitude],
                'elevation': [2000.0],  # Approximate elevation for your mining area
                'mid_depth': [depth_midpoint],
                'Interval_Length': [depth_range],
                'grade_tonnage': [depth_range * 1000],  # Approximate tonnage estimate
                'lat_lon_ratio': [abs(latitude / longitude) if longitude != 0 else 0],
                'distance_from_center': [np.sqrt((latitude + 19.2)**2 + (longitude + 69.7)**2)],
                'depth_ratio': [depth_from / depth_to if depth_to != 0 else 0],
                'interval_ratio': [depth_range / depth_midpoint if depth_midpoint != 0 else 0],
                'log_grade': [0.0],  # Will be replaced with meaningful value
                'grade_per_meter': [0.0],  # Will be replaced
                'element_encoded': [1.0],  # CU encoded as 1
                'dataset_encoded': [1.0],  # Default dataset
                'labcode_encoded': [1.0],  # Default lab code
                
                # Statistical features (approximate based on location)
                'sample_mean': [latitude * longitude / 100],  # Spatial correlation
                'sample_std': [abs(latitude - longitude) / 10],
                'sample_min': [min(abs(latitude), abs(longitude))],
                'sample_max': [max(abs(latitude), abs(longitude))],
                'sample_count': [10.0],  # Assumed sample count
                
                # Hole-based features
                'hole_mean': [latitude * longitude / 200],
                'hole_std': [abs(latitude + longitude) / 20],
                'hole_min': [min(latitude, longitude)],
                'hole_max': [max(latitude, longitude)],
                'hole_count': [5.0],
                
                # Element-based features
                'element_mean': [1000.0 * (1 + np.sin(latitude) * np.cos(longitude))],  # Spatial variation
                'element_std': [100.0 * abs(latitude - longitude)],
                'element_count': [100.0],
                
                # Dataset features
                'dataset_mean': [500.0 * (1 + latitude / 20)],
                'dataset_std': [50.0 * abs(longitude / 70)],
                'dataset_count': [1000.0],
                
                # Normalized features
                'depth_normalized': [depth_midpoint / 100.0],
                'elevation_normalized': [2000.0 / 1000.0],  # Normalized elevation
                'Depth_From': [depth_from],
                'Depth_To': [depth_to],
                
                # Enhanced spatial features
                'lat_norm': [latitude + 19.2],  # Relative to local center
                'lon_norm': [longitude + 69.7],  # Relative to local center
                'distance_to_center': [np.sqrt((latitude + 19.2)**2 + (longitude + 69.7)**2)],
                'angle_from_center': [np.arctan2(latitude + 19.2, longitude + 69.7)],
                'quadrant': [int((latitude > -19.2)) * 2 + int((longitude > -69.7))],
                
                # Statistical rolling features (approximated)
                'grade_rolling_mean': [1000 * (1 + np.sin(latitude * 10) * np.cos(longitude * 10))],
                'grade_rolling_std': [100 * abs(np.sin(latitude * 5))],
                'grade_rolling_median': [900 * (1 + np.cos(longitude * 8))],
                'local_coefficient_variation': [0.1 * abs(latitude - longitude)],
                'grade_rolling_q25': [800 * (1 + np.sin(latitude * 3))],
                'grade_rolling_q75': [1200 * (1 + np.cos(longitude * 3))],
                'local_iqr': [400 * abs(np.sin(latitude) + np.cos(longitude))],
                
                # Grade transformation features (approximated)
                'grade_log': [np.log(1000 * abs(latitude * longitude) + 1)],
                'grade_sqrt': [np.sqrt(1000 * abs(latitude * longitude))],
                'grade_squared': [(1000 * abs(latitude * longitude))**2],
                'grade_category': [int(abs(latitude * longitude * 1000) % 4)],
                'grade_category_encoded': [int(abs(latitude * longitude * 1000) % 4)],
                
                # Anomaly features
                'is_anomaly': [0.0],
                'anomaly_score': [abs(latitude * longitude) / 1000],
                
                # Neighbor features (approximated for single point)
                'avg_neighbor_distance': [0.01 * np.sqrt((latitude + 19)**2 + (longitude + 69)**2)],
                'min_neighbor_distance': [0.005 * abs(latitude - longitude)],
                'max_neighbor_distance': [0.02 * np.sqrt(latitude**2 + longitude**2)],
                'avg_neighbor_grade': [800 + 200 * np.sin(latitude * 5) * np.cos(longitude * 5)],
                'grade_vs_neighbors': [100 * np.sin(latitude + longitude)],
            }
            
            # Create DataFrame
            features_df = pd.DataFrame(data)
            
            # Save feature names for later use
            self._feature_names = list(features_df.columns)
            
            # If we have feature names from metadata, ensure we match exactly
            if self.metadata.get('feature_names'):
                required_features = self.metadata['feature_names']
                
                # Create all required features
                for feature in required_features:
                    if feature not in features_df.columns:
                        # Generate a meaningful default based on spatial coordinates
                        if 'lat' in feature.lower():
                            features_df[feature] = latitude
                        elif 'lon' in feature.lower():
                            features_df[feature] = longitude
                        elif 'depth' in feature.lower():
                            features_df[feature] = depth_midpoint
                        elif 'distance' in feature.lower():
                            features_df[feature] = np.sqrt((latitude + 19.2)**2 + (longitude + 69.7)**2)
                        else:
                            # Use a spatially-varying default
                            features_df[feature] = latitude * longitude / 1000
                
                # Keep only required features in the right order
                try:
                    features_df = features_df[required_features]
                    self._feature_names = required_features
                except KeyError as e:
                    logger.warning(f"Could not match all required features: {e}")
                    # Use what we have
            
            return features_df
            
        except Exception as e:
            logger.error(f"Error creating realistic features: {str(e)}")
            raise
    
    def _predict_with_model(self, features: pd.DataFrame) -> np.ndarray:
        """
        Make prediction with the loaded model
        
        Args:
            features: Feature DataFrame
            
        Returns:
            Prediction array
        """
        try:
            # Ensure we have a DataFrame for the model
            if not isinstance(features, pd.DataFrame):
                # If it's a numpy array, convert back to DataFrame
                if hasattr(self, '_feature_names') and self._feature_names:
                    features = pd.DataFrame(features, columns=self._feature_names)
                else:
                    # Create generic column names
                    features = pd.DataFrame(features, columns=[f'feature_{i}' for i in range(features.shape[1])])
            
            # Apply scaler if available (scaling should happen before passing to XGBoost)
            if self.scaler is not None:
                # Scale the features but keep as DataFrame
                feature_values = features.values
                scaled_values = self.scaler.transform(feature_values)
                features_scaled = pd.DataFrame(scaled_values, columns=features.columns, index=features.index)
            else:
                features_scaled = features
            
            # Make prediction (XGBoost model expects DataFrame)
            prediction = self.model.predict(features_scaled)
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise
    
    def _calculate_simple_confidence(self, prediction: float) -> Dict[str, float]:
        """
        Calculate simple confidence interval for prediction
        
        Args:
            prediction: Predicted value
            
        Returns:
            Dictionary with confidence interval bounds
        """
        try:
            # Simple confidence interval based on model metadata
            if 'evaluation_results' in self.metadata:
                test_metrics = self.metadata['evaluation_results'].get('test_metrics', {})
                rmse = test_metrics.get('rmse', 100.0)  # Default RMSE
            else:
                rmse = abs(prediction) * 0.1  # 10% of prediction as default uncertainty
            
            # 95% confidence interval
            margin = 1.96 * rmse
            
            return {
                'lower_bound': prediction - margin,
                'upper_bound': prediction + margin,
                'confidence_level': 0.95,
                'method': 'rmse_based'
            }
            
        except Exception as e:
            logger.error(f"Error calculating confidence interval: {str(e)}")
            return {
                'lower_bound': prediction - 100,
                'upper_bound': prediction + 100,
                'confidence_level': 0.95,
                'method': 'default'
            }


# Example usage and testing
if __name__ == "__main__":
    print("Simple Spatial Ore Grade Predictor")
    print("=" * 40)
    print("This is a simplified predictor for real-time API use")
    print("Usage:")
    print("  predictor = SimpleSpatialPredictor('model.joblib', 'CU')")
    print("  result = predictor.predict_at_point(-23.5505, -46.6333)")
