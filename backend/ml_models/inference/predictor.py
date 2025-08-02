"""
Spatial Ore Grade Predictor

This module provides spatial prediction capabilities for ore grades at specific coordinates.
Input: latitude, longitude, depth (optional), element
Output: predicted grade in PPM
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union, List, Tuple
import logging
from pathlib import Path
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from ml_models.data_processing.feature_engineering import GeologicalFeatureEngineer
from ml_models.data_processing.preprocessing import GeologicalDataPreprocessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpatialOreGradePredictor:
    """
    Spatial ore grade predictor for point-based predictions
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
        self.feature_engineer = GeologicalFeatureEngineer()
        
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
            metadata_path = self.model_path.replace('.joblib', '_metadata.json')
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
                        depth_from: float = 0.0, depth_to: float = 10.0,
                        additional_features: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Predict ore grade at a specific spatial point
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            depth_from: Depth from surface (meters)
            depth_to: Depth to (meters)
            additional_features: Additional features for prediction
            
        Returns:
            Dictionary with prediction results
        """
        try:
            # Create input data
            input_data = {
                'latitude': latitude,
                'longitude': longitude,
                'depth_from': depth_from,
                'depth_to': depth_to,
                'element': self.element
            }
            
            # Add additional features if provided
            if additional_features:
                input_data.update(additional_features)
            
            # Convert to DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Feature engineering
            features = self._prepare_features(input_df)
            
            # Make prediction
            prediction = self._predict_with_model(features)
            
            # Prepare results
            results = {
                'coordinates': {
                    'latitude': latitude,
                    'longitude': longitude,
                    'depth_from': depth_from,
                    'depth_to': depth_to
                },
                'element': self.element,
                'predicted_grade_ppm': float(prediction[0]),
                'confidence_interval': self._calculate_confidence_interval(features),
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
    
    def predict_multiple_points(self, coordinates: List[Dict[str, float]], 
                              additional_features: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """
        Predict ore grades at multiple spatial points
        
        Args:
            coordinates: List of coordinate dictionaries with lat, long, depth_from, depth_to
            additional_features: List of additional features for each point
            
        Returns:
            List of prediction results
        """
        try:
            results = []
            
            for i, coord in enumerate(coordinates):
                # Get additional features for this point
                add_features = additional_features[i] if additional_features else None
                
                # Predict at this point
                result = self.predict_at_point(
                    latitude=coord['latitude'],
                    longitude=coord['longitude'],
                    depth_from=coord.get('depth_from', 0.0),
                    depth_to=coord.get('depth_to', 10.0),
                    additional_features=add_features
                )
                
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error predicting multiple points: {str(e)}")
            raise
    
    def predict_grid(self, lat_range: Tuple[float, float], long_range: Tuple[float, float],
                    grid_resolution: float = 0.001, depth_from: float = 0.0, 
                    depth_to: float = 10.0) -> pd.DataFrame:
        """
        Predict ore grades over a spatial grid
        
        Args:
            lat_range: Tuple of (min_lat, max_lat)
            long_range: Tuple of (min_long, max_long)
            grid_resolution: Grid resolution in degrees
            depth_from: Depth from surface
            depth_to: Depth to
            
        Returns:
            DataFrame with grid predictions
        """
        try:
            logger.info(f"Generating grid predictions for {self.element}")
            
            # Create grid coordinates
            lats = np.arange(lat_range[0], lat_range[1], grid_resolution)
            longs = np.arange(long_range[0], long_range[1], grid_resolution)
            
            # Create coordinate pairs
            coordinates = []
            for lat in lats:
                for long in longs:
                    coordinates.append({
                        'latitude': lat,
                        'longitude': long,
                        'depth_from': depth_from,
                        'depth_to': depth_to
                    })
            
            logger.info(f"Predicting for {len(coordinates)} grid points")
            
            # Predict for all points
            predictions = self.predict_multiple_points(coordinates)
            
            # Convert to DataFrame
            grid_data = []
            for pred in predictions:
                grid_data.append({
                    'latitude': pred['coordinates']['latitude'],
                    'longitude': pred['coordinates']['longitude'],
                    'depth_from': pred['coordinates']['depth_from'],
                    'depth_to': pred['coordinates']['depth_to'],
                    'element': pred['element'],
                    'predicted_grade_ppm': pred['predicted_grade_ppm']
                })
            
            grid_df = pd.DataFrame(grid_data)
            
            logger.info(f"Grid prediction completed: {len(grid_df)} predictions")
            
            return grid_df
            
        except Exception as e:
            logger.error(f"Error generating grid predictions: {str(e)}")
            raise
    
    def _prepare_features(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for prediction
        
        Args:
            input_df: Input DataFrame with coordinates and element
            
        Returns:
            Feature DataFrame ready for prediction
        """
        try:
            # Add a dummy target column for feature engineering
            if 'standardized_grade_ppm' not in input_df.columns:
                input_df['standardized_grade_ppm'] = 0.0  # Dummy value
            
            # Use feature engineer to create features
            features = self.feature_engineer.create_all_features(input_df, include_neighbors=False)
            
            # Remove target column if present
            if 'standardized_grade_ppm' in features.columns:
                features = features.drop(columns=['standardized_grade_ppm'])
            
            # Ensure we have all required features
            if hasattr(self.metadata, 'feature_names') and self.metadata.get('feature_names'):
                required_features = self.metadata['feature_names']
                
                # Add missing features with default values
                for feature in required_features:
                    if feature not in features.columns:
                        features[feature] = 0.0
                
                # Reorder features to match training order
                features = features[required_features]
            
            return features
            
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
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
            # Apply scaler if available
            if self.scaler is not None:
                features_scaled = self.scaler.transform(features)
            else:
                features_scaled = features
            
            # Make prediction
            prediction = self.model.predict(features_scaled)
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise
    
    def _calculate_confidence_interval(self, features: pd.DataFrame, 
                                     confidence_level: float = 0.95) -> Dict[str, float]:
        """
        Calculate confidence interval for prediction
        
        Args:
            features: Feature DataFrame
            confidence_level: Confidence level (0.95 = 95%)
            
        Returns:
            Dictionary with confidence interval bounds
        """
        try:
            # This is a simplified confidence interval calculation
            # In practice, you might want to use more sophisticated methods
            
            # For now, we'll use a simple approach based on model metadata
            if 'evaluation_results' in self.metadata:
                test_metrics = self.metadata['evaluation_results'].get('test_metrics', {})
                rmse = test_metrics.get('rmse', 100.0)  # Default RMSE
                
                # Simple confidence interval using RMSE
                z_score = 1.96 if confidence_level == 0.95 else 2.576  # 95% or 99%
                margin_of_error = z_score * rmse
                
                return {
                    'lower_bound': -margin_of_error,
                    'upper_bound': margin_of_error,
                    'confidence_level': confidence_level,
                    'method': 'rmse_based'
                }
            
            # Default confidence interval
            return {
                'lower_bound': -200.0,
                'upper_bound': 200.0,
                'confidence_level': confidence_level,
                'method': 'default'
            }
            
        except Exception as e:
            logger.error(f"Error calculating confidence interval: {str(e)}")
            return {'lower_bound': 0.0, 'upper_bound': 0.0, 'confidence_level': 0.0, 'method': 'error'}
    
    def get_prediction_summary(self, results: Union[Dict[str, Any], List[Dict[str, Any]]]) -> str:
        """
        Get a summary of prediction results
        
        Args:
            results: Single prediction result or list of results
            
        Returns:
            Summary string
        """
        try:
            if isinstance(results, dict):
                # Single prediction
                return f"""
SPATIAL PREDICTION SUMMARY
========================
Element: {results['element']}
Coordinates: ({results['coordinates']['latitude']:.6f}, {results['coordinates']['longitude']:.6f})
Depth: {results['coordinates']['depth_from']:.1f} - {results['coordinates']['depth_to']:.1f} m
Predicted Grade: {results['predicted_grade_ppm']:.2f} PPM
Confidence Interval: [{results['confidence_interval']['lower_bound']:.2f}, {results['confidence_interval']['upper_bound']:.2f}] PPM
"""
            else:
                # Multiple predictions
                summary_lines = []
                summary_lines.append("SPATIAL PREDICTIONS SUMMARY")
                summary_lines.append("=" * 40)
                summary_lines.append(f"Element: {results[0]['element']}")
                summary_lines.append(f"Total Predictions: {len(results)}")
                
                # Calculate statistics
                grades = [r['predicted_grade_ppm'] for r in results]
                summary_lines.append(f"Grade Range: {min(grades):.2f} - {max(grades):.2f} PPM")
                summary_lines.append(f"Mean Grade: {np.mean(grades):.2f} PPM")
                summary_lines.append(f"Std Dev: {np.std(grades):.2f} PPM")
                
                return "\n".join(summary_lines)
                
        except Exception as e:
            logger.error(f"Error generating prediction summary: {str(e)}")
            return "Error generating summary"
    
    def save_predictions(self, results: Union[Dict[str, Any], List[Dict[str, Any]]], 
                        filepath: str):
        """
        Save prediction results to file
        
        Args:
            results: Prediction results
            filepath: Path to save file
        """
        try:
            # Convert to list if single result
            if isinstance(results, dict):
                results = [results]
            
            # Convert to DataFrame
            data = []
            for result in results:
                data.append({
                    'latitude': result['coordinates']['latitude'],
                    'longitude': result['coordinates']['longitude'],
                    'depth_from': result['coordinates']['depth_from'],
                    'depth_to': result['coordinates']['depth_to'],
                    'element': result['element'],
                    'predicted_grade_ppm': result['predicted_grade_ppm'],
                    'confidence_lower': result['confidence_interval']['lower_bound'],
                    'confidence_upper': result['confidence_interval']['upper_bound'],
                    'prediction_timestamp': result['prediction_metadata']['prediction_timestamp']
                })
            
            df = pd.DataFrame(data)
            
            # Save based on file extension
            if filepath.endswith('.csv'):
                df.to_csv(filepath, index=False)
            elif filepath.endswith('.json'):
                df.to_json(filepath, orient='records', indent=2)
            else:
                # Default to CSV
                df.to_csv(filepath + '.csv', index=False)
            
            logger.info(f"Predictions saved to: {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving predictions: {str(e)}")
            raise


# Example usage
if __name__ == "__main__":
    # Example usage of the spatial predictor
    print("Spatial Ore Grade Predictor")
    print("=" * 40)
    print("Usage examples:")
    print("1. Single point prediction:")
    print("   predictor = SpatialOreGradePredictor('model.joblib', 'CU')")
    print("   result = predictor.predict_at_point(-23.5505, -46.6333)")
    print()
    print("2. Multiple points prediction:")
    print("   coords = [{'latitude': -23.5505, 'longitude': -46.6333}]")
    print("   results = predictor.predict_multiple_points(coords)")
    print()
    print("3. Grid prediction:")
    print("   grid = predictor.predict_grid((-23.6, -23.5), (-46.7, -46.6))")
    print()
    print("Features:")
    print("- Point-based spatial prediction")
    print("- Batch prediction for multiple points")
    print("- Grid-based prediction for spatial analysis")
    print("- Confidence intervals")
    print("- Export capabilities (CSV, JSON)")
