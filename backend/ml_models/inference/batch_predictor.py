"""
Batch Spatial Ore Grade Predictor

This module handles large-scale batch predictions for ore grades across multiple
spatial points with optimized processing and memory management.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple, Union
import logging
from pathlib import Path
import json
from datetime import datetime
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from ml_models.inference.predictor import SpatialOreGradePredictor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BatchSpatialPredictor:
    """
    Batch processor for large-scale spatial ore grade predictions
    """
    
    def __init__(self, model_path: str, element: str = "CU", 
                 batch_size: int = 1000, n_workers: int = None):
        """
        Initialize batch predictor
        
        Args:
            model_path: Path to trained model
            element: Element to predict
            batch_size: Number of predictions per batch
            n_workers: Number of parallel workers (None = auto)
        """
        self.model_path = model_path
        self.element = element.upper()
        self.batch_size = batch_size
        self.n_workers = n_workers or min(mp.cpu_count(), 4)
        
        # Initialize base predictor
        self.predictor = SpatialOreGradePredictor(model_path, element)
        
        logger.info(f"Batch predictor initialized:")
        logger.info(f"  Element: {self.element}")
        logger.info(f"  Batch size: {self.batch_size}")
        logger.info(f"  Workers: {self.n_workers}")
    
    def predict_from_csv(self, csv_path: str, output_path: str,
                        coordinate_columns: Dict[str, str] = None) -> Dict[str, Any]:
        """
        Predict ore grades from CSV file with coordinates
        
        Args:
            csv_path: Path to CSV file with coordinates
            output_path: Path to save predictions
            coordinate_columns: Column mapping for coordinates
            
        Returns:
            Processing results summary
        """
        try:
            logger.info(f"Starting batch prediction from CSV: {csv_path}")
            start_time = time.time()
            
            # Default column mapping
            if coordinate_columns is None:
                coordinate_columns = {
                    'latitude': 'latitude',
                    'longitude': 'longitude',
                    'depth_from': 'depth_from',
                    'depth_to': 'depth_to'
                }
            
            # Load data
            logger.info("Loading input data...")
            data = pd.read_csv(csv_path)
            total_records = len(data)
            
            logger.info(f"Loaded {total_records} records for prediction")
            
            # Validate required columns
            required_cols = ['latitude', 'longitude']
            missing_cols = [col for col in required_cols if coordinate_columns[col] not in data.columns]
            
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Prepare coordinates
            coordinates = []
            for _, row in data.iterrows():
                coord = {
                    'latitude': row[coordinate_columns['latitude']],
                    'longitude': row[coordinate_columns['longitude']],
                    'depth_from': row.get(coordinate_columns.get('depth_from', 'depth_from'), 0.0),
                    'depth_to': row.get(coordinate_columns.get('depth_to', 'depth_to'), 10.0)
                }
                coordinates.append(coord)
            
            # Process in batches
            logger.info(f"Processing {total_records} predictions in batches of {self.batch_size}")
            
            all_predictions = []
            processed_count = 0
            
            for i in range(0, len(coordinates), self.batch_size):
                batch_coords = coordinates[i:i + self.batch_size]
                batch_num = i // self.batch_size + 1
                total_batches = (len(coordinates) + self.batch_size - 1) // self.batch_size
                
                logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch_coords)} points)")
                
                # Predict batch
                batch_predictions = self._predict_batch_parallel(batch_coords)
                all_predictions.extend(batch_predictions)
                
                processed_count += len(batch_predictions)
                
                # Progress update
                progress = (processed_count / total_records) * 100
                logger.info(f"Progress: {progress:.1f}% ({processed_count}/{total_records})")
            
            # Save results
            logger.info("Saving predictions...")
            self._save_batch_predictions(all_predictions, output_path)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Generate summary
            results_summary = {
                'input_file': csv_path,
                'output_file': output_path,
                'element': self.element,
                'total_records': total_records,
                'processed_records': processed_count,
                'processing_time_seconds': processing_time,
                'predictions_per_second': processed_count / processing_time,
                'batch_size': self.batch_size,
                'n_workers': self.n_workers,
                'timestamp': datetime.now().isoformat()
            }
            
            # Add prediction statistics
            grades = [p['predicted_grade_ppm'] for p in all_predictions]
            results_summary['prediction_stats'] = {
                'min_grade': min(grades),
                'max_grade': max(grades),
                'mean_grade': np.mean(grades),
                'std_grade': np.std(grades),
                'median_grade': np.median(grades)
            }
            
            logger.info(f"Batch prediction completed in {processing_time:.2f} seconds")
            logger.info(f"Processing rate: {processed_count / processing_time:.2f} predictions/second")
            
            return results_summary
            
        except Exception as e:
            logger.error(f"Error in batch prediction from CSV: {str(e)}")
            raise
    
    def predict_spatial_grid(self, bounds: Dict[str, Tuple[float, float]],
                           resolution: float, output_path: str,
                           depth_from: float = 0.0, depth_to: float = 10.0) -> Dict[str, Any]:
        """
        Generate predictions over a spatial grid
        
        Args:
            bounds: Dictionary with 'latitude' and 'longitude' bounds
            resolution: Grid resolution in degrees
            output_path: Path to save grid predictions
            depth_from: Depth from surface
            depth_to: Depth to
            
        Returns:
            Processing results summary
        """
        try:
            logger.info(f"Starting spatial grid prediction for {self.element}")
            start_time = time.time()
            
            # Calculate grid dimensions
            lat_range = bounds['latitude']
            long_range = bounds['longitude']
            
            n_lat = int((lat_range[1] - lat_range[0]) / resolution) + 1
            n_long = int((long_range[1] - long_range[0]) / resolution) + 1
            total_points = n_lat * n_long
            
            logger.info(f"Grid dimensions: {n_lat} x {n_long} = {total_points} points")
            logger.info(f"Grid bounds: Lat {lat_range}, Long {long_range}")
            logger.info(f"Resolution: {resolution} degrees")
            
            # Generate grid coordinates
            logger.info("Generating grid coordinates...")
            coordinates = []
            
            for lat in np.linspace(lat_range[0], lat_range[1], n_lat):
                for long in np.linspace(long_range[0], long_range[1], n_long):
                    coordinates.append({
                        'latitude': lat,
                        'longitude': long,
                        'depth_from': depth_from,
                        'depth_to': depth_to
                    })
            
            # Process grid in batches
            logger.info(f"Processing {total_points} grid points in batches of {self.batch_size}")
            
            all_predictions = []
            processed_count = 0
            
            for i in range(0, len(coordinates), self.batch_size):
                batch_coords = coordinates[i:i + self.batch_size]
                batch_num = i // self.batch_size + 1
                total_batches = (len(coordinates) + self.batch_size - 1) // self.batch_size
                
                logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch_coords)} points)")
                
                # Predict batch
                batch_predictions = self._predict_batch_parallel(batch_coords)
                all_predictions.extend(batch_predictions)
                
                processed_count += len(batch_predictions)
                
                # Progress update
                progress = (processed_count / total_points) * 100
                logger.info(f"Progress: {progress:.1f}% ({processed_count}/{total_points})")
            
            # Save grid predictions
            logger.info("Saving grid predictions...")
            self._save_grid_predictions(all_predictions, output_path, n_lat, n_long)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Generate summary
            results_summary = {
                'prediction_type': 'spatial_grid',
                'element': self.element,
                'grid_dimensions': {'n_lat': n_lat, 'n_long': n_long},
                'total_points': total_points,
                'processed_points': processed_count,
                'resolution': resolution,
                'bounds': bounds,
                'depth_range': {'from': depth_from, 'to': depth_to},
                'processing_time_seconds': processing_time,
                'predictions_per_second': processed_count / processing_time,
                'output_file': output_path,
                'timestamp': datetime.now().isoformat()
            }
            
            # Add prediction statistics
            grades = [p['predicted_grade_ppm'] for p in all_predictions]
            results_summary['prediction_stats'] = {
                'min_grade': min(grades),
                'max_grade': max(grades),
                'mean_grade': np.mean(grades),
                'std_grade': np.std(grades),
                'median_grade': np.median(grades)
            }
            
            logger.info(f"Grid prediction completed in {processing_time:.2f} seconds")
            logger.info(f"Processing rate: {processed_count / processing_time:.2f} predictions/second")
            
            return results_summary
            
        except Exception as e:
            logger.error(f"Error in spatial grid prediction: {str(e)}")
            raise
    
    def _predict_batch_parallel(self, coordinates: List[Dict[str, float]]) -> List[Dict[str, Any]]:
        """
        Predict a batch of coordinates using parallel processing
        
        Args:
            coordinates: List of coordinate dictionaries
            
        Returns:
            List of prediction results
        """
        try:
            if self.n_workers == 1:
                # Single-threaded processing
                return self.predictor.predict_multiple_points(coordinates)
            else:
                # Multi-threaded processing
                chunk_size = max(1, len(coordinates) // self.n_workers)
                chunks = [coordinates[i:i + chunk_size] for i in range(0, len(coordinates), chunk_size)]
                
                all_predictions = []
                
                with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                    future_to_chunk = {
                        executor.submit(self._predict_chunk, chunk): chunk 
                        for chunk in chunks
                    }
                    
                    for future in future_to_chunk:
                        chunk_predictions = future.result()
                        all_predictions.extend(chunk_predictions)
                
                return all_predictions
                
        except Exception as e:
            logger.error(f"Error in parallel batch prediction: {str(e)}")
            raise
    
    def _predict_chunk(self, coordinates: List[Dict[str, float]]) -> List[Dict[str, Any]]:
        """
        Predict a chunk of coordinates (for parallel processing)
        
        Args:
            coordinates: List of coordinate dictionaries
            
        Returns:
            List of prediction results
        """
        try:
            # Create a new predictor instance for this thread
            chunk_predictor = SpatialOreGradePredictor(self.model_path, self.element)
            return chunk_predictor.predict_multiple_points(coordinates)
            
        except Exception as e:
            logger.error(f"Error predicting chunk: {str(e)}")
            raise
    
    def _save_batch_predictions(self, predictions: List[Dict[str, Any]], output_path: str):
        """
        Save batch predictions to file
        
        Args:
            predictions: List of prediction results
            output_path: Path to save predictions
        """
        try:
            # Convert to DataFrame
            data = []
            for pred in predictions:
                data.append({
                    'latitude': pred['coordinates']['latitude'],
                    'longitude': pred['coordinates']['longitude'],
                    'depth_from': pred['coordinates']['depth_from'],
                    'depth_to': pred['coordinates']['depth_to'],
                    'element': pred['element'],
                    'predicted_grade_ppm': pred['predicted_grade_ppm'],
                    'confidence_lower': pred['confidence_interval']['lower_bound'],
                    'confidence_upper': pred['confidence_interval']['upper_bound'],
                    'prediction_timestamp': pred['prediction_metadata']['prediction_timestamp']
                })
            
            df = pd.DataFrame(data)
            
            # Save to CSV
            df.to_csv(output_path, index=False)
            logger.info(f"Batch predictions saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving batch predictions: {str(e)}")
            raise
    
    def _save_grid_predictions(self, predictions: List[Dict[str, Any]], output_path: str,
                             n_lat: int, n_long: int):
        """
        Save grid predictions to file with grid metadata
        
        Args:
            predictions: List of prediction results
            output_path: Path to save predictions
            n_lat: Number of latitude points
            n_long: Number of longitude points
        """
        try:
            # Convert to DataFrame
            data = []
            for pred in predictions:
                data.append({
                    'latitude': pred['coordinates']['latitude'],
                    'longitude': pred['coordinates']['longitude'],
                    'depth_from': pred['coordinates']['depth_from'],
                    'depth_to': pred['coordinates']['depth_to'],
                    'element': pred['element'],
                    'predicted_grade_ppm': pred['predicted_grade_ppm'],
                    'confidence_lower': pred['confidence_interval']['lower_bound'],
                    'confidence_upper': pred['confidence_interval']['upper_bound']
                })
            
            df = pd.DataFrame(data)
            
            # Save grid data
            df.to_csv(output_path, index=False)
            
            # Save grid metadata
            metadata_path = output_path.replace('.csv', '_metadata.json')
            metadata = {
                'grid_dimensions': {'n_lat': n_lat, 'n_long': n_long},
                'total_points': len(predictions),
                'element': self.element,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Grid predictions saved to: {output_path}")
            logger.info(f"Grid metadata saved to: {metadata_path}")
            
        except Exception as e:
            logger.error(f"Error saving grid predictions: {str(e)}")
            raise
    
    def get_processing_summary(self, results: Dict[str, Any]) -> str:
        """
        Get a summary of batch processing results
        
        Args:
            results: Processing results dictionary
            
        Returns:
            Summary string
        """
        try:
            summary_lines = []
            summary_lines.append("BATCH PROCESSING SUMMARY")
            summary_lines.append("=" * 40)
            summary_lines.append(f"Element: {results['element']}")
            summary_lines.append(f"Total Records: {results.get('total_records', results.get('total_points', 'N/A'))}")
            summary_lines.append(f"Processing Time: {results['processing_time_seconds']:.2f} seconds")
            summary_lines.append(f"Processing Rate: {results['predictions_per_second']:.2f} predictions/second")
            
            if 'prediction_stats' in results:
                stats = results['prediction_stats']
                summary_lines.append("")
                summary_lines.append("PREDICTION STATISTICS:")
                summary_lines.append(f"  Min Grade: {stats['min_grade']:.2f} PPM")
                summary_lines.append(f"  Max Grade: {stats['max_grade']:.2f} PPM")
                summary_lines.append(f"  Mean Grade: {stats['mean_grade']:.2f} PPM")
                summary_lines.append(f"  Std Dev: {stats['std_grade']:.2f} PPM")
                summary_lines.append(f"  Median Grade: {stats['median_grade']:.2f} PPM")
            
            return "\n".join(summary_lines)
            
        except Exception as e:
            logger.error(f"Error generating processing summary: {str(e)}")
            return "Error generating summary"


# Example usage
if __name__ == "__main__":
    print("Batch Spatial Ore Grade Predictor")
    print("=" * 50)
    print("Usage examples:")
    print()
    print("1. Batch prediction from CSV:")
    print("   batch_predictor = BatchSpatialPredictor('model.joblib', 'CU')")
    print("   results = batch_predictor.predict_from_csv('coords.csv', 'predictions.csv')")
    print()
    print("2. Grid prediction:")
    print("   bounds = {'latitude': (-23.6, -23.5), 'longitude': (-46.7, -46.6)}")
    print("   results = batch_predictor.predict_spatial_grid(bounds, 0.001, 'grid.csv')")
    print()
    print("Features:")
    print("- Parallel processing for large datasets")
    print("- Memory-efficient batch processing")
    print("- CSV input/output support")
    print("- Spatial grid generation")
    print("- Processing statistics and monitoring")
    print("- Configurable batch sizes and workers")
