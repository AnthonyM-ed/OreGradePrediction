"""
Batch Prediction Script for Spatial Ore Grade Prediction

This script provides command-line interface for batch prediction of ore grades
at multiple spatial coordinates.
"""

import os
import sys
import argparse
import json
from datetime import datetime
from pathlib import Path
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main batch prediction function"""
    parser = argparse.ArgumentParser(description='Batch Spatial Ore Grade Prediction')
    
    # Model parameters
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to the trained model')
    parser.add_argument('--element', type=str, default='CU',
                       help='Element to predict (default: CU)')
    
    # Input parameters
    parser.add_argument('--input-csv', type=str, required=False,
                       help='Path to input CSV file with coordinates')
    parser.add_argument('--output-path', type=str, required=True,
                       help='Path to save predictions')
    
    # Grid parameters (alternative to CSV input)
    parser.add_argument('--grid-mode', action='store_true',
                       help='Generate spatial grid predictions')
    parser.add_argument('--lat-min', type=float, default=-25.0,
                       help='Minimum latitude for grid (default: -25.0)')
    parser.add_argument('--lat-max', type=float, default=-20.0,
                       help='Maximum latitude for grid (default: -20.0)')
    parser.add_argument('--long-min', type=float, default=-50.0,
                       help='Minimum longitude for grid (default: -50.0)')
    parser.add_argument('--long-max', type=float, default=-45.0,
                       help='Maximum longitude for grid (default: -45.0)')
    parser.add_argument('--resolution', type=float, default=0.01,
                       help='Grid resolution in degrees (default: 0.01)')
    
    # Depth parameters
    parser.add_argument('--depth-from', type=float, default=0.0,
                       help='Depth from surface in meters (default: 0.0)')
    parser.add_argument('--depth-to', type=float, default=10.0,
                       help='Depth to in meters (default: 10.0)')
    
    # Processing parameters
    parser.add_argument('--batch-size', type=int, default=1000,
                       help='Batch size for processing (default: 1000)')
    parser.add_argument('--n-workers', type=int, default=None,
                       help='Number of parallel workers (default: auto)')
    
    # Column mapping for CSV input
    parser.add_argument('--lat-column', type=str, default='latitude',
                       help='Latitude column name in CSV (default: latitude)')
    parser.add_argument('--long-column', type=str, default='longitude',
                       help='Longitude column name in CSV (default: longitude)')
    parser.add_argument('--depth-from-column', type=str, default='depth_from',
                       help='Depth from column name in CSV (default: depth_from)')
    parser.add_argument('--depth-to-column', type=str, default='depth_to',
                       help='Depth to column name in CSV (default: depth_to)')
    
    args = parser.parse_args()
    
    try:
        # Import batch predictor
        from ml_models.inference.batch_predictor import BatchSpatialPredictor
        
        logger.info("Starting batch prediction process")
        logger.info(f"Model: {args.model_path}")
        logger.info(f"Element: {args.element}")
        logger.info(f"Output: {args.output_path}")
        
        # Initialize batch predictor
        batch_predictor = BatchSpatialPredictor(
            model_path=args.model_path,
            element=args.element,
            batch_size=args.batch_size,
            n_workers=args.n_workers
        )
        
        # Create output directory if needed
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if args.grid_mode:
            # Grid prediction mode
            logger.info("Grid prediction mode")
            logger.info(f"Latitude range: {args.lat_min} to {args.lat_max}")
            logger.info(f"Longitude range: {args.long_min} to {args.long_max}")
            logger.info(f"Resolution: {args.resolution} degrees")
            
            bounds = {
                'latitude': (args.lat_min, args.lat_max),
                'longitude': (args.long_min, args.long_max)
            }
            
            results = batch_predictor.predict_spatial_grid(
                bounds=bounds,
                resolution=args.resolution,
                output_path=str(output_path),
                depth_from=args.depth_from,
                depth_to=args.depth_to
            )
            
        else:
            # CSV prediction mode
            if not args.input_csv:
                raise ValueError("Input CSV file is required for CSV prediction mode")
            
            logger.info("CSV prediction mode")
            logger.info(f"Input CSV: {args.input_csv}")
            
            # Column mapping
            coordinate_columns = {
                'latitude': args.lat_column,
                'longitude': args.long_column,
                'depth_from': args.depth_from_column,
                'depth_to': args.depth_to_column
            }
            
            results = batch_predictor.predict_from_csv(
                csv_path=args.input_csv,
                output_path=str(output_path),
                coordinate_columns=coordinate_columns
            )
        
        # Save processing results
        results_file = output_path.parent / f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        summary = batch_predictor.get_processing_summary(results)
        print("\n" + "="*60)
        print(summary)
        print("="*60)
        
        logger.info(f"Batch prediction completed successfully!")
        logger.info(f"Results saved to: {results_file}")
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
