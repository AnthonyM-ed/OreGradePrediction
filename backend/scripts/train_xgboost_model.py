"""
XGBoost Training Script for Ore Grade Prediction
================================================

Main script to train XGBoost model using geological data from SQL Server.
"""

import sys
import os
import logging
from pathlib import Path

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent.parent
sys.path.append(str(backend_dir))

# Django setup
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
import django
django.setup()

from ml_models.data_processing.db_loader import XGBoostGeologicalDataLoader
from ml_models.models.xgboost_predictor import XGBoostOreGradePredictor
import pandas as pd
import numpy as np
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train_xgboost_model(elements=None, datasets=None, optimize_hyperparams=False):
    """
    Train XGBoost model for ore grade prediction using the new view
    
    Args:
        elements: List of chemical elements to include in training
        datasets: List of datasets to include in training
        optimize_hyperparams: Whether to perform hyperparameter optimization
    """
    try:
        # Load configuration
        config_path = backend_dir / 'config' / 'ml_config.json'
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        if elements is None:
            elements = config.get('data_sources', {}).get('elements', ['Cu'])
        
        if datasets is None:
            datasets = config.get('data_sources', {}).get('datasets', ['Drilling_INF', 'Drilling_BF', 'Drilling_OP'])
        
        logger.info(f"Starting XGBoost training for elements: {elements}")
        logger.info(f"Using datasets: {datasets}")
        
        # Initialize data loader and model
        data_loader = XGBoostGeologicalDataLoader()
        model = XGBoostOreGradePredictor(config)
        
        # Load and prepare training data
        logger.info("Loading training data from vw_HoleSamples_ElementGrades...")
        X_train, X_test, y_train, y_test = data_loader.load_xgboost_training_data(
            elements=elements,
            datasets=datasets
        )
        
        logger.info(f"Training data shape: {X_train.shape}")
        logger.info(f"Test data shape: {X_test.shape}")
        
        # Optimize hyperparameters if requested
        if optimize_hyperparams:
            logger.info("Optimizing hyperparameters...")
            optimization_result = model.optimize_hyperparameters(X_train, y_train)
            
            if optimization_result['success']:
                logger.info(f"Best parameters: {optimization_result['best_params']}")
                logger.info(f"Best CV score: {optimization_result['best_score']:.4f}")
            else:
                logger.warning("Hyperparameter optimization failed, using default parameters")
        
        # Train model
        logger.info("Training XGBoost model...")
        training_result = model.train(X_train, y_train, X_test, y_test)
        
        if training_result['success']:
            logger.info("Model training completed successfully!")
            
            # Display training metrics
            metrics = training_result['training_metrics']
            logger.info(f"Training R²: {metrics['train_metrics']['train_r2_score']:.4f}")
            logger.info(f"Validation R²: {metrics['val_metrics']['validation_r2_score']:.4f}")
            logger.info(f"Cross-validation R²: {metrics['cv_scores']['mean']:.4f} ± {metrics['cv_scores']['std']:.4f}")
            
            # Display feature importance
            feature_importance = model.get_feature_importance()
            logger.info("Top 10 most important features:")
            for i, (feature, importance) in enumerate(list(feature_importance.items())[:10]):
                logger.info(f"  {i+1}. {feature}: {importance:.4f}")
            
            # Save model
            model_dir = backend_dir / 'data' / 'models'
            model_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = model_dir / f'xgboost_model_{timestamp}.joblib'
            
            if model.save_model(str(model_path)):
                logger.info(f"Model saved to: {model_path}")
                
                # Also save as latest model
                latest_model_path = model_dir / 'xgboost_model_latest.joblib'
                if model.save_model(str(latest_model_path)):
                    logger.info(f"Latest model saved to: {latest_model_path}")
            
            # Save training report
            report_dir = backend_dir / 'data' / 'reports'
            report_dir.mkdir(parents=True, exist_ok=True)
            
            training_report = {
                'timestamp': datetime.now().isoformat(),
                'elements': elements,
                'data_shape': {
                    'train': X_train.shape,
                    'test': X_test.shape
                },
                'training_metrics': metrics,
                'model_params': model.model_params,
                'feature_importance': feature_importance
            }
            
            report_path = report_dir / f'training_report_{timestamp}.json'
            with open(report_path, 'w') as f:
                json.dump(training_report, f, indent=2, default=str)
            
            logger.info(f"Training report saved to: {report_path}")
            
            return {
                'success': True,
                'model_path': str(model_path),
                'metrics': metrics,
                'feature_importance': feature_importance
            }
            
        else:
            logger.error(f"Model training failed: {training_result.get('error', 'Unknown error')}")
            return {
                'success': False,
                'error': training_result.get('error', 'Unknown error')
            }
    
    except Exception as e:
        logger.error(f"Error in training script: {e}")
        return {
            'success': False,
            'error': str(e)
        }
    
    finally:
        # Clean up
        if 'data_loader' in locals():
            data_loader.close_connection()

def evaluate_model(model_path=None):
    """
    Evaluate a trained XGBoost model
    
    Args:
        model_path: Path to the trained model file
    """
    try:
        if model_path is None:
            model_path = backend_dir / 'data' / 'models' / 'xgboost_model_latest.joblib'
        
        logger.info(f"Evaluating model: {model_path}")
        
        # Load model
        model = XGBoostOreGradePredictor()
        if not model.load_model(str(model_path)):
            logger.error("Failed to load model")
            return False
        
        # Load test data
        data_loader = XGBoostGeologicalDataLoader()
        X_train, X_test, y_train, y_test = data_loader.load_xgboost_training_data(['Cu'])
        
        # Evaluate on test set
        test_metrics = model.evaluate(X_test, y_test)
        
        logger.info("Test Set Evaluation:")
        for metric, value in test_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error evaluating model: {e}")
        return False

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train XGBoost model for ore grade prediction')
    parser.add_argument('--elements', nargs='+', default=['Cu'], 
                       help='Chemical elements to include in training')
    parser.add_argument('--datasets', nargs='+', default=['Drilling_INF', 'Drilling_BF', 'Drilling_OP'],
                       help='Datasets to include in training')
    parser.add_argument('--optimize', action='store_true',
                       help='Perform hyperparameter optimization')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate existing model')
    parser.add_argument('--model-path', type=str,
                       help='Path to model file for evaluation')
    
    args = parser.parse_args()
    
    if args.evaluate:
        evaluate_model(args.model_path)
    else:
        result = train_xgboost_model(args.elements, args.datasets, args.optimize)
        
        if result['success']:
            logger.info("Training completed successfully!")
            print(f"Model saved to: {result['model_path']}")
            print(f"Final R² score: {result['metrics']['cv_scores']['mean']:.4f}")
        else:
            logger.error(f"Training failed: {result['error']}")
            sys.exit(1)

if __name__ == "__main__":
    main()
