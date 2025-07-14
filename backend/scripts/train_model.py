"""
Main Training Script for Ore Grade Prediction Model

This script provides a simple interface to train XGBoost models with comprehensive
evaluation using RMSE, R², and MAE metrics.
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

# Configure Django settings
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train Ore Grade Prediction Model')
    
    # Training parameters
    parser.add_argument('--element', type=str, default='CU', 
                       help='Element to predict (default: CU)')
    parser.add_argument('--dataset', type=str, default='MAIN',
                       help='Dataset to use (default: MAIN)')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of records for testing')
    
    # Model parameters
    parser.add_argument('--model-type', type=str, default='xgboost',
                       choices=['xgboost'], help='Model type (default: xgboost)')
    parser.add_argument('--tune-hyperparameters', action='store_true',
                       help='Perform hyperparameter tuning')
    parser.add_argument('--tuning-method', type=str, default='random',
                       choices=['grid', 'random', 'bayesian', 'progressive'],
                       help='Hyperparameter tuning method (default: random)')
    
    # Evaluation parameters
    parser.add_argument('--cv-folds', type=int, default=5,
                       help='Number of cross-validation folds (default: 5)')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Test set size (default: 0.2)')
    parser.add_argument('--validation-size', type=float, default=0.2,
                       help='Validation set size (default: 0.2)')
    
    # Output parameters
    parser.add_argument('--output-dir', type=str, default='data/models',
                       help='Output directory for models (default: data/models)')
    parser.add_argument('--save-plots', action='store_true',
                       help='Save evaluation plots')
    parser.add_argument('--save-reports', action='store_true',
                       help='Save evaluation reports')
    
    args = parser.parse_args()
    
    try:
        # Import training modules
        from ml_models.training.train_pipeline import TrainingPipeline
        from ml_models.training.hyperparameter_tuning import XGBoostHyperparameterTuner
        from ml_models.training.cross_validation import CrossValidator
        
        logger.info("Starting model training process")
        logger.info(f"Element: {args.element}")
        logger.info(f"Dataset: {args.dataset}")
        logger.info(f"Model type: {args.model_type}")
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize training pipeline
        pipeline = TrainingPipeline()
        
        # Load and prepare data
        logger.info("Loading training data...")
        data = pipeline.load_data(
            element=args.element,
            dataset=args.dataset,
            limit=args.limit
        )
        
        logger.info(f"Data loaded: {len(data)} records")
        
        # Prepare features and target
        logger.info("Preparing features...")
        features, target = pipeline.prepare_features(data)
        
        logger.info(f"Features prepared: {len(features.columns)} features")
        
        # Split and preprocess data
        logger.info("Splitting and preprocessing data...")
        processed_data = pipeline.split_and_preprocess_data(features, target)
        
        # Hyperparameter tuning (if requested)
        best_params = None
        if args.tune_hyperparameters:
            logger.info(f"Performing hyperparameter tuning using {args.tuning_method} method...")
            
            tuner = XGBoostHyperparameterTuner()
            
            if args.tuning_method == 'grid':
                tuning_results = tuner.grid_search_tuning(
                    processed_data['X_train'], 
                    processed_data['y_train'],
                    cv=args.cv_folds
                )
            elif args.tuning_method == 'random':
                tuning_results = tuner.random_search_tuning(
                    processed_data['X_train'], 
                    processed_data['y_train'],
                    cv=args.cv_folds
                )
            elif args.tuning_method == 'bayesian':
                tuning_results = tuner.bayesian_optimization_tuning(
                    processed_data['X_train'], 
                    processed_data['y_train'],
                    cv=args.cv_folds
                )
            elif args.tuning_method == 'progressive':
                tuning_results = tuner.progressive_tuning(
                    processed_data['X_train'], 
                    processed_data['y_train'],
                    cv=args.cv_folds
                )
            
            if tuning_results:
                best_params = tuning_results.get('best_params')
                logger.info(f"Best parameters found: {best_params}")
                
                # Save tuning results
                tuning_file = output_dir / f"tuning_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                tuner.save_tuning_results(str(tuning_file))
                
                # Update pipeline config with best parameters
                if best_params:
                    pipeline.config['model']['hyperparameters'] = best_params
        
        # Train model
        logger.info("Training model...")
        model = pipeline.train_model(processed_data)
        
        # Cross-validation evaluation
        logger.info("Performing cross-validation...")
        cv_evaluator = CrossValidator(cv_folds=args.cv_folds)
        cv_results = cv_evaluator.perform_kfold_cv(
            model,
            processed_data['X_train'],
            processed_data['y_train']
        )
        
        # Print CV results
        logger.info("Cross-validation results:")
        logger.info(f"CV RMSE: {cv_results['test_scores']['rmse']['mean']:.4f} ± {cv_results['test_scores']['rmse']['std']:.4f}")
        logger.info(f"CV R²: {cv_results['test_scores']['r2']['mean']:.4f} ± {cv_results['test_scores']['r2']['std']:.4f}")
        logger.info(f"CV MAE: {cv_results['test_scores']['mae']['mean']:.4f} ± {cv_results['test_scores']['mae']['std']:.4f}")
        
        # Model evaluation
        logger.info("Evaluating model...")
        
        # Update pipeline config for plots and reports
        pipeline.config['evaluation']['save_plots'] = args.save_plots
        pipeline.config['evaluation']['save_reports'] = args.save_reports
        
        evaluation_results = pipeline.evaluate_model(model, processed_data)
        
        # Print evaluation results
        test_metrics = evaluation_results.get('test_metrics', {})
        logger.info("Final test results:")
        logger.info(f"Test RMSE: {test_metrics.get('rmse', 'N/A'):.4f}")
        logger.info(f"Test R²: {test_metrics.get('r2_score', 'N/A'):.4f}")
        logger.info(f"Test MAE: {test_metrics.get('mae', 'N/A'):.4f}")
        logger.info(f"Test MAPE: {test_metrics.get('mape', 'N/A'):.2f}%")
        
        # Save model
        logger.info("Saving model...")
        model_path = pipeline.save_model(model, processed_data, evaluation_results)
        
        # Create comprehensive results
        final_results = {
            'training_parameters': {
                'element': args.element,
                'dataset': args.dataset,
                'model_type': args.model_type,
                'data_records': len(data),
                'features_count': len(features.columns),
                'test_size': args.test_size,
                'validation_size': args.validation_size,
                'cv_folds': args.cv_folds
            },
            'hyperparameter_tuning': {
                'enabled': args.tune_hyperparameters,
                'method': args.tuning_method if args.tune_hyperparameters else None,
                'best_params': best_params
            },
            'cross_validation_results': {
                'rmse_mean': cv_results['test_scores']['rmse']['mean'],
                'rmse_std': cv_results['test_scores']['rmse']['std'],
                'r2_mean': cv_results['test_scores']['r2']['mean'],
                'r2_std': cv_results['test_scores']['r2']['std'],
                'mae_mean': cv_results['test_scores']['mae']['mean'],
                'mae_std': cv_results['test_scores']['mae']['std']
            },
            'test_results': test_metrics,
            'model_path': model_path,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save final results
        results_file = output_dir / f"training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        logger.info(f"Training completed successfully!")
        logger.info(f"Model saved to: {model_path}")
        logger.info(f"Results saved to: {results_file}")
        
        # Print summary
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        print(f"Element: {args.element}")
        print(f"Dataset: {args.dataset}")
        print(f"Data records: {len(data):,}")
        print(f"Features: {len(features.columns)}")
        print(f"Model type: {args.model_type}")
        print(f"Hyperparameter tuning: {'Yes' if args.tune_hyperparameters else 'No'}")
        print()
        print("CROSS-VALIDATION RESULTS:")
        print(f"  RMSE: {cv_results['test_scores']['rmse']['mean']:.4f} ± {cv_results['test_scores']['rmse']['std']:.4f}")
        print(f"  R²:   {cv_results['test_scores']['r2']['mean']:.4f} ± {cv_results['test_scores']['r2']['std']:.4f}")
        print(f"  MAE:  {cv_results['test_scores']['mae']['mean']:.4f} ± {cv_results['test_scores']['mae']['std']:.4f}")
        print()
        print("TEST SET RESULTS:")
        print(f"  RMSE: {test_metrics.get('rmse', 'N/A'):.4f}")
        print(f"  R²:   {test_metrics.get('r2_score', 'N/A'):.4f}")
        print(f"  MAE:  {test_metrics.get('mae', 'N/A'):.4f}")
        print(f"  MAPE: {test_metrics.get('mape', 'N/A'):.2f}%")
        print()
        print(f"Model saved to: {model_path}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
