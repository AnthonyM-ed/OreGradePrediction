"""
Model Evaluation Script for Ore Grade Prediction

This script evaluates trained models and generates comprehensive reports
with RMSE, R², MAE, and geological-specific metrics.
"""

import os
import sys
import argparse
import json
import joblib
from datetime import datetime
from pathlib import Path
import logging
import numpy as np
import pandas as pd

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_model_and_metadata(model_path: str):
    """Load model and its metadata"""
    try:
        # Load model
        model = joblib.load(model_path)
        
        # Try to load metadata
        metadata_path = model_path.replace('.joblib', '_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}
        
        # Try to load scaler
        scaler_path = model_path.replace('grade_model_', 'scaler_')
        scaler = None
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
        
        return model, metadata, scaler
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def load_test_data(element: str = 'CU', dataset: str = 'MAIN', limit: int = None):
    """Load test data for evaluation"""
    try:
        from ml_models.data_processing.db_loader import DatabaseLoader
        from ml_models.data_processing.query_builder import SQLQueryBuilder
        from ml_models.data_processing.feature_engineering import GeologicalFeatureEngineer
        
        db_loader = DatabaseLoader()
        query_builder = SQLQueryBuilder()
        feature_engineer = GeologicalFeatureEngineer()
        
        # Build query for test data
        query = query_builder.build_training_query(
            element=element,
            dataset=dataset,
            limit=limit
        )
        
        # Load data
        data = db_loader.load_data_from_query(query)
        
        # Create features
        feature_data = feature_engineer.create_features(data)
        
        return feature_data
        
    except Exception as e:
        logger.error(f"Error loading test data: {str(e)}")
        raise

def evaluate_model_performance(model, X_test, y_test, model_name: str = "Model"):
    """Evaluate model performance with comprehensive metrics"""
    try:
        from ml_models.models.model_evaluation import ModelEvaluator
        
        evaluator = ModelEvaluator(model_name)
        
        # Comprehensive evaluation
        results = evaluator.evaluate_model_comprehensive(
            model, X_test, y_test
        )
        
        return results, evaluator
        
    except Exception as e:
        logger.error(f"Error evaluating model: {str(e)}")
        raise

def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description='Evaluate Ore Grade Prediction Model')
    
    # Model parameters
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to the trained model')
    parser.add_argument('--model-name', type=str, default=None,
                       help='Name for the model (default: extracted from path)')
    
    # Data parameters
    parser.add_argument('--element', type=str, default='CU',
                       help='Element to evaluate (default: CU)')
    parser.add_argument('--dataset', type=str, default='MAIN',
                       help='Dataset to use (default: MAIN)')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of records for evaluation')
    parser.add_argument('--test-data-path', type=str, default=None,
                       help='Path to test data CSV file (optional)')
    
    # Evaluation parameters
    parser.add_argument('--cv-folds', type=int, default=5,
                       help='Number of cross-validation folds (default: 5)')
    parser.add_argument('--include-cv', action='store_true',
                       help='Include cross-validation in evaluation')
    
    # Output parameters
    parser.add_argument('--output-dir', type=str, default='data/exports',
                       help='Output directory for results (default: data/exports)')
    parser.add_argument('--save-plots', action='store_true',
                       help='Save evaluation plots')
    parser.add_argument('--save-reports', action='store_true',
                       help='Save evaluation reports')
    parser.add_argument('--compare-with', type=str, nargs='+', default=[],
                       help='Paths to other models for comparison')
    
    args = parser.parse_args()
    
    try:
        logger.info("Starting model evaluation")
        logger.info(f"Model path: {args.model_path}")
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model and metadata
        logger.info("Loading model...")
        model, metadata, scaler = load_model_and_metadata(args.model_path)
        
        # Determine model name
        model_name = args.model_name or Path(args.model_path).stem
        
        logger.info(f"Model loaded: {model_name}")
        
        # Load test data
        logger.info("Loading test data...")
        if args.test_data_path:
            # Load from CSV
            test_data = pd.read_csv(args.test_data_path)
        else:
            # Load from database
            test_data = load_test_data(
                element=args.element,
                dataset=args.dataset,
                limit=args.limit
            )
        
        logger.info(f"Test data loaded: {len(test_data)} records")
        
        # Prepare features and target
        target_col = 'standardized_grade_ppm'
        if target_col not in test_data.columns:
            logger.error(f"Target column '{target_col}' not found in test data")
            raise ValueError(f"Target column '{target_col}' not found")
        
        y_test = test_data[target_col]
        X_test = test_data.drop(columns=[target_col])
        
        # Apply scaler if available
        if scaler is not None:
            logger.info("Applying scaler to test data...")
            X_test_scaled = scaler.transform(X_test)
        else:
            X_test_scaled = X_test
        
        # Evaluate model
        logger.info("Evaluating model performance...")
        results, evaluator = evaluate_model_performance(
            model, X_test_scaled, y_test, model_name
        )
        
        # Print results
        test_metrics = results.get('test_metrics', {})
        logger.info("Evaluation results:")
        logger.info(f"RMSE: {test_metrics.get('rmse', 'N/A'):.4f}")
        logger.info(f"R²: {test_metrics.get('r2_score', 'N/A'):.4f}")
        logger.info(f"MAE: {test_metrics.get('mae', 'N/A'):.4f}")
        logger.info(f"MAPE: {test_metrics.get('mape', 'N/A'):.2f}%")
        
        # Cross-validation (if requested)
        if args.include_cv:
            logger.info("Performing cross-validation...")
            from ml_models.training.cross_validation import CrossValidator
            
            cv_evaluator = CrossValidator(cv_folds=args.cv_folds)
            cv_results = cv_evaluator.perform_kfold_cv(model, X_test_scaled, y_test)
            
            logger.info("Cross-validation results:")
            logger.info(f"CV RMSE: {cv_results['test_scores']['rmse']['mean']:.4f} ± {cv_results['test_scores']['rmse']['std']:.4f}")
            logger.info(f"CV R²: {cv_results['test_scores']['r2']['mean']:.4f} ± {cv_results['test_scores']['r2']['std']:.4f}")
            logger.info(f"CV MAE: {cv_results['test_scores']['mae']['mean']:.4f} ± {cv_results['test_scores']['mae']['std']:.4f}")
            
            results['cv_results'] = cv_results
        
        # Generate predictions for plotting
        y_pred = model.predict(X_test_scaled)
        
        # Create plots if requested
        if args.save_plots:
            logger.info("Generating plots...")
            plots_dir = output_dir / "plots"
            plots_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Prediction plot
            evaluator.plot_predictions(
                y_test, y_pred,
                title=f"Model Evaluation - {model_name}",
                save_path=plots_dir / f"predictions_{model_name}_{timestamp}.png"
            )
            
            # Residual plot
            evaluator.plot_residuals(
                y_test, y_pred,
                title=f"Residual Analysis - {model_name}",
                save_path=plots_dir / f"residuals_{model_name}_{timestamp}.png"
            )
        
        # Generate reports if requested
        if args.save_reports:
            logger.info("Generating reports...")
            reports_dir = output_dir / "reports"
            reports_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Generate evaluation report
            report = evaluator.generate_evaluation_report(
                results,
                save_path=reports_dir / f"evaluation_{model_name}_{timestamp}.txt"
            )
            
            print("\n" + "="*60)
            print("EVALUATION REPORT")
            print("="*60)
            print(report)
        
        # Model comparison (if requested)
        if args.compare_with:
            logger.info("Comparing with other models...")
            from ml_models.models.model_evaluation import ModelComparator
            
            comparator = ModelComparator()
            comparator.add_model_results(model_name, results)
            
            # Load and evaluate comparison models
            for comp_model_path in args.compare_with:
                try:
                    comp_model, comp_metadata, comp_scaler = load_model_and_metadata(comp_model_path)
                    comp_name = Path(comp_model_path).stem
                    
                    # Apply scaler if available
                    X_test_comp = X_test_scaled if comp_scaler is None else comp_scaler.transform(X_test)
                    
                    comp_results, _ = evaluate_model_performance(comp_model, X_test_comp, y_test, comp_name)
                    comparator.add_model_results(comp_name, comp_results)
                    
                except Exception as e:
                    logger.warning(f"Could not load comparison model {comp_model_path}: {str(e)}")
            
            # Generate comparison plots
            if args.save_plots:
                for metric in ['rmse', 'r2_score', 'mae']:
                    comparator.plot_model_comparison(
                        metric=metric,
                        save_path=plots_dir / f"comparison_{metric}_{timestamp}.png"
                    )
            
            # Print comparison
            for metric in ['rmse', 'r2_score', 'mae']:
                comp_df = comparator.compare_models(metric)
                if not comp_df.empty:
                    print(f"\n{metric.upper()} COMPARISON:")
                    print(comp_df.to_string(index=False))
        
        # Save results
        results_file = output_dir / f"evaluation_results_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Prepare results for JSON serialization
        serializable_results = {
            'model_name': model_name,
            'model_path': args.model_path,
            'evaluation_parameters': {
                'element': args.element,
                'dataset': args.dataset,
                'test_records': len(test_data),
                'include_cv': args.include_cv,
                'cv_folds': args.cv_folds if args.include_cv else None
            },
            'test_metrics': test_metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        # Add CV results if available
        if args.include_cv and 'cv_results' in results:
            cv_results = results['cv_results']
            serializable_results['cv_metrics'] = {
                'rmse_mean': cv_results['test_scores']['rmse']['mean'],
                'rmse_std': cv_results['test_scores']['rmse']['std'],
                'r2_mean': cv_results['test_scores']['r2']['mean'],
                'r2_std': cv_results['test_scores']['r2']['std'],
                'mae_mean': cv_results['test_scores']['mae']['mean'],
                'mae_std': cv_results['test_scores']['mae']['std']
            }
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Evaluation completed successfully!")
        logger.info(f"Results saved to: {results_file}")
        
        # Print summary
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"Model: {model_name}")
        print(f"Element: {args.element}")
        print(f"Dataset: {args.dataset}")
        print(f"Test records: {len(test_data):,}")
        print()
        print("TEST RESULTS:")
        print(f"  RMSE: {test_metrics.get('rmse', 'N/A'):.4f}")
        print(f"  R²:   {test_metrics.get('r2_score', 'N/A'):.4f}")
        print(f"  MAE:  {test_metrics.get('mae', 'N/A'):.4f}")
        print(f"  MAPE: {test_metrics.get('mape', 'N/A'):.2f}%")
        
        if args.include_cv:
            cv_results = results['cv_results']
            print()
            print("CROSS-VALIDATION RESULTS:")
            print(f"  RMSE: {cv_results['test_scores']['rmse']['mean']:.4f} ± {cv_results['test_scores']['rmse']['std']:.4f}")
            print(f"  R²:   {cv_results['test_scores']['r2']['mean']:.4f} ± {cv_results['test_scores']['r2']['std']:.4f}")
            print(f"  MAE:  {cv_results['test_scores']['mae']['mean']:.4f} ± {cv_results['test_scores']['mae']['std']:.4f}")
        
        print("="*60)
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
