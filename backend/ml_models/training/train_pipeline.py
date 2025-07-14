"""
Training Pipeline for Ore Grade Prediction Models

This module orchestrates the complete training pipeline including data loading,
preprocessing, model training, and evaluation.
"""

import os
import sys
import json     
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List
import logging
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Import ML modules (Django setup is handled automatically by ml_models.__init__)
from ml_models.data_processing.db_loader import XGBoostGeologicalDataLoader
from ml_models.data_processing.query_builder import SQLQueryBuilder
from ml_models.data_processing.feature_engineering import GeologicalFeatureEngineer
from ml_models.data_processing.preprocessing import GeologicalDataPreprocessor
from ml_models.models.xgboost_predictor import XGBoostOreGradePredictor
from ml_models.models.model_evaluation import ModelEvaluator
from ml_models.utils.logging_config import setup_logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainingPipeline:
    """
    Complete training pipeline for ore grade prediction models
    """
    
    def __init__(self, config_path: str = "config/ml_config.json"):
        """
        Initialize training pipeline
        
        Args:
            config_path: Path to ML configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.pipeline_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize components
        self.db_loader = XGBoostGeologicalDataLoader()
        self.query_builder = SQLQueryBuilder()
        self.feature_engineer = GeologicalFeatureEngineer()
        self.preprocessor = GeologicalDataPreprocessor()
        self.model = None
        self.evaluator = None
        
        # Training results
        self.training_results = {}
        self.model_path = None
        
    def _load_config(self) -> Dict[str, Any]:
        """Load ML configuration"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "data_source": {
                "view_name": "vw_HoleSamples_ElementGrades",
                "target_column": "standardized_grade_ppm",
                "feature_columns": ["latitude", "longitude", "depth_from", "depth_to"]
            },
            "model": {
                "type": "xgboost",
                "hyperparameters": {
                    "n_estimators": 100,
                    "max_depth": 6,
                    "learning_rate": 0.1,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8
                }
            },
            "training": {
                "test_size": 0.2,
                "validation_size": 0.2,
                "random_state": 42,
                "cv_folds": 5
            },
            "preprocessing": {
                "scaling": {
                    "method": "standard"
                }
            },
            "evaluation": {
                "metrics": ["rmse", "r2_score", "mae"],
                "save_plots": True,
                "save_reports": True
            }
        }
    
    def load_data(self, element: str = "CU", dataset: str = "MAIN", 
                  limit: Optional[int] = None) -> pd.DataFrame:
        """
        Load training data from database
        
        Args:
            element: Element to predict (default: CU)
            dataset: Dataset to use (default: MAIN)
            limit: Limit number of records (for testing)
            
        Returns:
            DataFrame with training data
        """
        try:
            logger.info(f"Loading training data for element: {element}, dataset: {dataset}")
            
            # Load data using XGBoost data loader
            X_train, X_test, y_train, y_test = self.db_loader.load_xgboost_training_data(
                elements=[element]
            )
            
            # Combine features and target into single dataframe for processing
            data = X_train.copy()
            data['standardized_grade_ppm'] = y_train
            
            # Apply limit if specified
            if limit and len(data) > limit:
                data = data.head(limit)
                logger.info(f"Limited data to {limit} records")
            
            if data.empty:
                raise ValueError("No data loaded from database")
            
            logger.info(f"Loaded {len(data)} records for training")
            
            return data
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def prepare_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare features and target from raw data
        
        Args:
            data: Raw data from database
            
        Returns:
            Tuple of (features, target)
        """
        try:
            logger.info("Preparing features and target")
            
            # Feature engineering
            feature_data = self.feature_engineer.create_all_features(data)

            # Convert categorical columns to codes to avoid XGBoost errors
            for col in feature_data.select_dtypes(include='category').columns:
                feature_data[col] = feature_data[col].cat.codes
                logger.info(f"Converted categorical column '{col}' to numeric codes")
            
            # Extract target column
            target_col = "standardized_grade_ppm"
            
            if target_col not in feature_data.columns:
                raise ValueError(f"Target column '{target_col}' not found in data")
            
            # Separate features and target
            target = feature_data[target_col].copy()
            features = feature_data.drop(columns=[target_col])
            
            logger.info(f"Prepared {len(features.columns)} features for {len(features)} samples")
            
            return features, target
            
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            raise
    
    def split_and_preprocess_data(self, features: pd.DataFrame, target: pd.DataFrame) -> Dict[str, Any]:
        """
        Split data and apply preprocessing
        
        Args:
            features: Feature matrix
            target: Target values
            
        Returns:
            Dictionary with split and preprocessed data
        """
        try:
            logger.info("Splitting and preprocessing data")
            
            # Split data using sklearn
            from sklearn.model_selection import train_test_split
            
            # First split: separate test set
            X_temp, X_test, y_temp, y_test = train_test_split(
                features, 
                target,
                test_size=self.config["model_config"]["test_size"],
                random_state=self.config["model_config"]["random_state"]
            )
            
            # Second split: separate training and validation sets
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, 
                y_temp,
                test_size=0.2,  # Default validation size
                random_state=self.config["model_config"]["random_state"]
            )
            
            # Apply preprocessing
            X_train_processed = self.preprocessor.handle_missing_values(X_train)
            X_val_processed = self.preprocessor.handle_missing_values(X_val)
            X_test_processed = self.preprocessor.handle_missing_values(X_test)
            
            # Scale features
            scaling_method = self.config["preprocessing"].get("scaling", "standard")
            X_train_processed = self.preprocessor.scale_features(
                X_train_processed, 
                method=scaling_method,
                fit=True
            )
            
            X_val_processed = self.preprocessor.scale_features(
                X_val_processed, 
                method=scaling_method,
                fit=False
            )
            
            X_test_processed = self.preprocessor.scale_features(
                X_test_processed, 
                method=scaling_method,
                fit=False
            )
            
            # Store scaler for later use (if available)
            scaler = getattr(self.preprocessor, 'scaler', None)
            
            # Store scaler for later use
            self.scaler = scaler
            
            processed_data = {
                "X_train": X_train_processed,
                "X_val": X_val_processed,
                "X_test": X_test_processed,
                "y_train": y_train,
                "y_val": y_val,
                "y_test": y_test,
                "scaler": scaler
            }
            
            logger.info(f"Data split - Train: {len(X_train_processed)}, Val: {len(X_val_processed)}, Test: {len(X_test_processed)}")
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Error in data splitting/preprocessing: {str(e)}")
            raise
    
    def train_model(self, processed_data: Dict[str, Any]) -> Any:
        """
        Train the model
        
        Args:
            processed_data: Preprocessed training data
            
        Returns:
            Trained model
        """
        try:
            logger.info("Training model")
            
            # Initialize model
            algorithm = self.config["model_config"]["algorithm"]
            if algorithm == "xgboost":
                self.model = XGBoostOreGradePredictor(config=self.config["xgboost_params"])
            else:
                raise ValueError(f"Unsupported model type: {algorithm}")
            
            # Train model
            self.model.train(
                processed_data["X_train"], 
                processed_data["y_train"],
                X_val=processed_data["X_val"],
                y_val=processed_data["y_val"]
            )
            
            logger.info("Model training completed")
            
            return self.model
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise
    
    def evaluate_model(self, model: Any, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate the trained model
        
        Args:
            model: Trained model
            processed_data: Preprocessed data
            
        Returns:
            Evaluation results
        """
        try:
            logger.info("Evaluating model")
            
            # Initialize evaluator
            self.evaluator = ModelEvaluator(f"{model.__class__.__name__}_{self.pipeline_id}")
            
            # Comprehensive evaluation
            cv_folds = self.config["model_config"]["cv_folds"]
            results = self.evaluator.evaluate_model_comprehensive(
                model.model,
                processed_data["X_test"],
                processed_data["y_test"],
                processed_data["X_train"],
                processed_data["y_train"],
                cv=cv_folds
            )
            
            # Generate predictions for plotting
            y_pred_test = model.predict(processed_data["X_test"])
            y_pred_train = model.predict(processed_data["X_train"])
            
            # Create plots if configured
            save_plots = True  # Default to True
            if save_plots:
                plots_dir = Path("data/exports/plots")
                plots_dir.mkdir(parents=True, exist_ok=True)
                
                # Prediction plots
                self.evaluator.plot_predictions(
                    processed_data["y_test"], 
                    y_pred_test,
                    title="Test Set Predictions",
                    save_path=plots_dir / f"test_predictions_{self.pipeline_id}.png"
                )
                
                # Residual plots
                self.evaluator.plot_residuals(
                    processed_data["y_test"], 
                    y_pred_test,
                    title="Test Set Residuals",
                    save_path=plots_dir / f"test_residuals_{self.pipeline_id}.png"
                )
            
            # Generate evaluation report
            save_reports = True  # Default to True
            if save_reports:
                reports_dir = Path("data/exports/reports")
                reports_dir.mkdir(parents=True, exist_ok=True)
                
                report = self.evaluator.generate_evaluation_report(
                    results,
                    save_path=reports_dir / f"evaluation_report_{self.pipeline_id}.txt"
                )
                
                logger.info(f"Evaluation report saved to: {reports_dir / f'evaluation_report_{self.pipeline_id}.txt'}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            raise
    
    def save_model(self, model: Any, processed_data: Dict[str, Any], 
                   evaluation_results: Dict[str, Any]) -> str:
        """
        Save the trained model and metadata
        
        Args:
            model: Trained model
            processed_data: Preprocessed data
            evaluation_results: Evaluation results
            
        Returns:
            Path to saved model
        """
        try:
            logger.info("Saving model")
            
            # Create models directory
            models_dir = Path("data/models")
            models_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model
            model_filename = f"grade_model_{self.pipeline_id}.joblib"
            model_path = models_dir / model_filename
            joblib.dump(model, model_path)
            
            # Save scaler
            scaler_filename = f"scaler_{self.pipeline_id}.joblib"
            scaler_path = models_dir / scaler_filename
            joblib.dump(processed_data["scaler"], scaler_path)
            
            # Save metadata
            metadata = {
                "pipeline_id": self.pipeline_id,
                "timestamp": datetime.now().isoformat(),
                "model_type": model.__class__.__name__,
                "model_path": str(model_path),
                "scaler_path": str(scaler_path),
                "config": self.config,
                "evaluation_results": evaluation_results,
                "feature_names": processed_data["X_train"].columns.tolist() if hasattr(processed_data["X_train"], 'columns') else None
            }
            
            metadata_filename = f"model_metadata_{self.pipeline_id}.json"
            metadata_path = models_dir / metadata_filename
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.model_path = str(model_path)
            
            logger.info(f"Model saved to: {model_path}")
            logger.info(f"Scaler saved to: {scaler_path}")
            logger.info(f"Metadata saved to: {metadata_path}")
            
            return str(model_path)
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def run_complete_pipeline(self, element: str = "CU", dataset: str = "MAIN", 
                            limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Run the complete training pipeline
        
        Args:
            element: Element to predict
            dataset: Dataset to use
            limit: Limit number of records (for testing)
            
        Returns:
            Complete pipeline results
        """
        try:
            logger.info(f"Starting complete training pipeline for {element}")
            
            # Step 1: Load data
            data = self.load_data(element, dataset, limit)
            
            # Step 2: Prepare features
            features, target = self.prepare_features(data)
            
            # Step 3: Split and preprocess
            processed_data = self.split_and_preprocess_data(features, target)
            
            # Step 4: Train model
            model = self.train_model(processed_data)
            
            # Step 5: Evaluate model
            evaluation_results = self.evaluate_model(model, processed_data)
            
            # Step 6: Save model
            model_path = self.save_model(model, processed_data, evaluation_results)
            
            # Compile results
            pipeline_results = {
                "pipeline_id": self.pipeline_id,
                "element": element,
                "dataset": dataset,
                "data_records": len(data),
                "features_count": len(features.columns),
                "model_path": model_path,
                "evaluation_results": evaluation_results,
                "timestamp": datetime.now().isoformat()
            }
            
            self.training_results = pipeline_results
            
            logger.info(f"Training pipeline completed successfully!")
            logger.info(f"Test RMSE: {evaluation_results.get('test_metrics', {}).get('rmse', 'N/A')}")
            logger.info(f"Test R²: {evaluation_results.get('test_metrics', {}).get('r2_score', 'N/A')}")
            logger.info(f"Test MAE: {evaluation_results.get('test_metrics', {}).get('mae', 'N/A')}")
            
            return pipeline_results
            
        except Exception as e:
            logger.error(f"Error in complete pipeline: {str(e)}")
            raise
    
    def get_training_summary(self) -> str:
        """
        Get a summary of the training results
        
        Returns:
            Training summary string
        """
        if not self.training_results:
            return "No training results available"
        
        results = self.training_results
        eval_results = results.get("evaluation_results", {})
        test_metrics = eval_results.get("test_metrics", {})
        
        summary = f"""
TRAINING PIPELINE SUMMARY
========================
Pipeline ID: {results.get('pipeline_id', 'N/A')}
Element: {results.get('element', 'N/A')}
Dataset: {results.get('dataset', 'N/A')}
Records: {results.get('data_records', 'N/A')}
Features: {results.get('features_count', 'N/A')}

MODEL PERFORMANCE:
- Test RMSE: {test_metrics.get('rmse', 'N/A'):.4f}
- Test R²: {test_metrics.get('r2_score', 'N/A'):.4f}
- Test MAE: {test_metrics.get('mae', 'N/A'):.4f}

Model saved to: {results.get('model_path', 'N/A')}
Completed: {results.get('timestamp', 'N/A')}
"""
        return summary


# Example usage
if __name__ == "__main__":
    # Initialize and run pipeline
    pipeline = TrainingPipeline()
    
    try:
        # Run complete pipeline
        results = pipeline.run_complete_pipeline(
            element="CU",
            dataset="MAIN",
            limit=10000  # Limit for testing
        )
        
        # Print summary
        print(pipeline.get_training_summary())
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        print(f"Error: {str(e)}")
