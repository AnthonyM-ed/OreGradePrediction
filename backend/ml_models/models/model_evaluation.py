"""
Model Evaluation Module for Ore Grade Prediction

This module provides comprehensive evaluation metrics for machine learning models
including RMSE, R², MAE, and specialized geological evaluation metrics.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error, 
    r2_score,
    explained_variance_score
)
from sklearn.model_selection import cross_val_score
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive model evaluation for ore grade prediction models
    """
    
    def __init__(self, model_name: str = "Unknown"):
        self.model_name = model_name
        self.evaluation_history = []
        
    def calculate_basic_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate basic regression metrics: RMSE, R², MAE
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            
        Returns:
            Dictionary with metric names and values
        """
        try:
            metrics = {
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'r2_score': r2_score(y_true, y_pred),
                'mae': mean_absolute_error(y_true, y_pred),
                'mse': mean_squared_error(y_true, y_pred),
                'explained_variance': explained_variance_score(y_true, y_pred)
            }
            
            # Additional geological metrics
            metrics.update(self._calculate_geological_metrics(y_true, y_pred))
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating basic metrics: {str(e)}")
            return {}
    
    def _calculate_geological_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate geological-specific metrics
        
        Args:
            y_true: Actual grade values (PPM)
            y_pred: Predicted grade values (PPM)
            
        Returns:
            Dictionary with geological metrics
        """
        try:
            # Percentage error metrics
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            
            # Grade classification accuracy (for cutoff grades)
            cutoff_grades = [100, 500, 1000, 2000]  # PPM cutoffs
            classification_accuracy = {}
            
            for cutoff in cutoff_grades:
                true_class = (y_true >= cutoff).astype(int)
                pred_class = (y_pred >= cutoff).astype(int)
                accuracy = np.mean(true_class == pred_class)
                classification_accuracy[f'cutoff_{cutoff}ppm_accuracy'] = accuracy
            
            # Bias metrics
            bias = np.mean(y_pred - y_true)
            relative_bias = bias / np.mean(y_true) * 100
            
            geological_metrics = {
                'mape': mape,
                'bias': bias,
                'relative_bias_percent': relative_bias,
                **classification_accuracy
            }
            
            return geological_metrics
            
        except Exception as e:
            logger.error(f"Error calculating geological metrics: {str(e)}")
            return {}
    
    def cross_validate_model(self, model: Any, X: np.ndarray, y: np.ndarray, 
                           cv: int = 5, scoring: str = 'neg_mean_squared_error') -> Dict[str, float]:
        """
        Perform cross-validation on the model
        
        Args:
            model: Trained model
            X: Feature matrix
            y: Target values
            cv: Number of cross-validation folds
            scoring: Scoring metric
            
        Returns:
            Cross-validation results
        """
        try:
            # Multiple scoring metrics
            scoring_metrics = ['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2']
            cv_results = {}
            
            for metric in scoring_metrics:
                scores = cross_val_score(model, X, y, cv=cv, scoring=metric)
                cv_results[f'{metric}_mean'] = scores.mean()
                cv_results[f'{metric}_std'] = scores.std()
                cv_results[f'{metric}_scores'] = scores.tolist()
            
            # Convert negative scores to positive for MSE and MAE
            cv_results['rmse_mean'] = np.sqrt(-cv_results['neg_mean_squared_error_mean'])
            cv_results['rmse_std'] = cv_results['neg_mean_squared_error_std']
            cv_results['mae_mean'] = -cv_results['neg_mean_absolute_error_mean']
            cv_results['mae_std'] = cv_results['neg_mean_absolute_error_std']
            
            return cv_results
            
        except Exception as e:
            logger.error(f"Error in cross-validation: {str(e)}")
            return {}
    
    def evaluate_model_comprehensive(self, model: Any, X_test: np.ndarray, y_test: np.ndarray,
                                   X_train: Optional[np.ndarray] = None, y_train: Optional[np.ndarray] = None,
                                   cv: int = 5) -> Dict[str, Any]:
        """
        Comprehensive model evaluation including train/test metrics and cross-validation
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets
            X_train: Training features (optional)
            y_train: Training targets (optional)
            cv: Cross-validation folds
            
        Returns:
            Comprehensive evaluation results
        """
        try:
            results = {
                'model_name': self.model_name,
                'evaluation_timestamp': datetime.now().isoformat(),
                'test_metrics': {},
                'train_metrics': {},
                'cv_metrics': {}
            }
            
            # Test set evaluation
            y_pred_test = model.predict(X_test)
            results['test_metrics'] = self.calculate_basic_metrics(y_test, y_pred_test)
            
            # Training set evaluation (if provided)
            if X_train is not None and y_train is not None:
                y_pred_train = model.predict(X_train)
                results['train_metrics'] = self.calculate_basic_metrics(y_train, y_pred_train)
            
            # Cross-validation (if training data provided)
            if X_train is not None and y_train is not None:
                results['cv_metrics'] = self.cross_validate_model(model, X_train, y_train, cv=cv)
            
            # Feature importance (if available)
            if hasattr(model, 'feature_importances_'):
                results['feature_importance'] = model.feature_importances_.tolist()
            
            # Store evaluation history
            self.evaluation_history.append(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in comprehensive evaluation: {str(e)}")
            return {}
    
    def plot_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, 
                        title: str = "Predictions vs Actual", save_path: Optional[str] = None):
        """
        Create prediction vs actual plot
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            title: Plot title
            save_path: Path to save plot
        """
        try:
            plt.figure(figsize=(10, 8))
            
            # Scatter plot
            plt.scatter(y_true, y_pred, alpha=0.6, s=50)
            
            # Perfect prediction line
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
            
            # Labels and title
            plt.xlabel('Actual Grade (PPM)')
            plt.ylabel('Predicted Grade (PPM)')
            plt.title(f'{title} - {self.model_name}')
            plt.legend()
            
            # Add metrics to plot
            r2 = r2_score(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            
            plt.text(0.05, 0.95, f'R² = {r2:.4f}\nRMSE = {rmse:.2f}\nMAE = {mae:.2f}', 
                    transform=plt.gca().transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error creating prediction plot: {str(e)}")
    
    def plot_residuals(self, y_true: np.ndarray, y_pred: np.ndarray, 
                      title: str = "Residual Plot", save_path: Optional[str] = None):
        """
        Create residual plot
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            title: Plot title
            save_path: Path to save plot
        """
        try:
            residuals = y_true - y_pred
            
            plt.figure(figsize=(12, 5))
            
            # Residuals vs Predicted
            plt.subplot(1, 2, 1)
            plt.scatter(y_pred, residuals, alpha=0.6)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.xlabel('Predicted Grade (PPM)')
            plt.ylabel('Residuals (PPM)')
            plt.title('Residuals vs Predicted')
            
            # Residuals histogram
            plt.subplot(1, 2, 2)
            plt.hist(residuals, bins=30, alpha=0.7)
            plt.xlabel('Residuals (PPM)')
            plt.ylabel('Frequency')
            plt.title('Residuals Distribution')
            
            plt.suptitle(f'{title} - {self.model_name}')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error creating residual plot: {str(e)}")
    
    def generate_evaluation_report(self, results: Dict[str, Any], save_path: Optional[str] = None) -> str:
        """
        Generate a comprehensive evaluation report
        
        Args:
            results: Evaluation results from evaluate_model_comprehensive
            save_path: Path to save report
            
        Returns:
            Report as string
        """
        try:
            report_lines = []
            report_lines.append(f"MODEL EVALUATION REPORT")
            report_lines.append("=" * 50)
            report_lines.append(f"Model: {results.get('model_name', 'Unknown')}")
            report_lines.append(f"Evaluation Date: {results.get('evaluation_timestamp', 'Unknown')}")
            report_lines.append("")
            
            # Test metrics
            if 'test_metrics' in results and results['test_metrics']:
                report_lines.append("TEST SET METRICS:")
                report_lines.append("-" * 20)
                test_metrics = results['test_metrics']
                report_lines.append(f"RMSE: {test_metrics.get('rmse', 'N/A'):.4f}")
                report_lines.append(f"R² Score: {test_metrics.get('r2_score', 'N/A'):.4f}")
                report_lines.append(f"MAE: {test_metrics.get('mae', 'N/A'):.4f}")
                report_lines.append(f"MAPE: {test_metrics.get('mape', 'N/A'):.2f}%")
                report_lines.append(f"Bias: {test_metrics.get('bias', 'N/A'):.4f}")
                report_lines.append("")
            
            # Training metrics
            if 'train_metrics' in results and results['train_metrics']:
                report_lines.append("TRAINING SET METRICS:")
                report_lines.append("-" * 20)
                train_metrics = results['train_metrics']
                report_lines.append(f"RMSE: {train_metrics.get('rmse', 'N/A'):.4f}")
                report_lines.append(f"R² Score: {train_metrics.get('r2_score', 'N/A'):.4f}")
                report_lines.append(f"MAE: {train_metrics.get('mae', 'N/A'):.4f}")
                report_lines.append("")
            
            # Cross-validation metrics
            if 'cv_metrics' in results and results['cv_metrics']:
                report_lines.append("CROSS-VALIDATION METRICS:")
                report_lines.append("-" * 25)
                cv_metrics = results['cv_metrics']
                report_lines.append(f"CV RMSE: {cv_metrics.get('rmse_mean', 'N/A'):.4f} ± {cv_metrics.get('rmse_std', 'N/A'):.4f}")
                report_lines.append(f"CV R²: {cv_metrics.get('r2_mean', 'N/A'):.4f} ± {cv_metrics.get('r2_std', 'N/A'):.4f}")
                report_lines.append(f"CV MAE: {cv_metrics.get('mae_mean', 'N/A'):.4f} ± {cv_metrics.get('mae_std', 'N/A'):.4f}")
                report_lines.append("")
            
            report_text = "\n".join(report_lines)
            
            if save_path:
                with open(save_path, 'w') as f:
                    f.write(report_text)
            
            return report_text
            
        except Exception as e:
            logger.error(f"Error generating evaluation report: {str(e)}")
            return ""


class ModelComparator:
    """
    Compare multiple models and their performance
    """
    
    def __init__(self):
        self.model_results = {}
    
    def add_model_results(self, model_name: str, results: Dict[str, Any]):
        """
        Add model evaluation results for comparison
        
        Args:
            model_name: Name of the model
            results: Evaluation results
        """
        self.model_results[model_name] = results
    
    def compare_models(self, metric: str = 'rmse', dataset: str = 'test') -> pd.DataFrame:
        """
        Compare models based on specified metric
        
        Args:
            metric: Metric to compare ('rmse', 'r2_score', 'mae')
            dataset: Dataset to compare ('test', 'train', 'cv')
            
        Returns:
            DataFrame with model comparison
        """
        try:
            comparison_data = []
            
            for model_name, results in self.model_results.items():
                metrics_key = f'{dataset}_metrics'
                if metrics_key in results:
                    metrics = results[metrics_key]
                    if dataset == 'cv':
                        metric_value = metrics.get(f'{metric}_mean', np.nan)
                    else:
                        metric_value = metrics.get(metric, np.nan)
                    
                    comparison_data.append({
                        'model': model_name,
                        'metric': metric,
                        'value': metric_value,
                        'dataset': dataset
                    })
            
            return pd.DataFrame(comparison_data).sort_values('value', ascending=(metric != 'r2_score'))
            
        except Exception as e:
            logger.error(f"Error comparing models: {str(e)}")
            return pd.DataFrame()
    
    def plot_model_comparison(self, metric: str = 'rmse', dataset: str = 'test', 
                            save_path: Optional[str] = None):
        """
        Plot model comparison
        
        Args:
            metric: Metric to compare
            dataset: Dataset to compare
            save_path: Path to save plot
        """
        try:
            comparison_df = self.compare_models(metric, dataset)
            
            if comparison_df.empty:
                logger.warning("No data available for comparison")
                return
            
            plt.figure(figsize=(10, 6))
            bars = plt.bar(comparison_df['model'], comparison_df['value'])
            
            # Color bars based on performance
            colors = ['green' if metric == 'r2_score' else 'red' if i == 0 else 'orange' 
                     for i, _ in enumerate(bars)]
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            plt.xlabel('Model')
            plt.ylabel(f'{metric.upper()} ({dataset.capitalize()} Set)')
            plt.title(f'Model Comparison - {metric.upper()} on {dataset.capitalize()} Set')
            plt.xticks(rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, comparison_df['value']):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                        f'{value:.4f}', ha='center', va='bottom')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting model comparison: {str(e)}")


# Example usage
if __name__ == "__main__":
    # This would be used in conjunction with actual model training
    logger.info("Model Evaluation Module loaded successfully")
    
    # Example initialization
    evaluator = ModelEvaluator("XGBoost_GradePredictor")
    comparator = ModelComparator()
    
    print("Model evaluation tools ready for use!")
