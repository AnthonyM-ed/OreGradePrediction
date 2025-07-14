"""
Cross-Validation Module for Ore Grade Prediction Models

This module provides comprehensive cross-validation functionality with RMSE, R², and MAE metrics
for evaluating model performance and stability.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    KFold, StratifiedKFold, TimeSeriesSplit, 
    cross_val_score, cross_validate
)
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    make_scorer
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CrossValidator:
    """
    Comprehensive cross-validation for ore grade prediction models
    """
    
    def __init__(self, cv_folds: int = 5, random_state: int = 42):
        """
        Initialize cross-validator
        
        Args:
            cv_folds: Number of cross-validation folds
            random_state: Random state for reproducibility
        """
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.cv_results = {}
        
    def _get_scoring_metrics(self) -> Dict[str, Any]:
        """
        Get scoring metrics for cross-validation
        
        Returns:
            Dictionary of scoring metrics
        """
        return {
            'neg_mean_squared_error': 'neg_mean_squared_error',
            'neg_mean_absolute_error': 'neg_mean_absolute_error',
            'r2': 'r2',
            'neg_root_mean_squared_error': 'neg_root_mean_squared_error'
        }
    
    def _calculate_geological_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate geological-specific metrics
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of geological metrics
        """
        try:
            # MAPE (Mean Absolute Percentage Error)
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            
            # Bias metrics
            bias = np.mean(y_pred - y_true)
            relative_bias = (bias / np.mean(y_true)) * 100
            
            # Grade classification metrics for different cutoffs
            cutoff_accuracies = {}
            cutoff_grades = [100, 500, 1000, 2000, 5000]  # PPM cutoffs
            
            for cutoff in cutoff_grades:
                true_class = (y_true >= cutoff).astype(int)
                pred_class = (y_pred >= cutoff).astype(int)
                accuracy = np.mean(true_class == pred_class)
                cutoff_accuracies[f'cutoff_{cutoff}ppm'] = accuracy
            
            geological_metrics = {
                'mape': mape,
                'bias': bias,
                'relative_bias_percent': relative_bias,
                **cutoff_accuracies
            }
            
            return geological_metrics
            
        except Exception as e:
            logger.error(f"Error calculating geological metrics: {str(e)}")
            return {}
    
    def perform_kfold_cv(self, model: Any, X: Union[np.ndarray, pd.DataFrame], 
                        y: Union[np.ndarray, pd.Series]) -> Dict[str, Any]:
        """
        Perform K-Fold cross-validation
        
        Args:
            model: Model to evaluate
            X: Feature matrix
            y: Target values
            
        Returns:
            Cross-validation results
        """
        try:
            logger.info(f"Performing {self.cv_folds}-fold cross-validation")
            
            # Initialize K-Fold
            kfold = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            
            # Get scoring metrics
            scoring = self._get_scoring_metrics()
            
            # Perform cross-validation
            cv_results = cross_validate(
                model, X, y, 
                cv=kfold, 
                scoring=scoring,
                return_train_score=True,
                return_estimator=True
            )
            
            # Calculate additional metrics manually
            fold_predictions = []
            fold_actuals = []
            geological_metrics = []
            
            for i, (train_idx, test_idx) in enumerate(kfold.split(X)):
                # Get fold data
                if isinstance(X, pd.DataFrame):
                    X_train_fold, X_test_fold = X.iloc[train_idx], X.iloc[test_idx]
                else:
                    X_train_fold, X_test_fold = X[train_idx], X[test_idx]
                
                if isinstance(y, pd.Series):
                    y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]
                else:
                    y_train_fold, y_test_fold = y[train_idx], y[test_idx]
                
                # Train and predict
                model.fit(X_train_fold, y_train_fold)
                y_pred_fold = model.predict(X_test_fold)
                
                # Store predictions
                fold_predictions.extend(y_pred_fold)
                fold_actuals.extend(y_test_fold)
                
                # Calculate geological metrics for this fold
                fold_geo_metrics = self._calculate_geological_metrics(y_test_fold, y_pred_fold)
                geological_metrics.append(fold_geo_metrics)
            
            # Process results
            results = {
                'cv_method': 'KFold',
                'n_folds': self.cv_folds,
                'train_scores': {},
                'test_scores': {},
                'fold_predictions': fold_predictions,
                'fold_actuals': fold_actuals,
                'geological_metrics': geological_metrics
            }
            
            # Process sklearn scoring results
            for metric_name, scores in cv_results.items():
                if metric_name.startswith('test_'):
                    clean_name = metric_name.replace('test_', '')
                    results['test_scores'][clean_name] = {
                        'scores': scores.tolist(),
                        'mean': scores.mean(),
                        'std': scores.std(),
                        'min': scores.min(),
                        'max': scores.max()
                    }
                elif metric_name.startswith('train_'):
                    clean_name = metric_name.replace('train_', '')
                    results['train_scores'][clean_name] = {
                        'scores': scores.tolist(),
                        'mean': scores.mean(),
                        'std': scores.std(),
                        'min': scores.min(),
                        'max': scores.max()
                    }
            
            # Convert negative scores to positive for interpretability
            if 'neg_mean_squared_error' in results['test_scores']:
                rmse_scores = np.sqrt(-np.array(results['test_scores']['neg_mean_squared_error']['scores']))
                results['test_scores']['rmse'] = {
                    'scores': rmse_scores.tolist(),
                    'mean': rmse_scores.mean(),
                    'std': rmse_scores.std(),
                    'min': rmse_scores.min(),
                    'max': rmse_scores.max()
                }
            
            if 'neg_mean_absolute_error' in results['test_scores']:
                mae_scores = -np.array(results['test_scores']['neg_mean_absolute_error']['scores'])
                results['test_scores']['mae'] = {
                    'scores': mae_scores.tolist(),
                    'mean': mae_scores.mean(),
                    'std': mae_scores.std(),
                    'min': mae_scores.min(),
                    'max': mae_scores.max()
                }
            
            # Average geological metrics across folds
            avg_geological_metrics = {}
            if geological_metrics:
                for metric_name in geological_metrics[0].keys():
                    metric_values = [fold_metrics[metric_name] for fold_metrics in geological_metrics]
                    avg_geological_metrics[metric_name] = {
                        'mean': np.mean(metric_values),
                        'std': np.std(metric_values),
                        'scores': metric_values
                    }
            
            results['geological_metrics_summary'] = avg_geological_metrics
            
            self.cv_results['kfold'] = results
            
            logger.info(f"K-Fold CV completed - RMSE: {results['test_scores'].get('rmse', {}).get('mean', 'N/A'):.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in K-Fold cross-validation: {str(e)}")
            raise
    
    def perform_stratified_cv(self, model: Any, X: Union[np.ndarray, pd.DataFrame], 
                            y: Union[np.ndarray, pd.Series], 
                            n_bins: int = 10) -> Dict[str, Any]:
        """
        Perform Stratified cross-validation for regression (binned targets)
        
        Args:
            model: Model to evaluate
            X: Feature matrix
            y: Target values
            n_bins: Number of bins for stratification
            
        Returns:
            Cross-validation results
        """
        try:
            logger.info(f"Performing stratified {self.cv_folds}-fold cross-validation")
            
            # Create bins for stratification
            y_binned = pd.cut(y, bins=n_bins, labels=False)
            
            # Initialize Stratified K-Fold
            skfold = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            
            # Get scoring metrics
            scoring = self._get_scoring_metrics()
            
            # Perform cross-validation
            cv_results = cross_validate(
                model, X, y, 
                cv=skfold.split(X, y_binned), 
                scoring=scoring,
                return_train_score=True
            )
            
            # Process results similar to K-Fold
            results = {
                'cv_method': 'StratifiedKFold',
                'n_folds': self.cv_folds,
                'n_bins': n_bins,
                'train_scores': {},
                'test_scores': {}
            }
            
            # Process sklearn scoring results
            for metric_name, scores in cv_results.items():
                if metric_name.startswith('test_'):
                    clean_name = metric_name.replace('test_', '')
                    results['test_scores'][clean_name] = {
                        'scores': scores.tolist(),
                        'mean': scores.mean(),
                        'std': scores.std()
                    }
                elif metric_name.startswith('train_'):
                    clean_name = metric_name.replace('train_', '')
                    results['train_scores'][clean_name] = {
                        'scores': scores.tolist(),
                        'mean': scores.mean(),
                        'std': scores.std()
                    }
            
            # Convert negative scores
            if 'neg_mean_squared_error' in results['test_scores']:
                rmse_scores = np.sqrt(-np.array(results['test_scores']['neg_mean_squared_error']['scores']))
                results['test_scores']['rmse'] = {
                    'scores': rmse_scores.tolist(),
                    'mean': rmse_scores.mean(),
                    'std': rmse_scores.std()
                }
            
            if 'neg_mean_absolute_error' in results['test_scores']:
                mae_scores = -np.array(results['test_scores']['neg_mean_absolute_error']['scores'])
                results['test_scores']['mae'] = {
                    'scores': mae_scores.tolist(),
                    'mean': mae_scores.mean(),
                    'std': mae_scores.std()
                }
            
            self.cv_results['stratified'] = results
            
            logger.info(f"Stratified CV completed - RMSE: {results['test_scores'].get('rmse', {}).get('mean', 'N/A'):.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in Stratified cross-validation: {str(e)}")
            raise
    
    def perform_time_series_cv(self, model: Any, X: Union[np.ndarray, pd.DataFrame], 
                             y: Union[np.ndarray, pd.Series]) -> Dict[str, Any]:
        """
        Perform Time Series cross-validation (for temporal data)
        
        Args:
            model: Model to evaluate
            X: Feature matrix
            y: Target values
            
        Returns:
            Cross-validation results
        """
        try:
            logger.info(f"Performing time series {self.cv_folds}-fold cross-validation")
            
            # Initialize Time Series Split
            tscv = TimeSeriesSplit(n_splits=self.cv_folds)
            
            # Get scoring metrics
            scoring = self._get_scoring_metrics()
            
            # Perform cross-validation
            cv_results = cross_validate(
                model, X, y, 
                cv=tscv, 
                scoring=scoring,
                return_train_score=True
            )
            
            # Process results
            results = {
                'cv_method': 'TimeSeriesSplit',
                'n_folds': self.cv_folds,
                'train_scores': {},
                'test_scores': {}
            }
            
            # Process sklearn scoring results
            for metric_name, scores in cv_results.items():
                if metric_name.startswith('test_'):
                    clean_name = metric_name.replace('test_', '')
                    results['test_scores'][clean_name] = {
                        'scores': scores.tolist(),
                        'mean': scores.mean(),
                        'std': scores.std()
                    }
                elif metric_name.startswith('train_'):
                    clean_name = metric_name.replace('train_', '')
                    results['train_scores'][clean_name] = {
                        'scores': scores.tolist(),
                        'mean': scores.mean(),
                        'std': scores.std()
                    }
            
            # Convert negative scores
            if 'neg_mean_squared_error' in results['test_scores']:
                rmse_scores = np.sqrt(-np.array(results['test_scores']['neg_mean_squared_error']['scores']))
                results['test_scores']['rmse'] = {
                    'scores': rmse_scores.tolist(),
                    'mean': rmse_scores.mean(),
                    'std': rmse_scores.std()
                }
            
            if 'neg_mean_absolute_error' in results['test_scores']:
                mae_scores = -np.array(results['test_scores']['neg_mean_absolute_error']['scores'])
                results['test_scores']['mae'] = {
                    'scores': mae_scores.tolist(),
                    'mean': mae_scores.mean(),
                    'std': mae_scores.std()
                }
            
            self.cv_results['time_series'] = results
            
            logger.info(f"Time Series CV completed - RMSE: {results['test_scores'].get('rmse', {}).get('mean', 'N/A'):.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in Time Series cross-validation: {str(e)}")
            raise
    
    def plot_cv_results(self, cv_method: str = 'kfold', metrics: List[str] = ['rmse', 'r2', 'mae'],
                       save_path: Optional[str] = None):
        """
        Plot cross-validation results
        
        Args:
            cv_method: CV method to plot ('kfold', 'stratified', 'time_series')
            metrics: Metrics to plot
            save_path: Path to save plot
        """
        try:
            if cv_method not in self.cv_results:
                logger.error(f"No results found for CV method: {cv_method}")
                return
            
            results = self.cv_results[cv_method]
            test_scores = results['test_scores']
            
            # Filter available metrics
            available_metrics = [m for m in metrics if m in test_scores]
            
            if not available_metrics:
                logger.error("No available metrics to plot")
                return
            
            # Create subplots
            fig, axes = plt.subplots(1, len(available_metrics), figsize=(5 * len(available_metrics), 6))
            if len(available_metrics) == 1:
                axes = [axes]
            
            for i, metric in enumerate(available_metrics):
                scores = test_scores[metric]['scores']
                mean_score = test_scores[metric]['mean']
                std_score = test_scores[metric]['std']
                
                # Box plot
                axes[i].boxplot(scores)
                axes[i].set_title(f'{metric.upper()} - {cv_method.upper()}\nMean: {mean_score:.4f} ± {std_score:.4f}')
                axes[i].set_ylabel(metric.upper())
                axes[i].set_xlabel('CV Fold')
                
                # Add mean line
                axes[i].axhline(y=mean_score, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_score:.4f}')
                axes[i].legend()
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting CV results: {str(e)}")
    
    def compare_cv_methods(self, metric: str = 'rmse') -> pd.DataFrame:
        """
        Compare different CV methods
        
        Args:
            metric: Metric to compare
            
        Returns:
            Comparison DataFrame
        """
        try:
            comparison_data = []
            
            for cv_method, results in self.cv_results.items():
                if metric in results['test_scores']:
                    metric_data = results['test_scores'][metric]
                    comparison_data.append({
                        'cv_method': cv_method,
                        'metric': metric,
                        'mean': metric_data['mean'],
                        'std': metric_data['std'],
                        'min': metric_data.get('min', np.min(metric_data['scores'])),
                        'max': metric_data.get('max', np.max(metric_data['scores']))
                    })
            
            if not comparison_data:
                logger.warning(f"No data available for metric: {metric}")
                return pd.DataFrame()
            
            comparison_df = pd.DataFrame(comparison_data)
            return comparison_df.sort_values('mean', ascending=(metric != 'r2'))
            
        except Exception as e:
            logger.error(f"Error comparing CV methods: {str(e)}")
            return pd.DataFrame()
    
    def get_cv_summary(self) -> str:
        """
        Get a summary of all cross-validation results
        
        Returns:
            Summary string
        """
        try:
            summary_lines = []
            summary_lines.append("CROSS-VALIDATION RESULTS SUMMARY")
            summary_lines.append("=" * 50)
            
            for cv_method, results in self.cv_results.items():
                summary_lines.append(f"\n{cv_method.upper()} RESULTS:")
                summary_lines.append("-" * 30)
                
                test_scores = results['test_scores']
                for metric, scores in test_scores.items():
                    if metric in ['rmse', 'mae', 'r2']:
                        summary_lines.append(f"{metric.upper()}: {scores['mean']:.4f} ± {scores['std']:.4f}")
                
                # Add geological metrics if available
                if 'geological_metrics_summary' in results:
                    geo_metrics = results['geological_metrics_summary']
                    if 'mape' in geo_metrics:
                        summary_lines.append(f"MAPE: {geo_metrics['mape']['mean']:.2f}% ± {geo_metrics['mape']['std']:.2f}%")
            
            return "\n".join(summary_lines)
            
        except Exception as e:
            logger.error(f"Error generating CV summary: {str(e)}")
            return "Error generating summary"


# Example usage
if __name__ == "__main__":
    logger.info("Cross-validation module loaded successfully")
    
    # Example initialization
    cv = CrossValidator(cv_folds=5, random_state=42)
    
    print("Cross-validation tools ready for use!")
    print("Available methods:")
    print("- perform_kfold_cv()")
    print("- perform_stratified_cv()")
    print("- perform_time_series_cv()")
    print("- plot_cv_results()")
    print("- compare_cv_methods()")
    print("- get_cv_summary()")
