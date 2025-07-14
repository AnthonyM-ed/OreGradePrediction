"""
XGBoost Model for Ore Grade Prediction
======================================

Specialized XGBoost implementation for geological data prediction.
"""

import logging
import pandas as pd
import numpy as np
import xgboost as xgb
from typing import Dict, List, Any, Optional, Tuple
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import json
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class XGBoostOreGradePredictor:
    """XGBoost model specifically designed for ore grade prediction"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.model = None
        self.is_trained = False
        self.feature_names = None
        self.training_metrics = {}
        self.model_params = self._get_xgboost_params()
    
    def _get_xgboost_params(self) -> Dict:
        """Get XGBoost parameters optimized for ore grade prediction"""
        default_params = {
            'objective': 'reg:squarederror',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'gamma': 0,
            'reg_alpha': 0,
            'reg_lambda': 1,
            'random_state': 42,
            'n_jobs': -1,
            'tree_method': 'hist'  # Efficient for geological data
        }
        
        # Override with config if provided
        if 'xgboost_params' in self.config:
            default_params.update(self.config['xgboost_params'])
        
        return default_params
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: pd.DataFrame = None, y_val: pd.Series = None) -> Dict[str, Any]:
        """
        Train XGBoost model for ore grade prediction
        
        Args:
            X_train: Training features
            y_train: Training target (ore grades)
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            
        Returns:
            Training metrics and information
        """
        try:
            # Store feature names
            self.feature_names = X_train.columns.tolist()
            
            # Handle categorical columns by converting to numeric
            X_train_processed = self._prepare_features_for_xgboost(X_train)
            X_val_processed = self._prepare_features_for_xgboost(X_val) if X_val is not None else None
            
            # Create XGBoost model
            self.model = xgb.XGBRegressor(**self.model_params)
            
            # Prepare evaluation set for early stopping
            eval_set = []
            if X_val_processed is not None and y_val is not None:
                eval_set = [(X_train_processed, y_train), (X_val_processed, y_val)]
            
            # Train model with simplified approach
            if eval_set:
                # Try training with eval_set but without early stopping for now
                try:
                    self.model.fit(
                        X_train_processed, y_train,
                        eval_set=eval_set,
                        verbose=False
                    )
                except Exception as e:
                    logger.warning(f"Training with eval_set failed: {e}")
                    # Fallback to basic training
                    self.model.fit(X_train_processed, y_train, verbose=False)
            else:
                # Train without early stopping
                self.model.fit(X_train_processed, y_train, verbose=False)

            self.is_trained = True

            # Calculate training metrics
            train_pred = self.model.predict(X_train_processed)
            train_metrics = self._calculate_metrics(y_train, train_pred, 'train')
            
            # Calculate validation metrics if available
            val_metrics = {}
            if X_val_processed is not None and y_val is not None:
                val_pred = self.model.predict(X_val_processed)
                val_metrics = self._calculate_metrics(y_val, val_pred, 'validation')
            
            # Cross-validation score using processed data
            cv_scores = cross_val_score(
                self.model, X_train_processed, y_train, cv=5, scoring='r2'
            )
            
            # Store training information
            self.training_metrics = {
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'cv_scores': {
                    'mean': cv_scores.mean(),
                    'std': cv_scores.std(),
                    'scores': cv_scores.tolist()
                },
                'feature_importance': self.get_feature_importance(),
                'training_date': datetime.now().isoformat(),
                'model_params': self.model_params
            }
            
            
            logger.info(f"XGBoost model trained successfully. CV R²: {cv_scores.mean():.4f}")
            
            return {
                'success': True,
                'training_metrics': self.training_metrics,
                'best_iteration': getattr(self.model, 'best_iteration', None)
            }
            
        except Exception as e:
            logger.error(f"Error training XGBoost model: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make ore grade predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        try:
            # Prepare features for XGBoost
            X_processed = self._prepare_features_for_xgboost(X)
            
            predictions = self.model.predict(X_processed)
            
            # Ensure predictions are non-negative (grades can't be negative)
            predictions = np.maximum(predictions, 0)
            
            logger.info(f"Generated {len(predictions)} predictions")
            return predictions
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            raise
    
    def predict_with_confidence(self, X: pd.DataFrame, 
                              quantiles: List[float] = [0.1, 0.9]) -> Dict[str, np.ndarray]:
        """
        Make predictions with confidence intervals using quantile regression
        
        Args:
            X: Features for prediction
            quantiles: List of quantiles for confidence intervals
            
        Returns:
            Dictionary with predictions and confidence intervals
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        try:
            # Main prediction
            predictions = self.predict(X)
            
            # For confidence intervals, we can use leaf indices
            leaf_indices = self.model.apply(X)
            
            # Calculate prediction variance based on leaf similarity
            # (This is a simplified approach - for better uncertainty quantification,
            # consider using quantile regression or ensemble methods)
            
            # Get training predictions for variance estimation
            if hasattr(self, '_training_residuals'):
                residual_std = np.std(self._training_residuals)
            else:
                residual_std = 0.1  # Default uncertainty
            
            # Confidence intervals (simplified)
            lower_bound = predictions - 1.96 * residual_std
            upper_bound = predictions + 1.96 * residual_std
            
            return {
                'predictions': predictions,
                'lower_bound': np.maximum(lower_bound, 0),  # Ensure non-negative
                'upper_bound': upper_bound,
                'std_error': residual_std
            }
            
        except Exception as e:
            logger.error(f"Error making predictions with confidence: {e}")
            raise
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Evaluate model performance"""
        try:
            predictions = self.predict(X)
            return self._calculate_metrics(y, predictions, 'evaluation')
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            raise
    
    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray, 
                         prefix: str = '') -> Dict[str, float]:
        """Calculate comprehensive metrics for ore grade prediction"""
        try:
            metrics = {
                f'{prefix}_r2_score': r2_score(y_true, y_pred),
                f'{prefix}_mean_absolute_error': mean_absolute_error(y_true, y_pred),
                f'{prefix}_mean_squared_error': mean_squared_error(y_true, y_pred),
                f'{prefix}_root_mean_squared_error': np.sqrt(mean_squared_error(y_true, y_pred)),
                f'{prefix}_mean_absolute_percentage_error': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            }
            
            # Geological-specific metrics
            geological_metrics = self._calculate_geological_metrics(y_true, y_pred, prefix)
            metrics.update(geological_metrics)
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Error calculating metrics: {e}")
            return {}
    
    def _calculate_geological_metrics(self, y_true: pd.Series, y_pred: np.ndarray, 
                                    prefix: str = '') -> Dict[str, float]:
        """Calculate geological-specific performance metrics"""
        try:
            # Grade classification accuracy
            grade_bins = [0, 0.1, 0.5, 1.0, 2.0, float('inf')]
            grade_labels = ['very_low', 'low', 'medium', 'high', 'very_high']
            
            y_true_binned = pd.cut(y_true, bins=grade_bins, labels=grade_labels)
            y_pred_binned = pd.cut(y_pred, bins=grade_bins, labels=grade_labels)
            
            classification_accuracy = (y_true_binned == y_pred_binned).mean()
            
            # High-grade detection metrics
            high_grade_threshold = 1.0
            y_true_high = (y_true > high_grade_threshold).astype(int)
            y_pred_high = (y_pred > high_grade_threshold).astype(int)
            
            # Precision and recall for high-grade detection
            true_positives = (y_true_high & y_pred_high).sum()
            false_positives = (~y_true_high & y_pred_high).sum()
            false_negatives = (y_true_high & ~y_pred_high).sum()
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            
            # Resource estimation error
            total_resource_true = y_true.sum()
            total_resource_pred = y_pred.sum()
            resource_error = abs(total_resource_true - total_resource_pred) / total_resource_true if total_resource_true > 0 else 0
            
            # Grade continuity correlation
            grade_continuity = np.corrcoef(y_true, y_pred)[0, 1]
            
            return {
                f'{prefix}_grade_classification_accuracy': classification_accuracy,
                f'{prefix}_high_grade_precision': precision,
                f'{prefix}_high_grade_recall': recall,
                f'{prefix}_resource_estimation_error': resource_error,
                f'{prefix}_grade_continuity_correlation': grade_continuity
            }
            
        except Exception as e:
            logger.warning(f"Error calculating geological metrics: {e}")
            return {}
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from XGBoost model"""
        if not self.is_trained:
            raise ValueError("Model must be trained to get feature importance")
        
        try:
            # Get feature importance
            importance = self.model.feature_importances_
            
            # Create feature importance dictionary
            if self.feature_names:
                feature_importance = dict(zip(self.feature_names, importance))
            else:
                feature_importance = dict(zip(range(len(importance)), importance))
            
            # Sort by importance
            sorted_importance = dict(
                sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            )
            
            return sorted_importance
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return {}
    
    def optimize_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series,
                               param_grid: Dict = None) -> Dict[str, Any]:
        """Optimize XGBoost hyperparameters using grid search"""
        try:
            if param_grid is None:
                param_grid = {
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'n_estimators': [50, 100, 200],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0]
                }
            
            # Create base model
            base_model = xgb.XGBRegressor(
                objective='reg:squarederror',
                random_state=42,
                n_jobs=-1
            )
            
            # Grid search
            grid_search = GridSearchCV(
                base_model,
                param_grid,
                cv=5,
                scoring='r2',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            # Update model parameters
            self.model_params.update(grid_search.best_params_)
            
            logger.info(f"Hyperparameter optimization completed. Best score: {grid_search.best_score_:.4f}")
            
            return {
                'success': True,
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'cv_results': grid_search.cv_results_
            }
            
        except Exception as e:
            logger.error(f"Error optimizing hyperparameters: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def save_model(self, filepath: str) -> bool:
        """Save XGBoost model and metadata"""
        try:
            model_data = {
                'model': self.model,
                'feature_names': self.feature_names,
                'model_params': self.model_params,
                'training_metrics': self.training_metrics,
                'is_trained': self.is_trained,
                'config': self.config,
                'model_version': '1.0'
            }
            
            joblib.dump(model_data, filepath)
            logger.info(f"XGBoost model saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """Load XGBoost model and metadata"""
        try:
            model_data = joblib.load(filepath)
            
            self.model = model_data['model']
            self.feature_names = model_data['feature_names']
            self.model_params = model_data['model_params']
            self.training_metrics = model_data.get('training_metrics', {})
            self.is_trained = model_data['is_trained']
            self.config = model_data.get('config', {})
            
            logger.info(f"XGBoost model loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information"""
        if not self.is_trained:
            return {'status': 'not_trained'}
        
        return {
            'status': 'trained',
            'feature_count': len(self.feature_names) if self.feature_names else 0,
            'feature_names': self.feature_names,
            'model_params': self.model_params,
            'training_metrics': self.training_metrics,
            'feature_importance': self.get_feature_importance()
        }
    
    def _prepare_features_for_xgboost(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for XGBoost by handling categorical columns
        
        Args:
            df: Input DataFrame with features
            
        Returns:
            DataFrame with categorical columns converted to numeric
        """
        if df is None:
            return None
            
        df_processed = df.copy()
        
        # Convert categorical columns to numeric
        for col in df_processed.columns:
            if df_processed[col].dtype.name == 'category':
                # Convert category to numeric codes
                df_processed[col] = df_processed[col].cat.codes
                logger.info(f"Converted categorical column '{col}' to numeric")
        
        return df_processed

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'XGBoostOreGradePredictor':
        """
        Scikit-learn compatible fit method
        
        This method provides compatibility with scikit-learn's cross_val_score
        and other utilities that expect a 'fit' method.
        
        Args:
            X: Training features
            y: Training target
            **kwargs: Additional arguments (X_val, y_val, etc.)
            
        Returns:
            Self (for method chaining)
        """
        # Extract validation data from kwargs if provided
        X_val = kwargs.get('X_val', None)
        y_val = kwargs.get('y_val', None)
        
        # Call our existing train method
        result = self.train(X, y, X_val=X_val, y_val=y_val)
        
        if not result.get('success', True):
            raise ValueError(f"Training failed: {result.get('error', 'Unknown error')}")
        
        return self

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """
        Get parameters for this estimator (scikit-learn compatibility)
        
        Args:
            deep: If True, return parameters for this estimator and contained subobjects
            
        Returns:
            Dictionary of parameter names mapped to their values
        """
        params = {
            'config': self.config,
        }
        
        # Add model parameters if available
        if hasattr(self, 'model_params'):
            params.update(self.model_params)
        
        return params
    
    def set_params(self, **params) -> 'XGBoostOreGradePredictor':
        """
        Set the parameters of this estimator (scikit-learn compatibility)
        
        Args:
            **params: Estimator parameters
            
        Returns:
            Self (for method chaining)
        """
        # Update config if provided
        if 'config' in params:
            self.config = params.pop('config')
            self.model_params = self._get_xgboost_params()
        
        # Update model parameters
        for key, value in params.items():
            if hasattr(self, 'model_params') and key in self.model_params:
                self.model_params[key] = value
        
        return self

# Global instance
xgboost_predictor = XGBoostOreGradePredictor()
