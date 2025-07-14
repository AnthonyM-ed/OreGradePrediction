"""
Ensemble Models for Ore Grade Prediction
========================================

Implements ensemble methods for improved prediction accuracy.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, 
    VotingRegressor, BaggingRegressor, ExtraTreesRegressor
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import json
import os

logger = logging.getLogger(__name__)

class EnsembleGradePredictor:
    """Ensemble model for ore grade prediction"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.models = {}
        self.ensemble_model = None
        self.scaler = None
        self.feature_names = None
        self.is_trained = False
        
        # Initialize base models
        self._initialize_base_models()
    
    def _initialize_base_models(self):
        """Initialize base models for ensemble"""
        self.models = {
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            ),
            'extra_trees': ExtraTreesRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ),
            'linear_regression': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=0.1),
            'svr': SVR(kernel='rbf', C=1.0, epsilon=0.1)
        }
        
        logger.info("Base models initialized")
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train ensemble model"""
        try:
            # Store feature names
            self.feature_names = X.columns.tolist()
            
            # Scale features
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Train individual models
            model_scores = {}
            trained_models = []
            
            for name, model in self.models.items():
                try:
                    # Train model
                    model.fit(X_scaled, y)
                    
                    # Cross-validation score
                    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
                    model_scores[name] = {
                        'cv_mean': cv_scores.mean(),
                        'cv_std': cv_scores.std(),
                        'trained': True
                    }
                    
                    trained_models.append((name, model))
                    
                except Exception as e:
                    logger.warning(f"Failed to train {name}: {e}")
                    model_scores[name] = {
                        'cv_mean': 0,
                        'cv_std': 0,
                        'trained': False,
                        'error': str(e)
                    }
            
            # Create voting ensemble with best models
            if trained_models:
                # Select top performing models
                top_models = sorted(
                    [(name, model) for name, model in trained_models],
                    key=lambda x: model_scores[x[0]]['cv_mean'],
                    reverse=True
                )[:5]  # Top 5 models
                
                # Create voting regressor
                self.ensemble_model = VotingRegressor(
                    estimators=top_models,
                    n_jobs=-1
                )
                
                # Train ensemble
                self.ensemble_model.fit(X_scaled, y)
                
                # Ensemble cross-validation score
                ensemble_cv_scores = cross_val_score(
                    self.ensemble_model, X_scaled, y, cv=5, scoring='r2'
                )
                
                model_scores['ensemble'] = {
                    'cv_mean': ensemble_cv_scores.mean(),
                    'cv_std': ensemble_cv_scores.std(),
                    'trained': True
                }
                
                self.is_trained = True
                
                logger.info(f"Ensemble model trained with {len(top_models)} base models")
                
                return {
                    'success': True,
                    'model_scores': model_scores,
                    'ensemble_score': ensemble_cv_scores.mean(),
                    'selected_models': [name for name, _ in top_models]
                }
            else:
                raise ValueError("No models could be trained successfully")
                
        except Exception as e:
            logger.error(f"Error training ensemble model: {e}")
            return {
                'success': False,
                'error': str(e),
                'model_scores': model_scores
            }
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using ensemble model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        try:
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Make predictions
            predictions = self.ensemble_model.predict(X_scaled)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            raise
    
    def predict_with_uncertainty(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with uncertainty estimates"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        try:
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Get predictions from each base model
            individual_predictions = []
            
            for name, model in self.ensemble_model.named_estimators_.items():
                pred = model.predict(X_scaled)
                individual_predictions.append(pred)
            
            # Stack predictions
            predictions_array = np.column_stack(individual_predictions)
            
            # Calculate mean and standard deviation
            mean_predictions = np.mean(predictions_array, axis=1)
            std_predictions = np.std(predictions_array, axis=1)
            
            return mean_predictions, std_predictions
            
        except Exception as e:
            logger.error(f"Error making predictions with uncertainty: {e}")
            raise
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Evaluate ensemble model performance"""
        try:
            predictions = self.predict(X)
            
            metrics = {
                'r2_score': r2_score(y, predictions),
                'mean_absolute_error': mean_absolute_error(y, predictions),
                'mean_squared_error': mean_squared_error(y, predictions),
                'root_mean_squared_error': np.sqrt(mean_squared_error(y, predictions)),
                'explained_variance': explained_variance_score(y, predictions)
            }
            
            # Add custom geological metrics
            metrics.update(self._calculate_geological_metrics(y, predictions))
            
            logger.info("Model evaluation completed")
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            raise
    
    def _calculate_geological_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate geological-specific metrics"""
        try:
            # Grade classification accuracy
            grade_bins = [0, 0.1, 0.5, 1.0, 2.0, float('inf')]
            grade_labels = ['very_low', 'low', 'medium', 'high', 'very_high']
            
            y_true_binned = pd.cut(y_true, bins=grade_bins, labels=grade_labels)
            y_pred_binned = pd.cut(y_pred, bins=grade_bins, labels=grade_labels)
            
            classification_accuracy = (y_true_binned == y_pred_binned).mean()
            
            # High-grade detection (>1.0% grade)
            high_grade_threshold = 1.0
            y_true_high = (y_true > high_grade_threshold).astype(int)
            y_pred_high = (y_pred > high_grade_threshold).astype(int)
            
            high_grade_precision = (
                (y_true_high & y_pred_high).sum() / max(y_pred_high.sum(), 1)
            )
            high_grade_recall = (
                (y_true_high & y_pred_high).sum() / max(y_true_high.sum(), 1)
            )
            
            # Resource estimation error
            total_resource_true = y_true.sum()
            total_resource_pred = y_pred.sum()
            resource_error = abs(total_resource_true - total_resource_pred) / total_resource_true
            
            return {
                'grade_classification_accuracy': classification_accuracy,
                'high_grade_precision': high_grade_precision,
                'high_grade_recall': high_grade_recall,
                'resource_estimation_error': resource_error
            }
            
        except Exception as e:
            logger.warning(f"Error calculating geological metrics: {e}")
            return {}
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from ensemble model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")
        
        try:
            # Get feature importance from models that support it
            importance_dict = {}
            
            for name, model in self.ensemble_model.named_estimators_.items():
                if hasattr(model, 'feature_importances_'):
                    importance_dict[name] = model.feature_importances_
                elif hasattr(model, 'coef_'):
                    importance_dict[name] = np.abs(model.coef_)
            
            # Average importance across models
            if importance_dict:
                avg_importance = np.mean(list(importance_dict.values()), axis=0)
                
                feature_importance = dict(zip(self.feature_names, avg_importance))
                
                # Sort by importance
                sorted_importance = dict(
                    sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
                )
                
                return sorted_importance
            else:
                return {}
                
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return {}
    
    def save_model(self, filepath: str) -> bool:
        """Save ensemble model to file"""
        try:
            model_data = {
                'ensemble_model': self.ensemble_model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'is_trained': self.is_trained,
                'config': self.config
            }
            
            joblib.dump(model_data, filepath)
            logger.info(f"Model saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """Load ensemble model from file"""
        try:
            model_data = joblib.load(filepath)
            
            self.ensemble_model = model_data['ensemble_model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.is_trained = model_data['is_trained']
            self.config = model_data.get('config', {})
            
            logger.info(f"Model loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

class StackingEnsemble:
    """Stacking ensemble for ore grade prediction"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.level_1_models = {}
        self.level_2_model = None
        self.scaler = None
        self.is_trained = False
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize models for stacking"""
        # Level 1 models (base models)
        self.level_1_models = {
            'rf': RandomForestRegressor(n_estimators=100, random_state=42),
            'gb': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'et': ExtraTreesRegressor(n_estimators=100, random_state=42),
            'ridge': Ridge(alpha=1.0),
            'svr': SVR(kernel='rbf', C=1.0)
        }
        
        # Level 2 model (meta-learner)
        self.level_2_model = LinearRegression()
        
        logger.info("Stacking models initialized")
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train stacking ensemble"""
        try:
            from sklearn.model_selection import KFold
            from sklearn.preprocessing import StandardScaler
            
            # Scale features
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Create meta-features using cross-validation
            kfold = KFold(n_splits=5, shuffle=True, random_state=42)
            meta_features = np.zeros((X.shape[0], len(self.level_1_models)))
            
            for i, (name, model) in enumerate(self.level_1_models.items()):
                for train_idx, val_idx in kfold.split(X_scaled):
                    X_train_fold = X_scaled[train_idx]
                    y_train_fold = y.iloc[train_idx]
                    X_val_fold = X_scaled[val_idx]
                    
                    # Train model on fold
                    model.fit(X_train_fold, y_train_fold)
                    
                    # Predict on validation fold
                    meta_features[val_idx, i] = model.predict(X_val_fold)
            
            # Train level 1 models on full data
            for name, model in self.level_1_models.items():
                model.fit(X_scaled, y)
            
            # Train level 2 model on meta-features
            self.level_2_model.fit(meta_features, y)
            
            self.is_trained = True
            
            # Evaluate stacking model
            stacking_pred = self.level_2_model.predict(meta_features)
            stacking_score = r2_score(y, stacking_pred)
            
            logger.info(f"Stacking ensemble trained. R² score: {stacking_score:.4f}")
            
            return {
                'success': True,
                'stacking_score': stacking_score,
                'meta_features_shape': meta_features.shape
            }
            
        except Exception as e:
            logger.error(f"Error training stacking ensemble: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using stacking ensemble"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        try:
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Generate meta-features
            meta_features = np.zeros((X.shape[0], len(self.level_1_models)))
            
            for i, (name, model) in enumerate(self.level_1_models.items()):
                meta_features[:, i] = model.predict(X_scaled)
            
            # Make final prediction
            predictions = self.level_2_model.predict(meta_features)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            raise

# Global instances
ensemble_predictor = EnsembleGradePredictor()
stacking_ensemble = StackingEnsemble()
