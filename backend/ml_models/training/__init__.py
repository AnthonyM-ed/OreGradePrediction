"""
Training Module for Ore Grade Prediction

This module provides comprehensive training functionality including:
- Complete training pipeline orchestration
- Cross-validation with multiple metrics (RMSE, R², MAE)
- Hyperparameter tuning (Grid Search, Random Search, Bayesian Optimization)
- Model evaluation and comparison
"""

from .train_pipeline import TrainingPipeline
from .cross_validation import CrossValidator
from .hyperparameter_tuning import XGBoostHyperparameterTuner

__all__ = [
    'TrainingPipeline',
    'CrossValidator', 
    'XGBoostHyperparameterTuner'
]

__version__ = '1.0.0'
