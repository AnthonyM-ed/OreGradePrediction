"""
Hyperparameter Tuning Module for XGBoost Ore Grade Prediction

This module provides automated hyperparameter tuning for XGBoost models
using Grid Search, Random Search, and Bayesian Optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
import logging
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
    from sklearn.metrics import make_scorer, mean_squared_error, r2_score
    import xgboost as xgb
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn or xgboost not available. Some functionality may be limited.")

try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    logger.warning("scikit-optimize not available. Bayesian optimization will not be available.")


class XGBoostHyperparameterTuner:
    """
    Hyperparameter tuning for XGBoost models with geological data focus
    """
    
    def __init__(self, random_state: int = 42, n_jobs: int = -1):
        """
        Initialize hyperparameter tuner
        
        Args:
            random_state: Random state for reproducibility
            n_jobs: Number of parallel jobs
        """
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.tuning_results = {}
        self.best_params = {}
        self.best_score = None
        
    def get_default_param_grid(self) -> Dict[str, List[Any]]:
        """
        Get default parameter grid for XGBoost
        
        Returns:
            Parameter grid dictionary
        """
        return {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [3, 4, 5, 6, 7, 8],
            'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.1, 0.5, 1.0],
            'reg_lambda': [0, 0.1, 0.5, 1.0, 2.0],
            'min_child_weight': [1, 3, 5, 7],
            'gamma': [0, 0.1, 0.2, 0.5]
        }
    
    def get_quick_param_grid(self) -> Dict[str, List[Any]]:
        """
        Get quick parameter grid for faster tuning
        
        Returns:
            Reduced parameter grid
        """
        return {
            'n_estimators': [100, 200, 300],
            'max_depth': [4, 5, 6],
            'learning_rate': [0.05, 0.1, 0.15],
            'subsample': [0.8, 0.9],
            'colsample_bytree': [0.8, 0.9],
            'reg_alpha': [0, 0.1],
            'reg_lambda': [0, 1.0]
        }
    
    def get_geological_param_grid(self) -> Dict[str, List[Any]]:
        """
        Get parameter grid optimized for geological data
        
        Returns:
            Geological-optimized parameter grid
        """
        return {
            'n_estimators': [200, 300, 500],  # More trees for complex geological patterns
            'max_depth': [5, 6, 7, 8],        # Deeper trees for spatial relationships
            'learning_rate': [0.05, 0.1, 0.15],  # Conservative learning rates
            'subsample': [0.8, 0.9],          # Reduce overfitting
            'colsample_bytree': [0.7, 0.8, 0.9],  # Feature sampling
            'reg_alpha': [0.1, 0.5, 1.0],     # L1 regularization
            'reg_lambda': [1.0, 2.0, 5.0],    # L2 regularization
            'min_child_weight': [3, 5, 7],    # Minimum samples per leaf
            'gamma': [0.1, 0.2, 0.5]          # Minimum loss reduction
        }
    
    def get_bayesian_search_space(self) -> Dict[str, Any]:
        """
        Get search space for Bayesian optimization
        
        Returns:
            Bayesian search space
        """
        if not SKOPT_AVAILABLE:
            logger.error("scikit-optimize not available for Bayesian optimization")
            return {}
        
        return {
            'n_estimators': Integer(100, 1000),
            'max_depth': Integer(3, 10),
            'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
            'subsample': Real(0.6, 1.0),
            'colsample_bytree': Real(0.6, 1.0),
            'reg_alpha': Real(0, 10, prior='log-uniform'),
            'reg_lambda': Real(0, 10, prior='log-uniform'),
            'min_child_weight': Integer(1, 10),
            'gamma': Real(0, 5)
        }
    
    def grid_search_tuning(self, X: np.ndarray, y: np.ndarray, 
                          param_grid: Optional[Dict[str, List[Any]]] = None,
                          cv: int = 5, scoring: str = 'neg_mean_squared_error') -> Dict[str, Any]:
        """
        Perform grid search hyperparameter tuning
        
        Args:
            X: Feature matrix
            y: Target values
            param_grid: Parameter grid (uses default if None)
            cv: Cross-validation folds
            scoring: Scoring metric
            
        Returns:
            Grid search results
        """
        if not SKLEARN_AVAILABLE:
            logger.error("scikit-learn not available for grid search")
            return {}
        
        try:
            logger.info("Starting Grid Search hyperparameter tuning")
            start_time = datetime.now()
            
            # Use default parameter grid if none provided
            if param_grid is None:
                param_grid = self.get_quick_param_grid()
            
            # Initialize XGBoost model
            xgb_model = xgb.XGBRegressor(
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                verbosity=0
            )
            
            # Perform grid search
            grid_search = GridSearchCV(
                estimator=xgb_model,
                param_grid=param_grid,
                cv=cv,
                scoring=scoring,
                n_jobs=self.n_jobs,
                verbose=1,
                return_train_score=True
            )
            
            grid_search.fit(X, y)
            
            # Calculate tuning time
            tuning_time = datetime.now() - start_time
            
            # Store results
            results = {
                'method': 'GridSearch',
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'best_estimator': grid_search.best_estimator_,
                'cv_results': grid_search.cv_results_,
                'tuning_time_seconds': tuning_time.total_seconds(),
                'param_grid': param_grid,
                'cv_folds': cv,
                'scoring': scoring,
                'n_combinations': len(grid_search.cv_results_['params'])
            }
            
            # Update instance variables
            self.best_params = grid_search.best_params_
            self.best_score = grid_search.best_score_
            self.tuning_results['grid_search'] = results
            
            logger.info(f"Grid Search completed in {tuning_time.total_seconds():.2f} seconds")
            logger.info(f"Best RMSE: {np.sqrt(-grid_search.best_score_):.4f}")
            logger.info(f"Best parameters: {grid_search.best_params_}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in grid search tuning: {str(e)}")
            return {}
    
    def random_search_tuning(self, X: np.ndarray, y: np.ndarray,
                           param_grid: Optional[Dict[str, List[Any]]] = None,
                           n_iter: int = 100, cv: int = 5, 
                           scoring: str = 'neg_mean_squared_error') -> Dict[str, Any]:
        """
        Perform random search hyperparameter tuning
        
        Args:
            X: Feature matrix
            y: Target values
            param_grid: Parameter grid (uses default if None)
            n_iter: Number of iterations
            cv: Cross-validation folds
            scoring: Scoring metric
            
        Returns:
            Random search results
        """
        if not SKLEARN_AVAILABLE:
            logger.error("scikit-learn not available for random search")
            return {}
        
        try:
            logger.info(f"Starting Random Search hyperparameter tuning with {n_iter} iterations")
            start_time = datetime.now()
            
            # Use default parameter grid if none provided
            if param_grid is None:
                param_grid = self.get_default_param_grid()
            
            # Initialize XGBoost model
            xgb_model = xgb.XGBRegressor(
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                verbosity=0
            )
            
            # Perform random search
            random_search = RandomizedSearchCV(
                estimator=xgb_model,
                param_distributions=param_grid,
                n_iter=n_iter,
                cv=cv,
                scoring=scoring,
                n_jobs=self.n_jobs,
                verbose=1,
                random_state=self.random_state,
                return_train_score=True
            )
            
            random_search.fit(X, y)
            
            # Calculate tuning time
            tuning_time = datetime.now() - start_time
            
            # Store results
            results = {
                'method': 'RandomizedSearch',
                'best_params': random_search.best_params_,
                'best_score': random_search.best_score_,
                'best_estimator': random_search.best_estimator_,
                'cv_results': random_search.cv_results_,
                'tuning_time_seconds': tuning_time.total_seconds(),
                'param_grid': param_grid,
                'n_iter': n_iter,
                'cv_folds': cv,
                'scoring': scoring
            }
            
            # Update instance variables
            self.best_params = random_search.best_params_
            self.best_score = random_search.best_score_
            self.tuning_results['random_search'] = results
            
            logger.info(f"Random Search completed in {tuning_time.total_seconds():.2f} seconds")
            logger.info(f"Best RMSE: {np.sqrt(-random_search.best_score_):.4f}")
            logger.info(f"Best parameters: {random_search.best_params_}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in random search tuning: {str(e)}")
            return {}
    
    def bayesian_optimization_tuning(self, X: np.ndarray, y: np.ndarray,
                                   n_calls: int = 50, cv: int = 5,
                                   scoring: str = 'neg_mean_squared_error') -> Dict[str, Any]:
        """
        Perform Bayesian optimization hyperparameter tuning
        
        Args:
            X: Feature matrix
            y: Target values
            n_calls: Number of optimization calls
            cv: Cross-validation folds
            scoring: Scoring metric
            
        Returns:
            Bayesian optimization results
        """
        if not SKOPT_AVAILABLE:
            logger.error("scikit-optimize not available for Bayesian optimization")
            return {}
        
        try:
            logger.info(f"Starting Bayesian Optimization with {n_calls} calls")
            start_time = datetime.now()
            
            # Get search space
            search_space = self.get_bayesian_search_space()
            
            # Initialize XGBoost model
            xgb_model = xgb.XGBRegressor(
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                verbosity=0
            )
            
            # Perform Bayesian optimization
            bayes_search = BayesSearchCV(
                estimator=xgb_model,
                search_spaces=search_space,
                n_iter=n_calls,
                cv=cv,
                scoring=scoring,
                n_jobs=self.n_jobs,
                verbose=1,
                random_state=self.random_state,
                return_train_score=True
            )
            
            bayes_search.fit(X, y)
            
            # Calculate tuning time
            tuning_time = datetime.now() - start_time
            
            # Store results
            results = {
                'method': 'BayesianOptimization',
                'best_params': bayes_search.best_params_,
                'best_score': bayes_search.best_score_,
                'best_estimator': bayes_search.best_estimator_,
                'cv_results': bayes_search.cv_results_,
                'tuning_time_seconds': tuning_time.total_seconds(),
                'search_space': search_space,
                'n_calls': n_calls,
                'cv_folds': cv,
                'scoring': scoring
            }
            
            # Update instance variables
            self.best_params = bayes_search.best_params_
            self.best_score = bayes_search.best_score_
            self.tuning_results['bayesian_optimization'] = results
            
            logger.info(f"Bayesian Optimization completed in {tuning_time.total_seconds():.2f} seconds")
            logger.info(f"Best RMSE: {np.sqrt(-bayes_search.best_score_):.4f}")
            logger.info(f"Best parameters: {bayes_search.best_params_}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in Bayesian optimization: {str(e)}")
            return {}
    
    def progressive_tuning(self, X: np.ndarray, y: np.ndarray, 
                          stages: List[str] = ['quick_grid', 'random', 'bayesian'],
                          cv: int = 5) -> Dict[str, Any]:
        """
        Perform progressive hyperparameter tuning
        
        Args:
            X: Feature matrix
            y: Target values
            stages: List of tuning stages
            cv: Cross-validation folds
            
        Returns:
            Progressive tuning results
        """
        try:
            logger.info("Starting progressive hyperparameter tuning")
            start_time = datetime.now()
            
            all_results = {}
            
            for stage in stages:
                logger.info(f"Starting stage: {stage}")
                
                if stage == 'quick_grid':
                    result = self.grid_search_tuning(
                        X, y, 
                        param_grid=self.get_quick_param_grid(),
                        cv=cv
                    )
                    all_results['quick_grid'] = result
                
                elif stage == 'random':
                    result = self.random_search_tuning(
                        X, y,
                        param_grid=self.get_default_param_grid(),
                        n_iter=100,
                        cv=cv
                    )
                    all_results['random_search'] = result
                
                elif stage == 'bayesian':
                    result = self.bayesian_optimization_tuning(
                        X, y,
                        n_calls=50,
                        cv=cv
                    )
                    all_results['bayesian_optimization'] = result
                
                elif stage == 'geological':
                    result = self.grid_search_tuning(
                        X, y,
                        param_grid=self.get_geological_param_grid(),
                        cv=cv
                    )
                    all_results['geological_grid'] = result
            
            # Calculate total tuning time
            total_time = datetime.now() - start_time
            
            # Find overall best result
            best_result = None
            best_score = float('-inf')
            
            for stage_name, result in all_results.items():
                if result and result.get('best_score', float('-inf')) > best_score:
                    best_score = result['best_score']
                    best_result = result
            
            progressive_results = {
                'method': 'Progressive',
                'stages': stages,
                'stage_results': all_results,
                'overall_best_params': best_result['best_params'] if best_result else None,
                'overall_best_score': best_score,
                'total_tuning_time_seconds': total_time.total_seconds(),
                'cv_folds': cv
            }
            
            self.tuning_results['progressive'] = progressive_results
            
            if best_result:
                self.best_params = best_result['best_params']
                self.best_score = best_score
                
                logger.info(f"Progressive tuning completed in {total_time.total_seconds():.2f} seconds")
                logger.info(f"Overall best RMSE: {np.sqrt(-best_score):.4f}")
                logger.info(f"Overall best parameters: {best_result['best_params']}")
            
            return progressive_results
            
        except Exception as e:
            logger.error(f"Error in progressive tuning: {str(e)}")
            return {}
    
    def save_tuning_results(self, filepath: str):
        """
        Save tuning results to file
        
        Args:
            filepath: Path to save results
        """
        try:
            # Prepare results for JSON serialization
            serializable_results = {}
            
            for method, results in self.tuning_results.items():
                serializable_results[method] = {
                    'method': results.get('method', method),
                    'best_params': results.get('best_params', {}),
                    'best_score': results.get('best_score', None),
                    'tuning_time_seconds': results.get('tuning_time_seconds', None),
                    'timestamp': datetime.now().isoformat()
                }
            
            with open(filepath, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            logger.info(f"Tuning results saved to: {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving tuning results: {str(e)}")
    
    def load_tuning_results(self, filepath: str) -> Dict[str, Any]:
        """
        Load tuning results from file
        
        Args:
            filepath: Path to load results from
            
        Returns:
            Loaded tuning results
        """
        try:
            with open(filepath, 'r') as f:
                results = json.load(f)
            
            logger.info(f"Tuning results loaded from: {filepath}")
            return results
            
        except Exception as e:
            logger.error(f"Error loading tuning results: {str(e)}")
            return {}
    
    def get_tuning_summary(self) -> str:
        """
        Get summary of all tuning results
        
        Returns:
            Summary string
        """
        if not self.tuning_results:
            return "No tuning results available"
        
        summary_lines = []
        summary_lines.append("HYPERPARAMETER TUNING SUMMARY")
        summary_lines.append("=" * 50)
        
        for method, results in self.tuning_results.items():
            summary_lines.append(f"\n{method.upper()}:")
            summary_lines.append("-" * 20)
            
            if 'best_score' in results:
                rmse = np.sqrt(-results['best_score'])
                summary_lines.append(f"Best RMSE: {rmse:.4f}")
            
            if 'tuning_time_seconds' in results:
                summary_lines.append(f"Time: {results['tuning_time_seconds']:.2f} seconds")
            
            if 'best_params' in results:
                summary_lines.append("Best Parameters:")
                for param, value in results['best_params'].items():
                    summary_lines.append(f"  {param}: {value}")
        
        return "\n".join(summary_lines)


# Example usage
if __name__ == "__main__":
    logger.info("Hyperparameter tuning module loaded successfully")
    
    # Example initialization
    tuner = XGBoostHyperparameterTuner(random_state=42)
    
    print("Hyperparameter tuning tools ready!")
    print("Available methods:")
    print("- grid_search_tuning()")
    print("- random_search_tuning()")
    print("- bayesian_optimization_tuning()")
    print("- progressive_tuning()")
    print("- save_tuning_results()")
    print("- get_tuning_summary()")
