"""
Advanced statistics utilities for model evaluation and analysis.

This module provides functions for advanced statistical analysis, bootstrap
confidence intervals, hypothesis testing, and distribution analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Callable
from scipy import stats
from scipy.stats import bootstrap
import logging

logger = logging.getLogger(__name__)


def calculate_advanced_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_std: np.ndarray = None
) -> Dict[str, float]:
    """
    Calculate advanced evaluation metrics for model performance.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        y_pred_std: Prediction standard deviations (optional)
        
    Returns:
        Dictionary with advanced metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['mse'] = float(np.mean((y_true - y_pred) ** 2))
    metrics['rmse'] = float(np.sqrt(metrics['mse']))
    metrics['mae'] = float(np.mean(np.abs(y_true - y_pred)))
    metrics['r2'] = float(1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
    
    # Advanced metrics
    metrics['mape'] = float(np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-10))) * 100)
    metrics['smape'] = float(np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))) * 100)
    
    # Bias metrics
    metrics['mean_bias'] = float(np.mean(y_pred - y_true))
    metrics['bias_percentage'] = float(metrics['mean_bias'] / np.mean(y_true) * 100)
    
    # Correlation metrics
    metrics['pearson_r'] = float(np.corrcoef(y_true, y_pred)[0, 1])
    metrics['spearman_r'] = float(stats.spearmanr(y_true, y_pred)[0])
    
    # Residual analysis
    residuals = y_pred - y_true
    metrics['residual_mean'] = float(np.mean(residuals))
    metrics['residual_std'] = float(np.std(residuals))
    metrics['residual_skewness'] = float(stats.skew(residuals))
    metrics['residual_kurtosis'] = float(stats.kurtosis(residuals))
    
    # Prediction interval metrics (if std provided)
    if y_pred_std is not None:
        # 95% prediction intervals
        lower_bound = y_pred - 1.96 * y_pred_std
        upper_bound = y_pred + 1.96 * y_pred_std
        
        # Coverage probability
        coverage = np.mean((y_true >= lower_bound) & (y_true <= upper_bound))
        metrics['coverage_95'] = float(coverage)
        
        # Mean prediction interval width
        metrics['mean_pi_width'] = float(np.mean(upper_bound - lower_bound))
        
        # Prediction interval normalized width
        metrics['pi_width_normalized'] = float(metrics['mean_pi_width'] / np.std(y_true))
    
    # Quantile-based metrics
    for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
        error_quantile = np.quantile(np.abs(residuals), q)
        metrics[f'abs_error_q{int(q*100)}'] = float(error_quantile)
    
    return metrics


def bootstrap_confidence_intervals(
    data: np.ndarray,
    statistic: Callable,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42
) -> Dict[str, float]:
    """
    Calculate bootstrap confidence intervals for a statistic.
    
    Args:
        data: Input data array
        statistic: Function to calculate statistic
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (0-1)
        random_state: Random seed
        
    Returns:
        Dictionary with statistic and confidence intervals
    """
    np.random.seed(random_state)
    
    # Calculate original statistic
    original_stat = statistic(data)
    
    # Bootstrap sampling
    bootstrap_stats = []
    n_samples = len(data)
    
    for _ in range(n_bootstrap):
        # Sample with replacement
        bootstrap_sample = np.random.choice(data, size=n_samples, replace=True)
        bootstrap_stat = statistic(bootstrap_sample)
        bootstrap_stats.append(bootstrap_stat)
    
    bootstrap_stats = np.array(bootstrap_stats)
    
    # Calculate confidence intervals
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    ci_lower = np.percentile(bootstrap_stats, lower_percentile)
    ci_upper = np.percentile(bootstrap_stats, upper_percentile)
    
    return {
        'statistic': float(original_stat),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'bootstrap_mean': float(np.mean(bootstrap_stats)),
        'bootstrap_std': float(np.std(bootstrap_stats)),
        'confidence_level': confidence_level
    }


def statistical_tests(
    data1: np.ndarray,
    data2: np.ndarray = None,
    test_type: str = 'normality'
) -> Dict[str, Union[float, bool]]:
    """
    Perform various statistical tests.
    
    Args:
        data1: First dataset
        data2: Second dataset (for two-sample tests)
        test_type: Type of test ('normality', 'ttest', 'mannwhitney', 'ks')
        
    Returns:
        Dictionary with test results
    """
    results = {}
    
    if test_type == 'normality':
        # Shapiro-Wilk test for normality
        if len(data1) <= 5000:  # Shapiro-Wilk has sample size limits
            stat, p_value = stats.shapiro(data1)
            results['shapiro_stat'] = float(stat)
            results['shapiro_p_value'] = float(p_value)
            results['is_normal_shapiro'] = bool(p_value > 0.05)
        
        # Kolmogorov-Smirnov test for normality
        stat, p_value = stats.kstest(data1, 'norm', args=(np.mean(data1), np.std(data1)))
        results['ks_stat'] = float(stat)
        results['ks_p_value'] = float(p_value)
        results['is_normal_ks'] = bool(p_value > 0.05)
        
        # Anderson-Darling test
        stat, critical_values, significance_levels = stats.anderson(data1, dist='norm')
        results['anderson_stat'] = float(stat)
        results['anderson_5pct_critical'] = float(critical_values[2])  # 5% significance level
        results['is_normal_anderson'] = bool(stat < critical_values[2])
    
    elif test_type == 'ttest' and data2 is not None:
        # Two-sample t-test
        stat, p_value = stats.ttest_ind(data1, data2)
        results['ttest_stat'] = float(stat)
        results['ttest_p_value'] = float(p_value)
        results['means_different'] = bool(p_value < 0.05)
        
        # Welch's t-test (unequal variances)
        stat, p_value = stats.ttest_ind(data1, data2, equal_var=False)
        results['welch_stat'] = float(stat)
        results['welch_p_value'] = float(p_value)
        results['means_different_welch'] = bool(p_value < 0.05)
    
    elif test_type == 'mannwhitney' and data2 is not None:
        # Mann-Whitney U test
        stat, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
        results['mannwhitney_stat'] = float(stat)
        results['mannwhitney_p_value'] = float(p_value)
        results['distributions_different'] = bool(p_value < 0.05)
    
    elif test_type == 'ks' and data2 is not None:
        # Kolmogorov-Smirnov two-sample test
        stat, p_value = stats.ks_2samp(data1, data2)
        results['ks_2samp_stat'] = float(stat)
        results['ks_2samp_p_value'] = float(p_value)
        results['distributions_different_ks'] = bool(p_value < 0.05)
    
    return results


def outlier_detection(
    data: np.ndarray,
    method: str = 'iqr',
    contamination: float = 0.1
) -> Dict[str, Union[np.ndarray, float]]:
    """
    Detect outliers using various methods.
    
    Args:
        data: Input data array
        method: Detection method ('iqr', 'zscore', 'modified_zscore', 'isolation_forest')
        contamination: Expected proportion of outliers
        
    Returns:
        Dictionary with outlier detection results
    """
    results = {}
    
    if method == 'iqr':
        # Interquartile Range method
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = (data < lower_bound) | (data > upper_bound)
        results['outlier_mask'] = outliers
        results['lower_bound'] = float(lower_bound)
        results['upper_bound'] = float(upper_bound)
        results['n_outliers'] = int(np.sum(outliers))
        results['outlier_percentage'] = float(np.mean(outliers) * 100)
    
    elif method == 'zscore':
        # Z-score method
        z_scores = np.abs(stats.zscore(data))
        threshold = 3.0
        
        outliers = z_scores > threshold
        results['outlier_mask'] = outliers
        results['z_scores'] = z_scores
        results['threshold'] = threshold
        results['n_outliers'] = int(np.sum(outliers))
        results['outlier_percentage'] = float(np.mean(outliers) * 100)
    
    elif method == 'modified_zscore':
        # Modified Z-score using median absolute deviation
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        modified_z_scores = 0.6745 * (data - median) / mad
        threshold = 3.5
        
        outliers = np.abs(modified_z_scores) > threshold
        results['outlier_mask'] = outliers
        results['modified_z_scores'] = modified_z_scores
        results['threshold'] = threshold
        results['n_outliers'] = int(np.sum(outliers))
        results['outlier_percentage'] = float(np.mean(outliers) * 100)
    
    elif method == 'isolation_forest':
        # Isolation Forest method
        try:
            from sklearn.ensemble import IsolationForest
            
            iso_forest = IsolationForest(contamination=contamination, random_state=42)
            outliers = iso_forest.fit_predict(data.reshape(-1, 1)) == -1
            scores = iso_forest.decision_function(data.reshape(-1, 1))
            
            results['outlier_mask'] = outliers
            results['anomaly_scores'] = scores
            results['contamination'] = contamination
            results['n_outliers'] = int(np.sum(outliers))
            results['outlier_percentage'] = float(np.mean(outliers) * 100)
        except ImportError:
            logger.warning("scikit-learn not available for isolation forest")
            return outlier_detection(data, method='iqr')
    
    return results


def distribution_analysis(
    data: np.ndarray,
    distributions: List[str] = None
) -> Dict[str, Dict[str, float]]:
    """
    Analyze data distribution and fit various distributions.
    
    Args:
        data: Input data array
        distributions: List of distribution names to fit
        
    Returns:
        Dictionary with distribution fitting results
    """
    if distributions is None:
        distributions = ['norm', 'lognorm', 'gamma', 'exponential', 'weibull_min']
    
    results = {}
    
    # Basic distribution statistics
    results['basic_stats'] = {
        'mean': float(np.mean(data)),
        'std': float(np.std(data)),
        'skewness': float(stats.skew(data)),
        'kurtosis': float(stats.kurtosis(data)),
        'min': float(np.min(data)),
        'max': float(np.max(data)),
        'median': float(np.median(data)),
        'q25': float(np.percentile(data, 25)),
        'q75': float(np.percentile(data, 75))
    }
    
    # Fit distributions
    for dist_name in distributions:
        try:
            distribution = getattr(stats, dist_name)
            params = distribution.fit(data)
            
            # Calculate goodness of fit
            ks_stat, ks_p_value = stats.kstest(data, distribution.cdf, args=params)
            
            # Calculate AIC and BIC
            log_likelihood = np.sum(distribution.logpdf(data, *params))
            k = len(params)  # Number of parameters
            n = len(data)   # Sample size
            
            aic = 2 * k - 2 * log_likelihood
            bic = k * np.log(n) - 2 * log_likelihood
            
            results[dist_name] = {
                'parameters': [float(p) for p in params],
                'ks_statistic': float(ks_stat),
                'ks_p_value': float(ks_p_value),
                'log_likelihood': float(log_likelihood),
                'aic': float(aic),
                'bic': float(bic),
                'fits_well': bool(ks_p_value > 0.05)
            }
            
        except Exception as e:
            logger.warning(f"Failed to fit {dist_name} distribution: {e}")
            results[dist_name] = {'error': str(e)}
    
    # Find best fitting distribution
    valid_dists = {name: result for name, result in results.items() 
                   if isinstance(result, dict) and 'aic' in result}
    
    if valid_dists:
        best_dist = min(valid_dists.keys(), key=lambda x: valid_dists[x]['aic'])
        results['best_distribution'] = {
            'name': best_dist,
            'aic': valid_dists[best_dist]['aic'],
            'bic': valid_dists[best_dist]['bic']
        }
    
    return results


def time_series_analysis(
    data: np.ndarray,
    timestamps: np.ndarray = None
) -> Dict[str, float]:
    """
    Perform basic time series analysis.
    
    Args:
        data: Time series data
        timestamps: Optional timestamps
        
    Returns:
        Dictionary with time series statistics
    """
    results = {}
    
    # Basic trend analysis
    if timestamps is not None:
        # Linear trend
        slope, intercept, r_value, p_value, std_err = stats.linregress(timestamps, data)
        results['trend_slope'] = float(slope)
        results['trend_r_squared'] = float(r_value ** 2)
        results['trend_p_value'] = float(p_value)
        results['trend_significant'] = bool(p_value < 0.05)
    
    # Autocorrelation
    def autocorrelation(x, max_lag=min(50, len(data)//4)):
        n = len(x)
        x = x - np.mean(x)
        autocorr = np.correlate(x, x, mode='full')
        autocorr = autocorr[n-1:]
        autocorr = autocorr / autocorr[0]
        return autocorr[:max_lag]
    
    autocorr = autocorrelation(data)
    results['autocorrelation'] = autocorr.tolist()
    
    # First-order autocorrelation
    if len(autocorr) > 1:
        results['autocorr_lag1'] = float(autocorr[1])
    
    # Variance analysis
    results['variance'] = float(np.var(data))
    results['coefficient_of_variation'] = float(np.std(data) / np.mean(data))
    
    return results
