"""
Logging configuration for the ML models system.

This module provides logging setup and utility functions for tracking
model training, predictions, and system performance.
"""

import logging
import sys
import os
from datetime import datetime
from typing import Dict, Any, Optional
import json


def setup_logging(
    log_level: str = 'INFO',
    log_file: str = None,
    log_format: str = None
) -> logging.Logger:
    """
    Setup logging configuration for the ML system.
    
    Args:
        log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        log_file: Optional log file path
        log_format: Custom log format string
        
    Returns:
        Configured logger instance
    """
    if log_format is None:
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[]
    )
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_formatter = logging.Formatter(log_format)
    console_handler.setFormatter(console_formatter)
    
    # Create file handler if log_file is specified
    handlers = [console_handler]
    if log_file:
        # Create log directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_formatter = logging.Formatter(log_format)
        file_handler.setFormatter(file_formatter)
        handlers.append(file_handler)
    
    # Configure root logger with handlers
    root_logger = logging.getLogger()
    root_logger.handlers = handlers
    
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def log_model_performance(
    logger: logging.Logger,
    model_name: str,
    metrics: Dict[str, float],
    training_time: float = None,
    dataset_info: Dict[str, Any] = None
) -> None:
    """
    Log model performance metrics in a structured format.
    
    Args:
        logger: Logger instance
        model_name: Name of the model
        metrics: Dictionary of performance metrics
        training_time: Training time in seconds
        dataset_info: Information about the dataset
    """
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'model_name': model_name,
        'metrics': metrics,
        'training_time_seconds': training_time,
        'dataset_info': dataset_info or {}
    }
    
    logger.info(f"Model Performance - {model_name}")
    logger.info(f"Metrics: {json.dumps(metrics, indent=2)}")
    
    if training_time:
        logger.info(f"Training Time: {training_time:.2f} seconds")
    
    if dataset_info:
        logger.info(f"Dataset Info: {json.dumps(dataset_info, indent=2)}")


def log_prediction_results(
    logger: logging.Logger,
    prediction_id: str,
    input_data: Dict[str, Any],
    predictions: Dict[str, Any],
    execution_time: float = None
) -> None:
    """
    Log prediction results in a structured format.
    
    Args:
        logger: Logger instance
        prediction_id: Unique identifier for the prediction
        input_data: Input data used for prediction
        predictions: Prediction results
        execution_time: Prediction execution time in seconds
    """
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'prediction_id': prediction_id,
        'input_data': input_data,
        'predictions': predictions,
        'execution_time_seconds': execution_time
    }
    
    logger.info(f"Prediction Results - ID: {prediction_id}")
    logger.info(f"Input: {json.dumps(input_data, indent=2)}")
    logger.info(f"Predictions: {json.dumps(predictions, indent=2)}")
    
    if execution_time:
        logger.info(f"Execution Time: {execution_time:.3f} seconds")


def log_data_processing(
    logger: logging.Logger,
    operation: str,
    input_shape: tuple,
    output_shape: tuple,
    processing_time: float = None,
    parameters: Dict[str, Any] = None
) -> None:
    """
    Log data processing operations.
    
    Args:
        logger: Logger instance
        operation: Name of the processing operation
        input_shape: Shape of input data
        output_shape: Shape of output data
        processing_time: Processing time in seconds
        parameters: Operation parameters
    """
    logger.info(f"Data Processing - {operation}")
    logger.info(f"Input Shape: {input_shape}")
    logger.info(f"Output Shape: {output_shape}")
    
    if processing_time:
        logger.info(f"Processing Time: {processing_time:.3f} seconds")
    
    if parameters:
        logger.info(f"Parameters: {json.dumps(parameters, indent=2)}")


def log_error_with_context(
    logger: logging.Logger,
    error: Exception,
    context: Dict[str, Any] = None,
    operation: str = None
) -> None:
    """
    Log errors with additional context information.
    
    Args:
        logger: Logger instance
        error: Exception object
        context: Additional context information
        operation: Operation being performed when error occurred
    """
    error_info = {
        'timestamp': datetime.now().isoformat(),
        'error_type': type(error).__name__,
        'error_message': str(error),
        'operation': operation,
        'context': context or {}
    }
    
    logger.error(f"Error occurred - {type(error).__name__}: {str(error)}")
    
    if operation:
        logger.error(f"Operation: {operation}")
    
    if context:
        logger.error(f"Context: {json.dumps(context, indent=2)}")


def log_system_resource_usage(
    logger: logging.Logger,
    operation: str = None
) -> None:
    """
    Log system resource usage (if psutil is available).
    
    Args:
        logger: Logger instance
        operation: Operation being monitored
    """
    try:
        import psutil
        
        # Get CPU and memory usage
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        resource_info = {
            'timestamp': datetime.now().isoformat(),
            'operation': operation,
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_available_mb': memory.available / (1024 * 1024),
            'memory_total_mb': memory.total / (1024 * 1024)
        }
        
        logger.info(f"System Resources - {operation or 'General'}")
        logger.info(f"CPU Usage: {cpu_percent}%")
        logger.info(f"Memory Usage: {memory.percent}%")
        logger.info(f"Available Memory: {memory.available / (1024 * 1024):.1f} MB")
        
    except ImportError:
        logger.warning("psutil not available for resource monitoring")


def create_structured_log_entry(
    level: str,
    message: str,
    category: str = None,
    metadata: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Create a structured log entry for JSON logging.
    
    Args:
        level: Log level
        message: Log message
        category: Log category
        metadata: Additional metadata
        
    Returns:
        Structured log entry
    """
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'level': level,
        'message': message,
        'category': category or 'general',
        'metadata': metadata or {}
    }
    
    return log_entry


class ModelPerformanceLogger:
    """
    Specialized logger for tracking model performance over time.
    """
    
    def __init__(self, logger_name: str = 'model_performance'):
        self.logger = get_logger(logger_name)
        self.performance_history = []
    
    def log_training_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float = None,
        metrics: Dict[str, float] = None
    ) -> None:
        """
        Log training epoch performance.
        
        Args:
            epoch: Epoch number
            train_loss: Training loss
            val_loss: Validation loss
            metrics: Additional metrics
        """
        epoch_info = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'metrics': metrics or {},
            'timestamp': datetime.now().isoformat()
        }
        
        self.performance_history.append(epoch_info)
        
        self.logger.info(f"Epoch {epoch} - Train Loss: {train_loss:.4f}")
        if val_loss:
            self.logger.info(f"Epoch {epoch} - Val Loss: {val_loss:.4f}")
        
        if metrics:
            metric_str = ', '.join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            self.logger.info(f"Epoch {epoch} - Metrics: {metric_str}")
    
    def log_final_performance(
        self,
        model_name: str,
        final_metrics: Dict[str, float],
        best_epoch: int = None
    ) -> None:
        """
        Log final model performance.
        
        Args:
            model_name: Name of the model
            final_metrics: Final performance metrics
            best_epoch: Best performing epoch
        """
        self.logger.info(f"Final Performance - {model_name}")
        self.logger.info(f"Final Metrics: {json.dumps(final_metrics, indent=2)}")
        
        if best_epoch:
            self.logger.info(f"Best Epoch: {best_epoch}")
    
    def get_performance_history(self) -> list:
        """
        Get the complete performance history.
        
        Returns:
            List of performance entries
        """
        return self.performance_history.copy()
    
    def export_performance_log(self, filepath: str) -> None:
        """
        Export performance history to JSON file.
        
        Args:
            filepath: Path to export file
        """
        with open(filepath, 'w') as f:
            json.dump(self.performance_history, f, indent=2)
        
        self.logger.info(f"Performance history exported to {filepath}")


# Configure default logger for the ML models module
default_logger = setup_logging(
    log_level='INFO',
    log_file=None,  # Set to file path if needed
    log_format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
