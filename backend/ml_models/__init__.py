"""
ML Models Module for Ore Grade Prediction
=========================================

This module contains all machine learning components for geological data processing
and ore grade prediction, including:

- Data processing and feature engineering
- Database management and connections
- ML model implementations
- Training and evaluation pipelines
- Inference and prediction services
- Caching and optimization utilities

Author: OreGradePrediction Team
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "OreGradePrediction Team"

# Setup Django before importing any modules that access Django settings
from .django_setup import setup_django
setup_django()

# Import main components
from .data_processing import *
from .database import *
from .models import *
from .utils import *
