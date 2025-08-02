"""
Model Manager for Spatial Ore Grade Prediction

This module provides utilities to find and load trained models.
"""

import os
import glob
import json
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Manages finding and loading trained ore grade prediction models
    """
    
    def __init__(self, models_dir: str):
        """
        Initialize model manager
        
        Args:
            models_dir: Directory containing trained models
        """
        self.models_dir = Path(models_dir)
        self._available_models = {}
        self._scan_models()
    
    def _scan_models(self):
        """Scan for available trained models"""
        try:
            if not self.models_dir.exists():
                logger.warning(f"Models directory does not exist: {self.models_dir}")
                return
            
            # Find all model files
            model_files = list(self.models_dir.glob("grade_model_*.joblib"))
            
            for model_file in model_files:
                # Extract timestamp from filename
                filename = model_file.stem
                timestamp = filename.replace("grade_model_", "")
                
                # Load metadata if available
                metadata_file = model_file.parent / f"model_metadata_{timestamp}.json"
                metadata = {}
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                    except Exception as e:
                        logger.warning(f"Could not load metadata for {model_file}: {e}")
                
                # Get elements this model can predict
                element = metadata.get('element', None)
                if element:
                    elements = [element]
                else:
                    # Fallback to config.elements or default to CU
                    elements = metadata.get('elements', ['CU'])
                    if isinstance(elements, str):
                        elements = [elements]
                
                # Store model info
                model_info = {
                    'path': str(model_file),
                    'timestamp': timestamp,
                    'metadata': metadata,
                    'elements': elements,
                    'scaler_path': str(model_file.parent / f"scaler_{timestamp}.joblib"),
                    'metadata_path': str(metadata_file) if metadata_file.exists() else None
                }
                
                # Add to available models for each element
                for element in elements:
                    if element not in self._available_models:
                        self._available_models[element] = []
                    self._available_models[element].append(model_info)
            
            # Sort models by timestamp (newest first)
            for element in self._available_models:
                self._available_models[element].sort(
                    key=lambda x: x['timestamp'], 
                    reverse=True
                )
            
            logger.info(f"Found models for elements: {list(self._available_models.keys())}")
            
        except Exception as e:
            logger.error(f"Error scanning models: {e}")
    
    def get_latest_model(self, element: str) -> Optional[Dict]:
        """
        Get the latest model for a specific element
        
        Args:
            element: Element symbol (e.g., 'CU', 'AU', 'AG')
            
        Returns:
            Model info dictionary or None if not found
        """
        element = element.upper()
        
        if element in self._available_models and self._available_models[element]:
            return self._available_models[element][0]  # First is newest
        
        # Try to find a general model if no element-specific model exists
        if 'CU' in self._available_models and self._available_models['CU']:
            logger.warning(f"No model found for {element}, using CU model as fallback")
            return self._available_models['CU'][0]
        
        return None
    
    def get_available_elements(self) -> List[str]:
        """Get list of elements with available models"""
        return list(self._available_models.keys())
    
    def get_all_models_info(self) -> Dict:
        """Get information about all available models"""
        info = {}
        for element, models in self._available_models.items():
            info[element] = []
            for model in models:
                model_summary = {
                    'timestamp': model['timestamp'],
                    'has_metadata': model['metadata_path'] is not None,
                    'accuracy': model['metadata'].get('evaluation_results', {}).get('test_metrics', {}).get('r2_score', 'Unknown'),
                    'features_count': len(model['metadata'].get('feature_names', [])),
                    'training_samples': model['metadata'].get('training_info', {}).get('total_samples', 'Unknown')
                }
                info[element].append(model_summary)
        
        return info
    
    def refresh_models(self):
        """Refresh the model scan"""
        self._available_models = {}
        self._scan_models()


# Helper function for Django views
def get_model_for_prediction(models_dir: str, element: str) -> Optional[str]:
    """
    Get the best model path for predicting a specific element
    
    Args:
        models_dir: Models directory path
        element: Element to predict
        
    Returns:
        Model file path or None if not found
    """
    try:
        manager = ModelManager(models_dir)
        model_info = manager.get_latest_model(element)
        return model_info['path'] if model_info else None
    except Exception as e:
        logger.error(f"Error getting model for {element}: {e}")
        return None
