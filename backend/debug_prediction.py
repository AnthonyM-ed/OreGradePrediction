"""
Test script to debug the prediction system locally
"""

import sys
import os
sys.path.append('d:/Projects/VSCode_Projects/OreGradePrediction/backend')
sys.path.append('d:/Projects/VSCode_Projects/OreGradePrediction/backend/ml_models')

try:
    from ml_models.inference.simple_predictor import SimpleSpatialPredictor
    from ml_models.inference.model_manager import get_model_for_prediction
    
    print("✅ Imports successful")
    
    # Find model
    models_dir = 'd:/Projects/VSCode_Projects/OreGradePrediction/backend/data/models'
    model_path = get_model_for_prediction(models_dir, 'CU')
    
    print(f"📁 Model path: {model_path}")
    
    if model_path:
        # Initialize predictor
        predictor = SimpleSpatialPredictor(model_path, 'CU')
        print("✅ Predictor initialized")
        
        # Test prediction
        result = predictor.predict_at_point(
            latitude=-19.123456,
            longitude=-69.654321,
            depth_from=0.0,
            depth_to=10.0
        )
        
        print(f"🎯 Prediction result: {result}")
        
    else:
        print("❌ No model found")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
