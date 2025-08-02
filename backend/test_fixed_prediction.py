"""
Test Prediction with Fixed Model

This script tests predictions using the newly trained model with working scaler.
"""

import os
import sys
import requests
import json

# Add backend to path
sys.path.append(os.path.dirname(__file__))

from ml_models.inference.simple_predictor import SimpleSpatialPredictor

def test_fixed_prediction():
    """Test predictions with the fixed model"""
    print("🧪 TESTING PREDICTIONS WITH FIXED MODEL")
    print("="*60)
    
    # Use the latest model (with working scaler)
    model_path = "data/models/grade_model_20250717_232857.joblib"
    
    if not os.path.exists(model_path):
        print("❌ Fixed model not found!")
        return
    
    try:
        # Initialize predictor with new model
        predictor = SimpleSpatialPredictor(model_path, "CU")
        
        # Test different coordinates
        test_points = [
            (-19.123456, -69.654321, 0.0, 10.0, "Point 1"),
            (-19.200000, -69.700000, 0.0, 10.0, "Point 2"), 
            (-19.100000, -69.600000, 5.0, 15.0, "Point 3"),
            (-19.150000, -69.680000, 10.0, 20.0, "Point 4"),
        ]
        
        print("📍 TESTING SPATIAL VARIATION:")
        print()
        
        for lat, lon, depth_from, depth_to, name in test_points:
            result = predictor.predict_at_point(lat, lon, depth_from, depth_to)
            grade = result['predicted_grade_ppm']
            
            print(f"🎯 {name}: ({lat:.3f}, {lon:.3f}) at {depth_from}-{depth_to}m")
            print(f"   Grade: {grade:.2f} ppm")
            print(f"   Confidence: {result['confidence_interval']['lower_bound']:.1f} - {result['confidence_interval']['upper_bound']:.1f} ppm")
            print()
        
        print("✅ Predictions completed successfully!")
        print("🔍 Check if predictions vary with coordinates...")
        
        # Check variation
        grades = []
        for lat, lon, depth_from, depth_to, name in test_points:
            result = predictor.predict_at_point(lat, lon, depth_from, depth_to)
            grades.append(result['predicted_grade_ppm'])
        
        unique_grades = len(set([round(g, 1) for g in grades]))
        print(f"📊 Got {unique_grades} unique grade values out of {len(grades)} predictions")
        
        if unique_grades > 1:
            print("🎉 SUCCESS: Predictions are varying spatially!")
        else:
            print("⚠️  WARNING: All predictions are still the same")
            
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_fixed_prediction()
