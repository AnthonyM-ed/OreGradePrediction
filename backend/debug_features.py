"""
Debug script to test feature variation in the simple predictor
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from ml_models.inference.simple_predictor import SimpleSpatialPredictor
from ml_models.inference.model_manager import get_model_for_prediction
import pandas as pd

def test_feature_variation():
    """Test if features vary with different inputs"""
    
    # Get model
    models_dir = os.path.join(os.path.dirname(__file__), 'data', 'models')
    model_path = get_model_for_prediction(models_dir, 'CU')
    
    if not model_path:
        print("❌ No model found!")
        return
    
    print(f"📁 Using model: {os.path.basename(model_path)}")
    
    # Initialize predictor
    predictor = SimpleSpatialPredictor(model_path, 'CU')
    
    # Test different coordinates
    test_points = [
        (-19.123456, -69.654321, 0.0, 10.0),
        (-19.200000, -69.700000, 0.0, 10.0),  # Different location
        (-19.100000, -69.600000, 5.0, 15.0),   # Different location and depth
        (-19.150000, -69.680000, 10.0, 20.0),  # Another variation
    ]
    
    print("\n🔬 TESTING FEATURE VARIATION")
    print("=" * 60)
    
    for i, (lat, lon, depth_from, depth_to) in enumerate(test_points):
        print(f"\n📍 Test Point {i+1}: ({lat}, {lon}) at {depth_from}-{depth_to}m")
        
        # Create features for this point
        features = predictor._create_minimal_features(lat, lon, depth_from, depth_to)
        
        print("🔧 Features created:")
        for col in features.columns:
            print(f"   {col}: {features[col].iloc[0]:.6f}")
        
        # Make prediction
        try:
            result = predictor.predict_at_point(lat, lon, depth_from, depth_to)
            predicted_grade = result['predicted_grade_ppm']
            print(f"🎯 Prediction: {predicted_grade:.2f} ppm")
        except Exception as e:
            print(f"❌ Prediction failed: {e}")
    
    print("\n" + "=" * 60)
    print("📊 ANALYSIS:")
    
    # Check if the features are actually different
    all_features = []
    for lat, lon, depth_from, depth_to in test_points:
        features = predictor._create_minimal_features(lat, lon, depth_from, depth_to)
        all_features.append(features.iloc[0].values)
    
    features_df = pd.DataFrame(all_features, columns=features.columns)
    
    print("🔍 Feature variation analysis:")
    for col in features_df.columns:
        unique_values = features_df[col].nunique()
        print(f"   {col}: {unique_values} unique values (range: {features_df[col].min():.6f} to {features_df[col].max():.6f})")
    
    # Check if model metadata has information about features
    print(f"\n📋 Model metadata features: {len(predictor.metadata.get('feature_names', []))} features")
    if predictor.metadata.get('feature_names'):
        print("   Required features:", predictor.metadata['feature_names'][:5], "..." if len(predictor.metadata['feature_names']) > 5 else "")

if __name__ == "__main__":
    test_feature_variation()
