"""
Manual Prediction Example

This script shows how to manually load and use the trained model files.
"""

import os
import sys
import joblib
import json
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from ml_models.data_processing.feature_engineering import GeologicalFeatureEngineer


def load_latest_model(element="CU"):
    """Load the latest trained model for an element"""
    
    print(f"🔍 LOADING LATEST MODEL FOR {element}")
    print("="*50)
    
    models_dir = Path("data/models")
    
    # Find all model files for this element
    model_files = list(models_dir.glob(f"*model*.joblib"))
    
    if not model_files:
        print(f"❌ No model files found in {models_dir}")
        return None, None, None
    
    # Get the latest model (by timestamp)
    latest_model_file = max(model_files, key=lambda x: x.stat().st_mtime)
    
    print(f"📁 Latest model file: {latest_model_file}")
    
    # Extract timestamp from filename
    timestamp = latest_model_file.stem.split('_')[-2] + '_' + latest_model_file.stem.split('_')[-1]
    
    # Load model files
    model_path = latest_model_file
    scaler_path = models_dir / f"scaler_{timestamp}.joblib"
    metadata_path = models_dir / f"model_metadata_{timestamp}.json"
    
    print(f"📁 Scaler file: {scaler_path}")
    print(f"📁 Metadata file: {metadata_path}")
    
    try:
        # Load model
        model = joblib.load(model_path)
        print("✅ Model loaded successfully")
        
        # Load scaler
        scaler = joblib.load(scaler_path)
        print("✅ Scaler loaded successfully")
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print("✅ Metadata loaded successfully")
        
        # Show model info
        print(f"\n📊 MODEL INFO:")
        print(f"   Element: {metadata.get('element', 'Unknown')}")
        print(f"   Model Type: {metadata.get('model_type', 'Unknown')}")
        print(f"   Training Date: {metadata.get('timestamp', 'Unknown')}")
        print(f"   Training Records: {metadata.get('data_records', 'Unknown')}")
        print(f"   Features: {metadata.get('features_count', 'Unknown')}")
        
        # Show performance
        eval_results = metadata.get('evaluation_results', {})
        test_metrics = eval_results.get('test_metrics', {})
        if test_metrics:
            print(f"   Test R²: {test_metrics.get('r2_score', 'N/A'):.4f}")
            print(f"   Test RMSE: {test_metrics.get('rmse', 'N/A'):.2f}")
            print(f"   Test MAE: {test_metrics.get('mae', 'N/A'):.2f}")
        
        return model, scaler, metadata
        
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return None, None, None


def manual_prediction(latitude, longitude, depth_from, depth_to, element="CU"):
    """Make a manual prediction using loaded model"""
    
    print(f"\n🔮 MANUAL PREDICTION")
    print("="*30)
    
    # Load model
    model, scaler, metadata = load_latest_model(element)
    
    if model is None:
        print("❌ Cannot make prediction - model not loaded")
        return None
    
    print(f"📍 Prediction coordinates:")
    print(f"   Latitude: {latitude}")
    print(f"   Longitude: {longitude}")
    print(f"   Depth: {depth_from} - {depth_to} meters")
    
    try:
        # Create input data
        input_data = pd.DataFrame({
            'latitude': [latitude],
            'longitude': [longitude],
            'depth_from': [depth_from],
            'depth_to': [depth_to],
            'standardized_grade_ppm': [0]  # Dummy value for feature engineering
        })
        
        print("📊 Input data created")
        
        # Feature engineering
        feature_engineer = GeologicalFeatureEngineer()
        features = feature_engineer.create_all_features(input_data)
        
        # Convert categorical columns to codes (same as training)
        for col in features.select_dtypes(include='category').columns:
            features[col] = features[col].cat.codes
        
        # Remove target column
        if 'standardized_grade_ppm' in features.columns:
            features = features.drop(columns=['standardized_grade_ppm'])
        
        print(f"🔧 Features engineered: {len(features.columns)} features")
        
        # Scale features
        features_scaled = scaler.transform(features)
        print("📏 Features scaled")
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        
        print(f"\n🎯 PREDICTION RESULT:")
        print(f"   Predicted Grade: {prediction:.2f} ppm")
        print(f"   Element: {element}")
        
        return {
            'predicted_grade': prediction,
            'element': element,
            'coordinates': {
                'latitude': latitude,
                'longitude': longitude,
                'depth_from': depth_from,
                'depth_to': depth_to
            }
        }
        
    except Exception as e:
        print(f"❌ Error making prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def compare_both_methods():
    """Compare inference system vs manual prediction"""
    
    print("\n🔍 COMPARING PREDICTION METHODS")
    print("="*50)
    
    # Test coordinates
    latitude = -23.550500
    longitude = -46.633300
    depth_from = 50.0
    depth_to = 55.0
    element = "CU"
    
    print(f"📍 Test coordinates:")
    print(f"   ({latitude}, {longitude}) at {depth_from}-{depth_to}m")
    
    # Method 1: Inference system
    try:
        from ml_models.inference.predictor import SpatialOreGradePredictor
        predictor = SpatialOreGradePredictor()
        
        result1 = predictor.predict_at_location(
            latitude=latitude,
            longitude=longitude,
            depth_from=depth_from,
            depth_to=depth_to,
            element=element
        )
        
        print(f"\n✅ Method 1 (Inference System): {result1['predicted_grade']:.2f} ppm")
        
    except Exception as e:
        print(f"❌ Method 1 failed: {str(e)}")
        result1 = None
    
    # Method 2: Manual prediction
    result2 = manual_prediction(latitude, longitude, depth_from, depth_to, element)
    
    if result2:
        print(f"✅ Method 2 (Manual): {result2['predicted_grade']:.2f} ppm")
    else:
        print("❌ Method 2 failed")
    
    # Compare results
    if result1 and result2:
        diff = abs(result1['predicted_grade'] - result2['predicted_grade'])
        print(f"\n📊 Comparison:")
        print(f"   Difference: {diff:.2f} ppm")
        if diff < 0.01:
            print("   ✅ Results are identical (expected)")
        else:
            print("   ⚠️  Results differ (unexpected)")


def main():
    """Main execution function"""
    
    print("🔮 MANUAL PREDICTION EXAMPLES")
    print("="*60)
    
    # Example 1: Load latest model
    model, scaler, metadata = load_latest_model("CU")
    
    if model is not None:
        # Example 2: Manual prediction
        result = manual_prediction(
            latitude=-23.550500,
            longitude=-46.633300,
            depth_from=50.0,
            depth_to=55.0,
            element="CU"
        )
        
        # Example 3: Compare methods
        compare_both_methods()
    
    print("\n🎉 MANUAL PREDICTION EXAMPLES COMPLETED!")
    print("="*60)
    
    print("\n📋 SUMMARY:")
    print("✅ Method 1 (Recommended): Use SpatialOreGradePredictor")
    print("✅ Method 2 (Advanced): Load model files manually")
    print("✅ Both methods should give identical results")
    print("✅ Use Method 1 for production applications")


if __name__ == "__main__":
    main()
