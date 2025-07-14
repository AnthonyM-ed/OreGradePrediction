"""
Simple Prediction Example

This script shows how to make predictions using the trained model.
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from ml_models.inference.predictor import SpatialOreGradePredictor


def make_single_prediction():
    """Make a single prediction at specific coordinates"""
    
    print("🔮 MAKING SINGLE PREDICTION")
    print("="*40)
    
    # Initialize predictor (automatically loads latest model)
    try:
        predictor = SpatialOreGradePredictor()
        print("✅ Predictor initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing predictor: {str(e)}")
        return
    
    # Example coordinates (you can change these)
    latitude = -23.550500
    longitude = -46.633300
    depth_from = 50.0
    depth_to = 55.0
    element = "CU"
    
    print(f"📍 Prediction Location:")
    print(f"   Latitude: {latitude}")
    print(f"   Longitude: {longitude}")
    print(f"   Depth: {depth_from} - {depth_to} meters")
    print(f"   Element: {element}")
    
    try:
        # Make prediction
        result = predictor.predict_at_location(
            latitude=latitude,
            longitude=longitude,
            depth_from=depth_from,
            depth_to=depth_to,
            element=element
        )
        
        print(f"\n🎯 PREDICTION RESULT:")
        print(f"   Predicted Grade: {result['predicted_grade']:.2f} ppm")
        print(f"   Confidence: {result.get('confidence', 'N/A')}")
        print(f"   Model Used: {result.get('model_info', {}).get('model_type', 'XGBoost')}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error making prediction: {str(e)}")
        return None


def make_batch_predictions():
    """Make predictions for multiple locations"""
    
    print("\n🔮 MAKING BATCH PREDICTIONS")
    print("="*40)
    
    # Initialize batch predictor
    try:
        from ml_models.inference.batch_predictor import BatchSpatialPredictor
        batch_predictor = BatchSpatialPredictor()
        print("✅ Batch predictor initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing batch predictor: {str(e)}")
        return
    
    # Multiple coordinates to predict
    coordinates = [
        {"latitude": -23.550500, "longitude": -46.633300, "depth_from": 50, "depth_to": 55},
        {"latitude": -23.551500, "longitude": -46.634300, "depth_from": 100, "depth_to": 105},
        {"latitude": -23.552500, "longitude": -46.635300, "depth_from": 150, "depth_to": 155}
    ]
    
    elements = ["CU"]  # Can add more elements like ["CU", "AU", "AG"]
    
    print(f"📍 Batch Prediction:")
    print(f"   Locations: {len(coordinates)}")
    print(f"   Elements: {elements}")
    
    try:
        # Make batch predictions
        results = batch_predictor.predict_multiple_locations(
            coordinates=coordinates,
            elements=elements
        )
        
        print(f"\n🎯 BATCH RESULTS:")
        for i, coord in enumerate(coordinates):
            for element in elements:
                key = f"{element}_{i}"
                if key in results:
                    grade = results[key]['predicted_grade']
                    print(f"   Location {i+1} ({element}): {grade:.2f} ppm")
                    print(f"      Coords: ({coord['latitude']:.6f}, {coord['longitude']:.6f})")
                    print(f"      Depth: {coord['depth_from']}-{coord['depth_to']}m")
                    print()
        
        return results
        
    except Exception as e:
        print(f"❌ Error making batch predictions: {str(e)}")
        return None


def verify_against_test_sample():
    """Verify prediction against a known test sample"""
    
    print("\n🧪 VERIFYING AGAINST TEST SAMPLE")
    print("="*40)
    
    # This would come from your CSV file of test samples
    test_sample = {
        'latitude': -23.550500,
        'longitude': -46.633300,
        'depth_from': 50.0,
        'depth_to': 55.0,
        'actual_grade': 1234.56,  # This would be from your CSV
        'element': 'CU'
    }
    
    print(f"📊 Test Sample:")
    print(f"   Location: ({test_sample['latitude']:.6f}, {test_sample['longitude']:.6f})")
    print(f"   Depth: {test_sample['depth_from']}-{test_sample['depth_to']}m")
    print(f"   Actual Grade: {test_sample['actual_grade']:.2f} ppm")
    
    # Make prediction
    try:
        predictor = SpatialOreGradePredictor()
        result = predictor.predict_at_location(
            latitude=test_sample['latitude'],
            longitude=test_sample['longitude'],
            depth_from=test_sample['depth_from'],
            depth_to=test_sample['depth_to'],
            element=test_sample['element']
        )
        
        predicted_grade = result['predicted_grade']
        actual_grade = test_sample['actual_grade']
        
        # Calculate accuracy
        absolute_error = abs(predicted_grade - actual_grade)
        relative_error = (absolute_error / actual_grade) * 100
        accuracy = max(0, 100 - relative_error)
        
        print(f"\n🎯 VERIFICATION RESULT:")
        print(f"   Predicted Grade: {predicted_grade:.2f} ppm")
        print(f"   Actual Grade: {actual_grade:.2f} ppm")
        print(f"   Absolute Error: {absolute_error:.2f} ppm")
        print(f"   Relative Error: {relative_error:.1f}%")
        print(f"   Accuracy: {accuracy:.1f}%")
        
        if accuracy >= 90:
            print("   ✅ EXCELLENT accuracy!")
        elif accuracy >= 80:
            print("   ✅ GOOD accuracy!")
        elif accuracy >= 70:
            print("   ⚠️  FAIR accuracy")
        else:
            print("   ❌ POOR accuracy")
        
        return result
        
    except Exception as e:
        print(f"❌ Error in verification: {str(e)}")
        return None


def main():
    """Main execution function"""
    
    print("🔮 PREDICTION EXAMPLES")
    print("="*60)
    
    # Example 1: Single prediction
    single_result = make_single_prediction()
    
    # Example 2: Batch predictions
    batch_results = make_batch_predictions()
    
    # Example 3: Verify against test sample
    verification_result = verify_against_test_sample()
    
    print("\n🎉 PREDICTION EXAMPLES COMPLETED!")
    print("="*60)
    
    print("\n📋 NEXT STEPS:")
    print("1. Use test samples CSV to get real coordinates")
    print("2. Make predictions at those coordinates")
    print("3. Compare predictions with actual grades")
    print("4. Calculate accuracy and assess model performance")


if __name__ == "__main__":
    main()
