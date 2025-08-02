"""
Test Scaler Issue in Training Pipeline

This script tests if the training pipeline correctly saves the scaler.
"""

import os
import sys
import joblib

# Add backend to path
sys.path.append(os.path.dirname(__file__))

def test_scaler_issue():
    """Test the scaler saving issue"""
    print("🔍 TESTING SCALER SAVING ISSUE")
    print("=" * 50)
    
    # Check existing scaler files
    models_dir = "data/models"
    if os.path.exists(models_dir):
        scaler_files = [f for f in os.listdir(models_dir) if f.startswith('scaler_')]
        print(f"📂 Found {len(scaler_files)} scaler files:")
        
        for scaler_file in scaler_files[:3]:  # Check first 3
            scaler_path = os.path.join(models_dir, scaler_file)
            file_size = os.path.getsize(scaler_path)
            print(f"   📄 {scaler_file}: {file_size} bytes")
            
            # Try to load and check content
            try:
                scaler = joblib.load(scaler_path)
                print(f"      ✅ Content: {type(scaler)}")
                if scaler is None:
                    print("      ❌ PROBLEM: Scaler is None!")
                else:
                    print(f"      ✅ Scaler type: {scaler.__class__.__name__}")
            except Exception as e:
                print(f"      ❌ Error loading: {e}")
            print()
    else:
        print("❌ Models directory not found!")

if __name__ == "__main__":
    test_scaler_issue()
