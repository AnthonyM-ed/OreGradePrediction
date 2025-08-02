"""
Quick Test Training for CU

This script does a quick training test for copper to verify the scaler fix.
"""

import os
import sys
import time
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(__file__))

from ml_models.training.train_pipeline import TrainingPipeline

def quick_train_cu():
    """Quick training test for copper"""
    print("🚀 QUICK TRAINING TEST FOR COPPER")
    print("="*50)
    print(f"⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # Initialize pipeline
        pipeline = TrainingPipeline()
        
        # Train CU model with small sample for quick test
        print("🔬 Training CU model with limited samples...")
        
        start_time = time.time()
        results = pipeline.run_complete_pipeline(
            element="CU",
            dataset="MAIN",
            limit=1000  # Small sample for quick test
        )
        end_time = time.time()
        
        print(f"✅ Training completed in {end_time - start_time:.2f} seconds")
        print()
        
        # Check results
        eval_results = results.get('evaluation_results', {})
        test_metrics = eval_results.get('test_metrics', {})
        
        print("📊 RESULTS:")
        print(f"   R² Score: {test_metrics.get('r2_score', 0):.4f}")
        print(f"   RMSE: {test_metrics.get('rmse', 0):.2f} ppm")
        print(f"   MAE: {test_metrics.get('mae', 0):.2f} ppm")
        print(f"   Model: {results.get('model_path', 'N/A')}")
        
        # Check if scaler was saved correctly
        model_path = results.get('model_path', '')
        if model_path:
            scaler_path = model_path.replace('grade_model_', 'scaler_')
            if os.path.exists(scaler_path):
                import joblib
                scaler = joblib.load(scaler_path)
                file_size = os.path.getsize(scaler_path)
                
                print()
                print("🔧 SCALER CHECK:")
                print(f"   File: {os.path.basename(scaler_path)}")
                print(f"   Size: {file_size} bytes")
                if scaler is None:
                    print("   ❌ SCALER IS STILL None!")
                else:
                    print(f"   ✅ Scaler type: {scaler.__class__.__name__}")
                    print(f"   ✅ Scaler ready for inference!")
            else:
                print("   ❌ Scaler file not found!")
        
    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    quick_train_cu()
