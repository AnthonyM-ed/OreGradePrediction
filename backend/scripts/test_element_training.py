"""
Quick Test for Element-Specific Training

This script tests training a single element model with proper metadata.
"""

import os
import sys
import time
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from ml_models.training.train_pipeline import TrainingPipeline

def test_element_training():
    """Test training with element metadata"""
    print("🧪 TESTING ELEMENT-SPECIFIC TRAINING")
    print("="*50)
    
    try:
        # Test with Au (Gold)
        element = "AU"
        print(f"🚀 Training {element} model with metadata fix...")
        
        pipeline = TrainingPipeline()
        start_time = time.time()
        
        results = pipeline.run_complete_pipeline(
            element=element,
            dataset="MAIN",
            limit=1000  # Small sample for quick test
        )
        
        end_time = time.time()
        
        print(f"✅ {element} training completed in {end_time - start_time:.2f} seconds")
        
        # Check if metadata contains element
        model_path = results.get('model_path', '')
        if model_path:
            import json
            metadata_path = model_path.replace('grade_model_', 'model_metadata_').replace('.joblib', '.json')
            
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                stored_element = metadata.get('element', 'NOT_FOUND')
                print(f"📊 Element in metadata: {stored_element}")
                
                if stored_element == element:
                    print("✅ Element metadata is correctly stored!")
                else:
                    print("❌ Element metadata is incorrect!")
            else:
                print("❌ Metadata file not found!")
        
        return results
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_element_training()
