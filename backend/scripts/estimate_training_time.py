"""
Training Time Estimation Script

This script runs a quick test to estimate how long full training will take.
"""

import os
import sys
import time
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from ml_models.training.train_pipeline import TrainingPipeline


def estimate_training_time():
    """Estimate training time for full dataset"""
    print("🔍 Estimating training time for full dataset...")
    print("="*50)
    
    pipeline = TrainingPipeline()
    
    # Test with different sample sizes
    test_sizes = [100, 500, 1000]
    times = []
    
    for size in test_sizes:
        print(f"\n📊 Testing with {size} samples...")
        
        start_time = time.time()
        
        try:
            results = pipeline.run_complete_pipeline(
                element='CU',
                dataset='MAIN',
                limit=size
            )
            
            end_time = time.time()
            duration = end_time - start_time
            times.append((size, duration))
            
            print(f"✅ Completed in {duration:.2f} seconds")
            print(f"   Time per sample: {duration/size:.4f} seconds")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            continue
    
    # Calculate average time per sample
    if times:
        total_samples = sum(size for size, _ in times)
        total_time = sum(duration for _, duration in times)
        avg_time_per_sample = total_time / total_samples
        
        print(f"\n📈 ESTIMATION RESULTS:")
        print(f"="*30)
        print(f"Average time per sample: {avg_time_per_sample:.4f} seconds")
        
        # Estimate for full dataset
        full_dataset_size = 58694  # From your previous test
        estimated_time_per_element = full_dataset_size * avg_time_per_sample
        
        print(f"Estimated time per element: {estimated_time_per_element:.1f} seconds ({estimated_time_per_element/60:.1f} minutes)")
        
        # Estimate for 10 elements (common ones)
        elements_to_train = 10
        total_estimated_time = estimated_time_per_element * elements_to_train
        
        print(f"\nFor {elements_to_train} elements:")
        print(f"Total estimated time: {total_estimated_time:.1f} seconds")
        print(f"Total estimated time: {total_estimated_time/60:.1f} minutes")
        print(f"Total estimated time: {total_estimated_time/3600:.1f} hours")
        
        # Time estimates
        eta = datetime.now() + timedelta(seconds=total_estimated_time)
        print(f"Estimated completion: {eta.strftime('%Y-%m-%d %H:%M:%S')}")
        
        return avg_time_per_sample, estimated_time_per_element, total_estimated_time
    
    return None, None, None


def main():
    """Main execution"""
    try:
        print("⏱️  TRAINING TIME ESTIMATION")
        print("="*60)
        
        avg_time, time_per_element, total_time = estimate_training_time()
        
        if avg_time:
            print(f"\n💡 RECOMMENDATIONS:")
            print(f"="*20)
            
            if total_time > 3600:  # More than 1 hour
                print("⚠️  Training will take more than 1 hour")
                print("💡 Consider:")
                print("   - Training elements one by one")
                print("   - Using a more powerful machine")
                print("   - Running overnight")
            
            if total_time > 7200:  # More than 2 hours
                print("⚠️  Training will take more than 2 hours")
                print("💡 Consider:")
                print("   - Start with most important elements (CU, AU, AG)")
                print("   - Use background processing")
                print("   - Monitor progress with training_monitor.py")
        
        print(f"\n🚀 To start full training, run:")
        print(f"   python scripts/train_all_elements.py")
        print(f"\n📊 To monitor progress, run:")
        print(f"   python scripts/training_monitor.py --continuous")
        
    except Exception as e:
        print(f"❌ Error during estimation: {str(e)}")


if __name__ == "__main__":
    main()
