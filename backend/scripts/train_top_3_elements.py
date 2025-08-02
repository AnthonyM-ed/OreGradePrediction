"""
Train Top 3 Most Important Elements

This script trains models for the 3 most economically important elements:
- CU (Copper): Most common and valuable base metal
- AU (Gold): High-value precious metal
- AG (Silver): Valuable precious metal
"""

import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from ml_models.training.train_pipeline import TrainingPipeline

# ====================================================================
# CONFIGURATION - TOP 3 MOST IMPORTANT ELEMENTS
# ====================================================================

# Top 3 elements by economic importance
TOP_3_ELEMENTS = [
    'CU',  # Copper - Most important base metal
    'AU',  # Gold - Most valuable precious metal  
    'AG'   # Silver - Important precious metal
]

SAMPLE_LIMIT = None  # Use all available data for best accuracy
DATASET = "MAIN"

# ====================================================================

def train_top_3_elements():
    """Train models for the top 3 most important elements"""
    
    print("🏆 TRAINING TOP 3 MOST IMPORTANT ELEMENTS")
    print("="*60)
    print("📊 Elements to train:")
    print("   1. CU (Copper) - Most important base metal")
    print("   2. AU (Gold) - Most valuable precious metal")
    print("   3. AG (Silver) - Important precious metal")
    print()
    print(f"📈 Sample limit: {'ALL DATA (~58,694 samples)' if SAMPLE_LIMIT is None else SAMPLE_LIMIT}")
    print(f"📁 Dataset: {DATASET}")
    print(f"⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    results = {}
    total_start_time = time.time()
    
    for i, element in enumerate(TOP_3_ELEMENTS, 1):
        print(f"🚀 TRAINING ELEMENT {i}/3: {element}")
        print("-" * 40)
        
        try:
            # Initialize pipeline for this element
            pipeline = TrainingPipeline()
            
            # Start timing for this element
            element_start_time = time.time()
            
            # Train the model
            element_results = pipeline.run_complete_pipeline(
                element=element,
                dataset=DATASET,
                limit=SAMPLE_LIMIT
            )
            
            # End timing
            element_end_time = time.time()
            element_duration = element_end_time - element_start_time
            
            # Extract metrics
            eval_results = element_results.get('evaluation_results', {})
            test_metrics = eval_results.get('test_metrics', {})
            
            # Store results
            results[element] = {
                'results': element_results,
                'training_time': element_duration,
                'test_r2': test_metrics.get('r2_score', 0),
                'test_rmse': test_metrics.get('rmse', 0),
                'test_mae': test_metrics.get('mae', 0),
                'model_path': element_results.get('model_path', 'N/A'),
                'pipeline_id': element_results.get('pipeline_id', 'N/A'),
                'data_records': element_results.get('data_records', 0),
                'features_count': element_results.get('features_count', 0),
                'status': 'success'
            }
            
            # Display results
            print(f"✅ {element} COMPLETED!")
            print(f"   ⏱️  Time: {element_duration:.2f} seconds")
            print(f"   📊 Test R²: {test_metrics.get('r2_score', 0):.4f} ({test_metrics.get('r2_score', 0)*100:.1f}% accuracy)")
            print(f"   📊 Test RMSE: {test_metrics.get('rmse', 0):.2f} ppm")
            print(f"   📊 Test MAE: {test_metrics.get('mae', 0):.2f} ppm")
            print(f"   📊 Records: {element_results.get('data_records', 0):,}")
            print(f"   💾 Model: {element_results.get('model_path', 'N/A')}")
            print()
            
        except Exception as e:
            print(f"❌ Error training {element}: {str(e)}")
            results[element] = {
                'status': 'failed',
                'error': str(e),
                'training_time': 0
            }
            print()
    
    # Final summary
    total_time = time.time() - total_start_time
    successful = [elem for elem, res in results.items() if res['status'] == 'success']
    failed = [elem for elem, res in results.items() if res['status'] == 'failed']
    
    print("🎉 TOP 3 ELEMENTS TRAINING COMPLETED!")
    print("="*50)
    print(f"⏱️  Total time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
    print(f"✅ Successful: {len(successful)} ({', '.join(successful)})")
    print(f"❌ Failed: {len(failed)} ({', '.join(failed) if failed else 'None'})")
    print()
    
    if successful:
        print("📊 SUMMARY OF SUCCESSFUL MODELS:")
        print("-" * 35)
        for element in successful:
            res = results[element]
            print(f"{element}: R²={res['test_r2']:.4f} ({res['test_r2']*100:.1f}%), "
                  f"RMSE={res['test_rmse']:.1f} ppm, "
                  f"Records={res['data_records']:,}")
        
        print(f"\n💾 All models saved to: data/models/")
        print(f"📄 Reports saved to: data/exports/reports/")
    
    # Save summary report
    save_top_3_summary(results, total_time)
    
    return results

def save_top_3_summary(results, total_time):
    """Save a summary report for the top 3 elements training"""
    
    try:
        reports_dir = Path("data/reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = reports_dir / f"top_3_elements_summary_{timestamp}.txt"
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("TOP 3 ELEMENTS TRAINING SUMMARY\\n")
            f.write("=" * 40 + "\\n\\n")
            
            f.write(f"Training completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n")
            f.write(f"Total training time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)\\n")
            f.write(f"Elements trained: {', '.join(TOP_3_ELEMENTS)}\\n\\n")
            
            successful = [elem for elem, res in results.items() if res['status'] == 'success']
            failed = [elem for elem, res in results.items() if res['status'] == 'failed']
            
            f.write(f"Successful: {len(successful)}/{len(TOP_3_ELEMENTS)}\\n")
            f.write(f"Failed: {len(failed)}/{len(TOP_3_ELEMENTS)}\\n\\n")
            
            if successful:
                f.write("INDIVIDUAL RESULTS:\\n")
                f.write("-" * 20 + "\\n")
                
                for element in successful:
                    res = results[element]
                    f.write(f"\\n{element} (Element {TOP_3_ELEMENTS.index(element)+1}/3):\\n")
                    f.write(f"  Status: SUCCESS\\n")
                    f.write(f"  Training time: {res['training_time']:.2f} seconds\\n")
                    f.write(f"  Test R²: {res['test_r2']:.4f} ({res['test_r2']*100:.1f}% accuracy)\\n")
                    f.write(f"  Test RMSE: {res['test_rmse']:.2f} ppm\\n")
                    f.write(f"  Test MAE: {res['test_mae']:.2f} ppm\\n")
                    f.write(f"  Data records: {res['data_records']:,}\\n")
                    f.write(f"  Features: {res['features_count']}\\n")
                    f.write(f"  Model file: {res['model_path']}\\n")
                    f.write(f"  Pipeline ID: {res['pipeline_id']}\\n")
            
            if failed:
                f.write("\\nFAILED ELEMENTS:\\n")
                f.write("-" * 17 + "\\n")
                for element in failed:
                    f.write(f"{element}: {results[element]['error']}\\n")
            
            f.write("\\nNOTE: Each element has its own separate model.\\n")
            f.write("These models work independently for predictions.\\n")
        
        print(f"📄 Summary report saved: {summary_path}")
        
    except Exception as e:
        print(f"⚠️  Could not save summary report: {str(e)}")

if __name__ == "__main__":
    print("🔧 CONFIGURATION:")
    print(f"   Elements: {', '.join(TOP_3_ELEMENTS)}")
    print(f"   Sample limit: {'ALL DATA' if SAMPLE_LIMIT is None else SAMPLE_LIMIT}")
    print(f"   Dataset: {DATASET}")
    print()
    
    results = train_top_3_elements()
    
    print("\\n🎯 TRAINING SESSION COMPLETED!")
    print("Each element now has its own trained model ready for predictions.")
