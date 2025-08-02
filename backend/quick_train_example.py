"""
Quick Training Example - Configurable Ore Grade Prediction Model

This script demonstrates how to train ore grade prediction models
with configurable sample sizes and elements.
"""

import os
import sys
import time
import json
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(__file__))

# Import ML modules
from ml_models.training.train_pipeline import TrainingPipeline

# ====================================================================
# CONFIGURATION VARIABLES - CHANGE THESE VALUES AS NEEDED
# ====================================================================

# Training Configuration
ELEMENT = "CU"           # Element to train: "CU", "AU", "AG", "PB", "ZN", "MO", "FE", "S", "AS", "SB"
SAMPLE_LIMIT = 6000      # Number of samples to use: 100, 1000, 5000, 10000, or None for all data
DATASET = "MAIN"         # Dataset to use (usually "MAIN")

# Available elements in your database:
# - CU: Copper
# - AU: Gold  
# - AG: Silver
# - PB: Lead
# - ZN: Zinc
# - MO: Molybdenum
# - FE: Iron
# - S: Sulfur
# - AS: Arsenic
# - SB: Antimony

# ====================================================================

def generate_comprehensive_reports(results, training_time, start_time):
    """Generate comprehensive training and evaluation reports"""
    
    try:
        # Create reports directories
        reports_dir = Path("data/reports")
        exports_reports_dir = Path("data/exports/reports")
        plots_dir = Path("data/exports/plots")
        
        for dir_path in [reports_dir, exports_reports_dir, plots_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pipeline_id = results.get('pipeline_id', timestamp)
        
        # Extract results
        eval_results = results.get('evaluation_results', {})
        test_metrics = eval_results.get('test_metrics', {})
        train_metrics = eval_results.get('train_metrics', {})
        cv_metrics = eval_results.get('cv_metrics', {})
        
        # 1. DETAILED TRAINING REPORT
        training_report_path = reports_dir / f"training_report_{ELEMENT}_{timestamp}.txt"
        
        with open(training_report_path, 'w', encoding='utf-8') as f:
            f.write("COMPREHENSIVE TRAINING REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("TRAINING CONFIGURATION:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Element: {ELEMENT}\n")
            f.write(f"Dataset: {DATASET}\n")
            f.write(f"Sample limit: {SAMPLE_LIMIT if SAMPLE_LIMIT else 'ALL DATA (~58,694)'}\n")
            f.write(f"Pipeline ID: {pipeline_id}\n")
            f.write(f"Training started: {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Training duration: {training_time:.2f} seconds\n")
            f.write(f"Total samples: {results.get('data_records', 'N/A')}\n")
            f.write(f"Features engineered: {results.get('features_count', 'N/A')}\n\n")
            
            f.write("DATA SAMPLING STRATEGY:\n")
            f.write("-" * 25 + "\n")
            f.write("Random sampling: YES (using random seed 42)\n")
            f.write("Split method: Stratified random split\n")
            f.write("Reproducibility: Same samples every run\n")
            f.write("Test data protection: Test samples never seen during training\n\n")
            
            # Data split details
            total_samples = results.get('data_records', 1000)
            train_samples = int(total_samples * 0.64)
            val_samples = int(total_samples * 0.16)
            test_samples = total_samples - train_samples - val_samples
            
            f.write("DATA SPLIT:\n")
            f.write("-" * 15 + "\n")
            f.write(f"Training samples: {train_samples} ({train_samples/total_samples*100:.1f}%)\n")
            f.write(f"Validation samples: {val_samples} ({val_samples/total_samples*100:.1f}%)\n")
            f.write(f"Test samples: {test_samples} ({test_samples/total_samples*100:.1f}%)\n\n")
            
            f.write("MODEL PERFORMANCE:\n")
            f.write("-" * 25 + "\n")
            f.write("Test Set (Final Evaluation):\n")
            f.write(f"  R² Score: {test_metrics.get('r2_score', 'N/A'):.4f}\n")
            f.write(f"  RMSE: {test_metrics.get('rmse', 'N/A'):.2f} ppm\n")
            f.write(f"  MAE: {test_metrics.get('mae', 'N/A'):.2f} ppm\n")
            f.write(f"  MAPE: {test_metrics.get('mape', 'N/A'):.2f}%\n")
            f.write(f"  Bias: {test_metrics.get('bias', 'N/A'):.2f}\n\n")
            
            f.write("Training Set (Training Performance):\n")
            f.write(f"  R² Score: {train_metrics.get('r2_score', 'N/A'):.4f}\n")
            f.write(f"  RMSE: {train_metrics.get('rmse', 'N/A'):.2f} ppm\n")
            f.write(f"  MAE: {train_metrics.get('mae', 'N/A'):.2f} ppm\n\n")
            
            if cv_metrics:
                f.write("Cross-Validation (5-fold):\n")
                f.write(f"  CV R² Mean: {cv_metrics.get('r2_mean', 'N/A'):.4f}\n")
                f.write(f"  CV R² Std: {cv_metrics.get('r2_std', 'N/A'):.4f}\n")
                f.write(f"  CV RMSE Mean: {cv_metrics.get('rmse_mean', 'N/A'):.2f}\n")
                f.write(f"  CV RMSE Std: {cv_metrics.get('rmse_std', 'N/A'):.2f}\n\n")
            
            # Model interpretation
            r2_score = test_metrics.get('r2_score', 0)
            if r2_score >= 0.95:
                f.write("MODEL ASSESSMENT: EXCELLENT\n")
                f.write("The model shows excellent predictive performance (R2 >= 0.95)\n")
            elif r2_score >= 0.85:
                f.write("MODEL ASSESSMENT: GOOD\n")
                f.write("The model shows good predictive performance (R2 >= 0.85)\n")
            elif r2_score >= 0.70:
                f.write("MODEL ASSESSMENT: FAIR\n")
                f.write("The model shows fair performance. Consider more data.\n")
            else:
                f.write("MODEL ASSESSMENT: POOR\n")
                f.write("The model needs improvement. Try more samples or different features.\n")
            
            f.write(f"\nModel files saved to: {results.get('model_path', 'N/A')}\n")
            f.write(f"Report generated: {datetime.now().isoformat()}\n")
        
        print(f" Training report saved: {training_report_path}")
        
        # 2. JSON SUMMARY REPORT
        json_report_path = reports_dir / f"training_summary_{ELEMENT}_{timestamp}.json"
        
        summary_data = {
            "training_info": {
                "element": ELEMENT,
                "dataset": DATASET,
                "sample_limit": SAMPLE_LIMIT,
                "pipeline_id": pipeline_id,
                "training_start": datetime.fromtimestamp(start_time).isoformat(),
                "training_duration_seconds": training_time,
                "total_samples": results.get('data_records', 'N/A'),
                "features_count": results.get('features_count', 'N/A'),
                "random_seed": 42,
                "sampling_method": "stratified_random"
            },
            "data_split": {
                "training_samples": train_samples,
                "validation_samples": val_samples,
                "test_samples": test_samples,
                "training_percentage": round(train_samples/total_samples*100, 1),
                "validation_percentage": round(val_samples/total_samples*100, 1),
                "test_percentage": round(test_samples/total_samples*100, 1)
            },
            "performance_metrics": {
                "test_metrics": test_metrics,
                "train_metrics": train_metrics,
                "cv_metrics": cv_metrics
            },
            "model_files": {
                "model_path": results.get('model_path', 'N/A'),
                "pipeline_id": pipeline_id
            },
            "report_timestamp": datetime.now().isoformat()
        }
        
        with open(json_report_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2)
        
        print(f" JSON summary saved: {json_report_path}")
        
        # 3. COPY EVALUATION REPORT TO MAIN REPORTS FOLDER
        eval_report_source = exports_reports_dir / f"evaluation_report_{pipeline_id}.txt"
        eval_report_dest = reports_dir / f"model_evaluation_{ELEMENT}_{timestamp}.txt"
        
        if eval_report_source.exists():
            import shutil
            shutil.copy2(eval_report_source, eval_report_dest)
            print(f" Evaluation report copied: {eval_report_dest}")
        
        # 4. QUICK REFERENCE CARD
        quick_ref_path = reports_dir / f"quick_reference_{ELEMENT}_{timestamp}.txt"
        
        with open(quick_ref_path, 'w', encoding='utf-8') as f:
            f.write(f"QUICK REFERENCE CARD - {ELEMENT} MODEL\n")
            f.write("=" * 45 + "\n\n")
            f.write(f"Pipeline ID: {pipeline_id}\n")
            f.write(f"Trained: {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Duration: {training_time:.2f} seconds\n")
            f.write(f"Samples: {results.get('data_records', 'N/A')}\n")
            f.write(f"Features: {results.get('features_count', 'N/A')}\n")
            f.write(f"Random seed: 42 (reproducible results)\n\n")
            
            f.write("PERFORMANCE SUMMARY:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Test R²: {test_metrics.get('r2_score', 'N/A'):.4f} ({test_metrics.get('r2_score', 0)*100:.1f}% accuracy)\n")
            f.write(f"Test RMSE: {test_metrics.get('rmse', 'N/A'):.2f} ppm\n")
            f.write(f"Test MAE: {test_metrics.get('mae', 'N/A'):.2f} ppm\n\n")
            
            f.write("HOW TO USE THIS MODEL:\n")
            f.write("-" * 22 + "\n")
            f.write("1. Load the model:\n")
            f.write(f"   import joblib\n")
            f.write(f"   model = joblib.load('{results.get('model_path', 'N/A')}')\n\n")
            f.write("2. Make predictions:\n")
            f.write("   from ml_models.inference.predictor import SpatialOreGradePredictor\n")
            f.write("   predictor = SpatialOreGradePredictor()\n")
            f.write("   result = predictor.predict_at_location(\n")
            f.write("       latitude=your_lat, longitude=your_lon,\n")
            f.write("       depth_from=your_depth_from, depth_to=your_depth_to,\n")
            f.write(f"       element='{ELEMENT}'\n")
            f.write("   )\n\n")
            f.write("3. Files location:\n")
            f.write(f"   Model: {results.get('model_path', 'N/A')}\n")
            f.write(f"   Reports: {reports_dir}\n")
            f.write(f"   Evaluation: {exports_reports_dir}\n")
        
        print(f" Quick reference saved: {quick_ref_path}")
        
        # 5. SUMMARY DISPLAY
        print("\n REPORTS GENERATED:")
        print("-" * 25)
        print(f" Training Report: {training_report_path.name}")
        print(f" JSON Summary: {json_report_path.name}")
        print(f" Model Evaluation: {eval_report_dest.name}")
        print(f" Quick Reference: {quick_ref_path.name}")
        print(f"\n All reports saved to: {reports_dir}")
        
        return {
            'training_report': str(training_report_path),
            'json_summary': str(json_report_path),
            'evaluation_report': str(eval_report_dest),
            'quick_reference': str(quick_ref_path)
        }
        
    except Exception as e:
        print(f"❌ Error generating reports: {str(e)}")
        return None


def quick_train_model():
    """Train a configurable ore grade prediction model"""
    
    print(" CONFIGURABLE ORE GRADE MODEL TRAINING")
    print("="*50)
    print(f" Element: {ELEMENT}")
    print(f" Sample limit: {SAMPLE_LIMIT if SAMPLE_LIMIT else 'ALL DATA (~58,694 samples)'}")
    print(f" Dataset: {DATASET}")
    print(f" Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Explain data sampling strategy
    print(" DATA SAMPLING STRATEGY:")
    print("-" * 30)
    print(" RANDOM SAMPLING: Yes, the data split is randomized")
    print(" Random seed: 42 (for reproducible results)")
    print(" Split ratios:")
    print("   • Training: 64% (model learns from this)")
    print("   • Validation: 16% (used during training for tuning)")
    print("   • Test: 20% (completely hidden, used for final evaluation)")
    print()
    print("⚠️  IMPORTANT: The same random seed (42) ensures:")
    print("   • Same data split every time you run the script")
    print("   • Test samples are ALWAYS the same (fair comparison)")
    print("   • Results are reproducible")
    print()
    
    try:
        # Initialize training pipeline
        pipeline = TrainingPipeline()
        
        # Start timing
        start_time = time.time()
        
        # Run complete pipeline with configured parameters
        results = pipeline.run_complete_pipeline(
            element=ELEMENT,
            dataset=DATASET,
            limit=SAMPLE_LIMIT
        )
        
        # End timing
        end_time = time.time()
        training_time = end_time - start_time
        
        # Extract results
        eval_results = results.get('evaluation_results', {})
        test_metrics = eval_results.get('test_metrics', {})
        train_metrics = eval_results.get('train_metrics', {})
        
        print(" TRAINING COMPLETED SUCCESSFULLY!")
        print("="*50)
        print(f" Training time: {training_time:.2f} seconds")
        print(f" Total samples: {results.get('data_records', 'N/A')}")
        print(f" Features: {results.get('features_count', 'N/A')}")
        print()
        
        print(" MODEL PERFORMANCE:")
        print("-" * 25)
        print(f"Test R²:    {test_metrics.get('r2_score', 'N/A'):.4f}")
        print(f"Test RMSE:  {test_metrics.get('rmse', 'N/A'):.2f} ppm")
        print(f"Test MAE:   {test_metrics.get('mae', 'N/A'):.2f} ppm")
        print(f"Test MAPE:  {test_metrics.get('mape', 'N/A'):.2f}%")
        print()
        
        print(" TRAINING SET PERFORMANCE:")
        print("-" * 30)
        print(f"Train R²:   {train_metrics.get('r2_score', 'N/A'):.4f}")
        print(f"Train RMSE: {train_metrics.get('rmse', 'N/A'):.2f} ppm")
        print(f"Train MAE:  {train_metrics.get('mae', 'N/A'):.2f} ppm")
        print()
        
        # Model interpretation
        r2_score = test_metrics.get('r2_score', 0)
        if r2_score >= 0.95:
            print(" EXCELLENT: Model shows excellent predictive performance!")
        elif r2_score >= 0.85:
            print(" GOOD: Model shows good predictive performance!")
        elif r2_score >= 0.70:
            print("  FAIR: Model shows fair performance, consider more data")
        else:
            print(" POOR: Model needs improvement, try more samples")
        
        # Data split information
        print("\n DATA SPLIT DETAILS:")
        print("-" * 20)
        total_samples = results.get('data_records', SAMPLE_LIMIT or 58694)
        train_samples = int(total_samples * 0.64)  # 64% training
        val_samples = int(total_samples * 0.16)    # 16% validation
        test_samples = total_samples - train_samples - val_samples  # ~20% test
        
        print(f"Training:   {train_samples} samples ({train_samples/total_samples*100:.1f}%)")
        print(f"Validation: {val_samples} samples ({val_samples/total_samples*100:.1f}%)")
        print(f"Test:       {test_samples} samples ({test_samples/total_samples*100:.1f}%)")
        
        print(f"\n Model saved: {results.get('model_path', 'N/A')}")
        print(f" Pipeline ID: {results.get('pipeline_id', 'N/A')}")
        
        # Generate comprehensive reports
        print("\n GENERATING COMPREHENSIVE REPORTS...")
        print("-" * 40)
        
        generate_comprehensive_reports(results, training_time, start_time)
        
        return results
        
    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    print(" CONFIGURATION:")
    print(f"   Element: {ELEMENT}")
    print(f"   Sample limit: {SAMPLE_LIMIT if SAMPLE_LIMIT else 'ALL DATA (~58,694)'}")
    print(f"   Dataset: {DATASET}")
    print()
    
    results = quick_train_model()
    
    print(f"\n {ELEMENT} MODEL TRAINING COMPLETED!")
    print("You can now use this model for predictions.")
    print(f" Reports saved to: data/reports/")
    print(f" Model saved to: {results.get('model_path', 'N/A')}")
    
    # Show how to change configuration
    print("\n TO CHANGE CONFIGURATION:")
    print("   Edit the variables at the top of this script:")
    print(f"   - ELEMENT = '{ELEMENT}'     # Change element")
    print(f"   - SAMPLE_LIMIT = {SAMPLE_LIMIT}    # Change sample count")
    print(f"   - DATASET = '{DATASET}'       # Change dataset")
