"""
Comprehensive Multi-Element Training Script

This script trains XGBoost models for all available elements using the complete dataset.
It includes progress tracking, time estimation, and robust error handling.
"""

import os
import sys
import time
import json
from datetime import datetime, timedelta
from pathlib import Path
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import ML modules
from ml_models.training.train_pipeline import TrainingPipeline
from ml_models.data_processing.db_loader import XGBoostGeologicalDataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_log.txt'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MultiElementTrainer:
    """
    Comprehensive trainer for all elements in the database
    """
    
    def __init__(self):
        self.pipeline = TrainingPipeline()
        self.db_loader = XGBoostGeologicalDataLoader()
        self.results = {}
        self.start_time = None
        self.total_elements = 0
        
        # Common elements to train (you can modify this list)
        self.elements_to_train = [
            'CU',  # Copper
            'AU',  # Gold
            'AG',  # Silver
            'PB',  # Lead
            'ZN',  # Zinc
            'MO',  # Molybdenum
            'FE',  # Iron
            'S',   # Sulfur
            'AS',  # Arsenic
            'SB'   # Antimony
        ]
        
    def get_available_elements(self):
        """Get all available elements from the database"""
        try:
            logger.info("Checking available elements in database...")
            
            # Load a small sample to check available elements
            X_train, X_test, y_train, y_test = self.db_loader.load_xgboost_training_data(
                elements=['CU']  # Use CU as reference
            )
            
            # Get unique elements from the database
            from ml_models.database.connections import get_db_connection
            
            with get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Query to get all available elements
                query = """
                SELECT DISTINCT element_code 
                FROM vw_HoleSamples_ElementGrades 
                WHERE standardized_grade_ppm IS NOT NULL 
                AND standardized_grade_ppm > 0
                ORDER BY element_code
                """
                
                cursor.execute(query)
                available_elements = [row[0] for row in cursor.fetchall()]
                
            logger.info(f"Available elements: {available_elements}")
            return available_elements
            
        except Exception as e:
            logger.error(f"Error getting available elements: {str(e)}")
            return self.elements_to_train  # Fallback to default list
    
    def estimate_training_time(self, sample_size: int, num_elements: int):
        """Estimate total training time based on sample run"""
        # Based on your test: 30 samples took ~18 seconds
        # Full dataset: 58,694 samples
        
        time_per_sample = 18 / 30  # seconds per sample
        time_per_element = sample_size * time_per_sample
        total_time = time_per_element * num_elements
        
        return total_time
    
    def format_time(self, seconds):
        """Format seconds into readable time"""
        if seconds < 60:
            return f"{seconds:.1f} seconds"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f} minutes"
        else:
            hours = seconds / 3600
            return f"{hours:.1f} hours"
    
    def train_single_element(self, element: str, element_index: int, total_elements: int):
        """Train model for a single element"""
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"TRAINING ELEMENT {element_index}/{total_elements}: {element}")
            logger.info(f"{'='*60}")
            
            element_start_time = time.time()
            
            # Train the model
            results = self.pipeline.run_complete_pipeline(
                element=element,
                dataset="MAIN"
                # No limit - use all samples
            )
            
            element_end_time = time.time()
            element_duration = element_end_time - element_start_time
            
            # Store results
            self.results[element] = {
                'results': results,
                'training_time': element_duration,
                'status': 'success',
                'timestamp': datetime.now().isoformat()
            }
            
            # Log performance
            eval_results = results.get('evaluation_results', {})
            test_metrics = eval_results.get('test_metrics', {})
            
            logger.info(f"✅ {element} COMPLETED in {self.format_time(element_duration)}")
            logger.info(f"   📊 Test R²: {test_metrics.get('r2_score', 'N/A'):.4f}")
            logger.info(f"   📊 Test RMSE: {test_metrics.get('rmse', 'N/A'):.2f}")
            logger.info(f"   📊 Test MAE: {test_metrics.get('mae', 'N/A'):.2f}")
            logger.info(f"   📁 Records: {results.get('data_records', 'N/A')}")
            logger.info(f"   🔧 Features: {results.get('features_count', 'N/A')}")
            
            # Estimate remaining time
            if element_index < total_elements:
                elapsed_time = time.time() - self.start_time
                avg_time_per_element = elapsed_time / element_index
                remaining_elements = total_elements - element_index
                estimated_remaining = avg_time_per_element * remaining_elements
                
                logger.info(f"   ⏱️  Estimated remaining: {self.format_time(estimated_remaining)}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error training {element}: {str(e)}")
            
            self.results[element] = {
                'results': None,
                'training_time': 0,
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            
            return False
    
    def save_comprehensive_results(self):
        """Save comprehensive training results"""
        try:
            results_dir = Path("data/exports/reports")
            results_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save detailed results
            results_file = results_dir / f"multi_element_training_{timestamp}.json"
            with open(results_file, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            
            # Create summary report
            summary_file = results_dir / f"training_summary_{timestamp}.txt"
            with open(summary_file, 'w') as f:
                f.write("MULTI-ELEMENT TRAINING SUMMARY\n")
                f.write("="*50 + "\n\n")
                
                total_time = time.time() - self.start_time
                f.write(f"Total Training Time: {self.format_time(total_time)}\n")
                f.write(f"Elements Trained: {len(self.results)}\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n\n")
                
                # Success/failure counts
                successful = sum(1 for r in self.results.values() if r['status'] == 'success')
                failed = sum(1 for r in self.results.values() if r['status'] == 'failed')
                
                f.write(f"Successful: {successful}\n")
                f.write(f"Failed: {failed}\n\n")
                
                # Individual element results
                f.write("INDIVIDUAL ELEMENT RESULTS:\n")
                f.write("-" * 40 + "\n")
                
                for element, result in self.results.items():
                    f.write(f"\n{element}:\n")
                    f.write(f"  Status: {result['status']}\n")
                    f.write(f"  Training Time: {self.format_time(result['training_time'])}\n")
                    
                    if result['status'] == 'success':
                        eval_results = result['results'].get('evaluation_results', {})
                        test_metrics = eval_results.get('test_metrics', {})
                        
                        f.write(f"  Test R²: {test_metrics.get('r2_score', 'N/A'):.4f}\n")
                        f.write(f"  Test RMSE: {test_metrics.get('rmse', 'N/A'):.2f}\n")
                        f.write(f"  Test MAE: {test_metrics.get('mae', 'N/A'):.2f}\n")
                        f.write(f"  Records: {result['results'].get('data_records', 'N/A')}\n")
                        f.write(f"  Features: {result['results'].get('features_count', 'N/A')}\n")
                    else:
                        f.write(f"  Error: {result.get('error', 'Unknown error')}\n")
            
            logger.info(f"📄 Comprehensive results saved to: {results_file}")
            logger.info(f"📄 Summary report saved to: {summary_file}")
            
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
    
    def train_all_elements(self, custom_elements=None):
        """Train models for all elements"""
        try:
            logger.info("🚀 Starting Multi-Element Training Pipeline")
            logger.info("="*60)
            
            self.start_time = time.time()
            
            # Get elements to train
            if custom_elements:
                elements = custom_elements
            else:
                elements = self.get_available_elements()
            
            self.total_elements = len(elements)
            
            # Estimate training time
            estimated_time = self.estimate_training_time(58694, self.total_elements)
            logger.info(f"📊 Elements to train: {self.total_elements}")
            logger.info(f"📊 Estimated total time: {self.format_time(estimated_time)}")
            logger.info(f"📊 Elements: {', '.join(elements)}")
            
            # Confirm before starting
            print(f"\n⚠️  WARNING: This will train {self.total_elements} models on ~58,694 samples each")
            print(f"⚠️  Estimated time: {self.format_time(estimated_time)}")
            print(f"⚠️  This will generate {self.total_elements * 3} files (model + scaler + metadata)")
            
            response = input("\n🤔 Continue with training? (y/n): ").lower().strip()
            if response != 'y':
                logger.info("Training cancelled by user")
                return
            
            logger.info("\n🎯 Starting training...")
            
            # Train each element
            for i, element in enumerate(elements, 1):
                success = self.train_single_element(element, i, self.total_elements)
                
                if not success:
                    logger.warning(f"⚠️  Failed to train {element}, continuing with next element...")
                
                # Small delay to prevent overwhelming the system
                time.sleep(1)
            
            # Final summary
            total_time = time.time() - self.start_time
            successful = sum(1 for r in self.results.values() if r['status'] == 'success')
            failed = sum(1 for r in self.results.values() if r['status'] == 'failed')
            
            logger.info(f"\n{'='*60}")
            logger.info(f"🎉 MULTI-ELEMENT TRAINING COMPLETED!")
            logger.info(f"{'='*60}")
            logger.info(f"⏱️  Total Time: {self.format_time(total_time)}")
            logger.info(f"✅ Successful: {successful}")
            logger.info(f"❌ Failed: {failed}")
            logger.info(f"📊 Success Rate: {successful/self.total_elements*100:.1f}%")
            
            # Save results
            self.save_comprehensive_results()
            
            return self.results
            
        except Exception as e:
            logger.error(f"Error in multi-element training: {str(e)}")
            raise


def main():
    """Main execution function"""
    try:
        trainer = MultiElementTrainer()
        
        # Option 1: Train all available elements
        results = trainer.train_all_elements()
        
        # Option 2: Train specific elements only
        # results = trainer.train_all_elements(['CU', 'AU', 'AG', 'PB', 'ZN'])
        
        print("\n✅ Multi-element training completed successfully!")
        print("📁 Check data/exports/reports/ for detailed results")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    main()
