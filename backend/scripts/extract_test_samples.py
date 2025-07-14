"""
Test Sample Extractor

This script extracts the test samples that were held out during training,
so you can manually verify predictions at specific coordinates.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from ml_models.training.train_pipeline import TrainingPipeline
from ml_models.inference.predictor import SpatialOreGradePredictor
from ml_models.data_processing.db_loader import XGBoostGeologicalDataLoader


class TestSampleExtractor:
    """Extract and analyze test samples for manual verification"""
    
    def __init__(self):
        self.pipeline = TrainingPipeline()
        self.db_loader = XGBoostGeologicalDataLoader()
        self.test_samples = {}
        
    def extract_test_samples(self, element: str = "CU", save_to_file: bool = True):
        """Extract the exact test samples used for model evaluation"""
        try:
            print(f"🔍 EXTRACTING TEST SAMPLES FOR {element}")
            print("="*60)
            
            # Load and process data exactly as done during training
            print("📊 Loading raw data...")
            raw_data = self.pipeline.load_data(element=element, dataset="MAIN")
            
            print("🔧 Preparing features...")
            features, target = self.pipeline.prepare_features(raw_data)
            
            print("✂️ Splitting data...")
            processed_data = self.pipeline.split_and_preprocess_data(features, target)
            
            # Extract test data
            X_test = processed_data["X_test"]
            y_test = processed_data["y_test"]
            
            print(f"📈 Test samples extracted: {len(X_test)}")
            
            # Get original coordinates from test set
            # We need to map back to original coordinates
            test_indices = X_test.index
            
            # Extract coordinate columns from original features (before scaling)
            coordinate_columns = ['latitude', 'longitude', 'depth_from', 'depth_to']
            
            # Get original coordinate data
            original_coords = features.loc[test_indices, coordinate_columns].copy()
            
            # Create comprehensive test dataset
            test_dataset = pd.DataFrame({
                'latitude': original_coords['latitude'],
                'longitude': original_coords['longitude'], 
                'depth_from': original_coords['depth_from'],
                'depth_to': original_coords['depth_to'],
                'actual_grade_ppm': y_test.values,
                'element': element
            })
            
            # Add sample ID for reference
            test_dataset['test_sample_id'] = range(1, len(test_dataset) + 1)
            
            # Sort by grade to find interesting samples
            test_dataset = test_dataset.sort_values('actual_grade_ppm', ascending=False)
            
            print(f"📊 Test dataset created with {len(test_dataset)} samples")
            print(f"📊 Grade range: {test_dataset['actual_grade_ppm'].min():.2f} - {test_dataset['actual_grade_ppm'].max():.2f} ppm")
            
            # Save to file if requested
            if save_to_file:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"test_samples_{element}_{timestamp}.csv"
                filepath = Path("data/exports") / filename
                filepath.parent.mkdir(parents=True, exist_ok=True)
                
                test_dataset.to_csv(filepath, index=False)
                print(f"💾 Test samples saved to: {filepath}")
            
            # Store in memory
            self.test_samples[element] = test_dataset
            
            return test_dataset
            
        except Exception as e:
            print(f"❌ Error extracting test samples: {str(e)}")
            raise
    
    def show_sample_verification_examples(self, element: str = "CU", num_samples: int = 10):
        """Show examples of how to verify predictions"""
        try:
            if element not in self.test_samples:
                print(f"⚠️  No test samples loaded for {element}. Run extract_test_samples first.")
                return
            
            test_data = self.test_samples[element]
            
            print(f"\n🧪 SAMPLE VERIFICATION EXAMPLES FOR {element}")
            print("="*60)
            
            # Show high-grade samples
            print("📈 HIGH-GRADE SAMPLES (Top 5):")
            print("-" * 40)
            high_grade = test_data.head(5)
            
            for idx, row in high_grade.iterrows():
                print(f"Sample {row['test_sample_id']:3d}: {row['actual_grade_ppm']:8.2f} ppm")
                print(f"   Location: ({row['latitude']:.6f}, {row['longitude']:.6f})")
                print(f"   Depth: {row['depth_from']:.1f} - {row['depth_to']:.1f} m")
                print(f"   Element: {row['element']}")
                print()
            
            # Show medium-grade samples
            print("📊 MEDIUM-GRADE SAMPLES (Random 5):")
            print("-" * 40)
            medium_grade = test_data.iloc[len(test_data)//4:len(test_data)//2].sample(5)
            
            for idx, row in medium_grade.iterrows():
                print(f"Sample {row['test_sample_id']:3d}: {row['actual_grade_ppm']:8.2f} ppm")
                print(f"   Location: ({row['latitude']:.6f}, {row['longitude']:.6f})")
                print(f"   Depth: {row['depth_from']:.1f} - {row['depth_to']:.1f} m")
                print(f"   Element: {row['element']}")
                print()
            
            # Show verification code
            print("🔍 VERIFICATION CODE EXAMPLE:")
            print("-" * 40)
            sample_row = test_data.iloc[0]
            
            verification_code = f'''
# Example verification for Sample {sample_row['test_sample_id']}
from ml_models.inference.predictor import SpatialOreGradePredictor

predictor = SpatialOreGradePredictor()
result = predictor.predict_at_location(
    latitude={sample_row['latitude']:.6f},
    longitude={sample_row['longitude']:.6f},
    depth_from={sample_row['depth_from']:.1f},
    depth_to={sample_row['depth_to']:.1f},
    element='{sample_row['element']}'
)

print(f"Actual grade: {sample_row['actual_grade_ppm']:.2f} ppm")
print(f"Predicted grade: {{result['predicted_grade']:.2f}} ppm")
print(f"Difference: {{abs(result['predicted_grade'] - {sample_row['actual_grade_ppm']:.2f}):.2f}} ppm")
print(f"Accuracy: {{(1 - abs(result['predicted_grade'] - {sample_row['actual_grade_ppm']:.2f}) / {sample_row['actual_grade_ppm']:.2f}) * 100:.1f}}%")
'''
            
            print(verification_code)
            
            return high_grade, medium_grade
            
        except Exception as e:
            print(f"❌ Error showing verification examples: {str(e)}")
            raise
    
    def create_verification_script(self, element: str = "CU", num_samples: int = 20):
        """Create a ready-to-run verification script"""
        try:
            if element not in self.test_samples:
                print(f"⚠️  No test samples loaded for {element}. Run extract_test_samples first.")
                return
            
            test_data = self.test_samples[element]
            
            # Select diverse samples for verification
            high_grade = test_data.head(5)
            medium_grade = test_data.iloc[len(test_data)//4:len(test_data)//2].sample(5)
            low_grade = test_data.tail(10).sample(5)
            random_samples = test_data.sample(5)
            
            verification_samples = pd.concat([high_grade, medium_grade, low_grade, random_samples])
            verification_samples = verification_samples.drop_duplicates().head(num_samples)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            script_content = f'''"""
Manual Verification Script for {element} Model
Generated: {datetime.now().isoformat()}

This script tests the trained {element} model against actual test samples
to verify prediction accuracy at specific coordinates.
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from ml_models.inference.predictor import SpatialOreGradePredictor


def verify_predictions():
    """Verify model predictions against actual test samples"""
    
    print("🧪 MANUAL VERIFICATION OF {element} MODEL")
    print("="*60)
    
    # Initialize predictor
    try:
        predictor = SpatialOreGradePredictor()
        print("✅ Predictor initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing predictor: {{str(e)}}")
        return
    
    # Test samples from held-out test set
    test_samples = [
'''
            
            for idx, row in verification_samples.iterrows():
                script_content += f'''        {{
            'id': {row['test_sample_id']},
            'latitude': {row['latitude']:.6f},
            'longitude': {row['longitude']:.6f},
            'depth_from': {row['depth_from']:.1f},
            'depth_to': {row['depth_to']:.1f},
            'actual_grade': {row['actual_grade_ppm']:.2f},
            'element': '{row['element']}'
        }},
'''
            
            script_content += f'''    ]
    
    print(f"📊 Testing {{len(test_samples)}} samples...")
    print()
    
    results = []
    
    for i, sample in enumerate(test_samples, 1):
        try:
            # Make prediction
            result = predictor.predict_at_location(
                latitude=sample['latitude'],
                longitude=sample['longitude'],
                depth_from=sample['depth_from'],
                depth_to=sample['depth_to'],
                element=sample['element']
            )
            
            predicted_grade = result['predicted_grade']
            actual_grade = sample['actual_grade']
            
            # Calculate accuracy metrics
            absolute_error = abs(predicted_grade - actual_grade)
            relative_error = (absolute_error / actual_grade) * 100 if actual_grade != 0 else 0
            accuracy = max(0, 100 - relative_error)
            
            # Store result
            results.append({{
                'sample_id': sample['id'],
                'actual': actual_grade,
                'predicted': predicted_grade,
                'absolute_error': absolute_error,
                'relative_error': relative_error,
                'accuracy': accuracy
            }})
            
            # Print individual result
            print(f"Sample {{sample['id']:2d}}: {{actual_grade:8.2f}} ppm → {{predicted_grade:8.2f}} ppm | Error: {{absolute_error:6.2f}} ppm ({{relative_error:5.1f}}%) | Accuracy: {{accuracy:5.1f}}%")
            
        except Exception as e:
            print(f"❌ Error with sample {{sample['id']}}: {{str(e)}}")
            results.append({{
                'sample_id': sample['id'],
                'actual': sample['actual_grade'],
                'predicted': None,
                'absolute_error': None,
                'relative_error': None,
                'accuracy': None,
                'error': str(e)
            }})
    
    # Calculate overall statistics
    successful_predictions = [r for r in results if r['predicted'] is not None]
    
    if successful_predictions:
        avg_accuracy = sum(r['accuracy'] for r in successful_predictions) / len(successful_predictions)
        avg_absolute_error = sum(r['absolute_error'] for r in successful_predictions) / len(successful_predictions)
        avg_relative_error = sum(r['relative_error'] for r in successful_predictions) / len(successful_predictions)
        
        print()
        print("📊 OVERALL STATISTICS:")
        print("-" * 30)
        print(f"Successful predictions: {{len(successful_predictions)}}/{{len(test_samples)}}")
        print(f"Average accuracy: {{avg_accuracy:.1f}}%")
        print(f"Average absolute error: {{avg_absolute_error:.2f}} ppm")
        print(f"Average relative error: {{avg_relative_error:.1f}}%")
        
        # Performance assessment
        if avg_accuracy >= 90:
            print("✅ EXCELLENT: Model performance is excellent!")
        elif avg_accuracy >= 80:
            print("✅ GOOD: Model performance is good!")
        elif avg_accuracy >= 70:
            print("⚠️  FAIR: Model performance is fair, consider retraining")
        else:
            print("❌ POOR: Model performance is poor, retraining recommended")
    
    return results


if __name__ == "__main__":
    results = verify_predictions()
    
    print()
    print("🎯 VERIFICATION COMPLETED!")
    print("Use these results to assess model accuracy on real coordinates.")
'''
            
            # Save verification script
            script_filename = f"verify_{element}_model_{timestamp}.py"
            script_path = Path("data/exports") / script_filename
            script_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(script_path, 'w') as f:
                f.write(script_content)
            
            print(f"📄 Verification script created: {script_path}")
            print(f"🚀 Run it with: python {script_path}")
            
            return script_path
            
        except Exception as e:
            print(f"❌ Error creating verification script: {str(e)}")
            raise
    
    def show_test_sample_locations(self, element: str = "CU"):
        """Show geographic distribution of test samples"""
        try:
            if element not in self.test_samples:
                print(f"⚠️  No test samples loaded for {element}. Run extract_test_samples first.")
                return
            
            test_data = self.test_samples[element]
            
            print(f"\n🗺️  TEST SAMPLE GEOGRAPHIC DISTRIBUTION FOR {element}")
            print("="*60)
            
            # Geographic bounds
            lat_min, lat_max = test_data['latitude'].min(), test_data['latitude'].max()
            lon_min, lon_max = test_data['longitude'].min(), test_data['longitude'].max()
            depth_min, depth_max = test_data['depth_from'].min(), test_data['depth_to'].max()
            
            print(f"📍 Geographic Coverage:")
            print(f"   Latitude: {lat_min:.6f} to {lat_max:.6f}")
            print(f"   Longitude: {lon_min:.6f} to {lon_max:.6f}")
            print(f"   Depth: {depth_min:.1f} to {depth_max:.1f} meters")
            
            # Grade distribution
            print(f"\\n📊 Grade Distribution:")
            print(f"   Min: {test_data['actual_grade_ppm'].min():.2f} ppm")
            print(f"   Max: {test_data['actual_grade_ppm'].max():.2f} ppm")
            print(f"   Mean: {test_data['actual_grade_ppm'].mean():.2f} ppm")
            print(f"   Median: {test_data['actual_grade_ppm'].median():.2f} ppm")
            print(f"   Std Dev: {test_data['actual_grade_ppm'].std():.2f} ppm")
            
            # Show sample locations by grade ranges
            print(f"\\n📈 Sample Locations by Grade Range:")
            print("-" * 40)
            
            # High grade (top 10%)
            high_threshold = test_data['actual_grade_ppm'].quantile(0.9)
            high_grade_samples = test_data[test_data['actual_grade_ppm'] >= high_threshold]
            
            print(f"HIGH GRADE (≥{high_threshold:.2f} ppm): {len(high_grade_samples)} samples")
            for idx, row in high_grade_samples.head(5).iterrows():
                print(f"   ({row['latitude']:.6f}, {row['longitude']:.6f}) - {row['actual_grade_ppm']:.2f} ppm")
            
            # Medium grade (25-75%)
            med_low = test_data['actual_grade_ppm'].quantile(0.25)
            med_high = test_data['actual_grade_ppm'].quantile(0.75)
            med_grade_samples = test_data[(test_data['actual_grade_ppm'] >= med_low) & 
                                        (test_data['actual_grade_ppm'] < med_high)]
            
            print(f"\\nMEDIUM GRADE ({med_low:.2f}-{med_high:.2f} ppm): {len(med_grade_samples)} samples")
            for idx, row in med_grade_samples.head(5).iterrows():
                print(f"   ({row['latitude']:.6f}, {row['longitude']:.6f}) - {row['actual_grade_ppm']:.2f} ppm")
            
            return test_data
            
        except Exception as e:
            print(f"❌ Error showing test sample locations: {str(e)}")
            raise


def main():
    """Main execution function"""
    try:
        extractor = TestSampleExtractor()
        
        # Choose element to analyze
        element = "CU"  # Change this to analyze different elements
        
        print("🔍 TEST SAMPLE EXTRACTION AND VERIFICATION")
        print("="*80)
        
        # Step 1: Extract test samples
        print("1. EXTRACTING TEST SAMPLES")
        print("-" * 30)
        test_samples = extractor.extract_test_samples(element=element)
        
        # Step 2: Show verification examples
        print("\\n2. VERIFICATION EXAMPLES")
        print("-" * 30)
        extractor.show_sample_verification_examples(element=element)
        
        # Step 3: Create verification script
        print("\\n3. CREATING VERIFICATION SCRIPT")
        print("-" * 30)
        script_path = extractor.create_verification_script(element=element)
        
        # Step 4: Show geographic distribution
        print("\\n4. GEOGRAPHIC DISTRIBUTION")
        print("-" * 30)
        extractor.show_test_sample_locations(element=element)
        
        print(f"\\n🎉 TEST SAMPLE EXTRACTION COMPLETED!")
        print(f"✅ Test samples saved to CSV file")
        print(f"✅ Verification script created: {script_path}")
        print(f"✅ Use these coordinates to manually verify model accuracy")
        
    except Exception as e:
        print(f"❌ Error in test sample extraction: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
