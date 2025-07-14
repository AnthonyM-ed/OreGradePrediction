"""
Test script to validate the new view integration
===============================================

This script tests the updated db_loader to ensure it works with the new view.
"""

import sys
import os
from pathlib import Path

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent.parent
sys.path.append(str(backend_dir))

# Django setup
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
import django
django.setup()

from ml_models.data_processing.db_loader import XGBoostGeologicalDataLoader
import pandas as pd
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_new_view_integration():
    """Test the new view integration with various methods"""
    
    try:
        # Initialize data loader
        data_loader = XGBoostGeologicalDataLoader()
        
        # Test 1: Get data summary
        logger.info("Testing data summary...")
        summary = data_loader.get_data_summary()
        print(f"Total records: {summary['total_records']}")
        print(f"Unique holes: {summary['unique_holes']}")
        print(f"Unique samples: {summary['unique_samples']}")
        print(f"Available elements: {[elem['Element'] for elem in summary['elements']]}")
        print(f"Available datasets: {[ds['DataSet'] for ds in summary['datasets']]}")
        
        # Test 2: Load drilling data for a specific element
        logger.info("Testing drilling data loading...")
        drilling_data = data_loader.load_drilling_data(element='Cu')
        print(f"Loaded {len(drilling_data)} records for Cu")
        print(f"Columns: {list(drilling_data.columns)}")
        
        # Test 3: Load multi-element data
        logger.info("Testing multi-element data loading...")
        multi_data = data_loader.load_multi_element_data(
            elements=['Cu', 'Au'], 
            datasets=['Drilling_INF', 'Drilling_BF']
        )
        print(f"Loaded {len(multi_data)} records for Cu and Au")
        
        # Test 4: Load training data
        logger.info("Testing training data loading...")
        X_train, X_test, y_train, y_test = data_loader.load_xgboost_training_data(
            elements=['Cu'], 
            datasets=['Drilling_INF']
        )
        print(f"Training data shape: {X_train.shape}")
        print(f"Test data shape: {X_test.shape}")
        print(f"Feature names: {data_loader.feature_names[:10]}...")  # Show first 10 features
        
        # Test 5: Check target variable (should be standardized_grade_ppm)
        print(f"Target variable range: {y_train.min():.2f} - {y_train.max():.2f} ppm")
        print(f"Target variable mean: {y_train.mean():.2f} ppm")
        
        # Test 6: Load spatial context data
        logger.info("Testing spatial context loading...")
        if len(drilling_data) > 0:
            center_lat = drilling_data['latitude'].mean()
            center_lon = drilling_data['longitude'].mean()
            spatial_data = data_loader.load_spatial_context(
                center_lat=center_lat,
                center_lon=center_lon,
                radius_km=5.0,
                elements=['Cu']
            )
            print(f"Loaded {len(spatial_data)} records within 5km of center")
        
        # Test 7: Load by lab code
        logger.info("Testing lab code filtering...")
        if summary['lab_codes']:
            first_lab = summary['lab_codes'][0]['LabCode']
            lab_data = data_loader.load_by_lab_code(lab_code=first_lab, elements=['Cu'])
            print(f"Loaded {len(lab_data)} records for lab code {first_lab}")
        
        logger.info("All tests completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Clean up
        if 'data_loader' in locals():
            data_loader.close_connection()

def main():
    """Main function"""
    print("Testing new view integration...")
    success = test_new_view_integration()
    
    if success:
        print("\n✅ All tests passed! The new view integration is working correctly.")
        print("\nYou can now use the updated db_loader with:")
        print("- standardized_grade_ppm as target variable")
        print("- Depth_From, Depth_To, Interval_Length for depth features")
        print("- SampleID for sample identification")
        print("- LabCode for lab-specific filtering")
        print("- Dataset filtering (Drilling_INF, Drilling_BF, Drilling_OP)")
    else:
        print("\n❌ Tests failed! Check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
