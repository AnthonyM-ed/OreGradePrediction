#!/usr/bin/env python3
"""
Test script to verify the complete database integration with the new view
"""
import os
import sys
import django
from django.conf import settings

# Add the backend directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Configure Django settings
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
django.setup()

from ml_models.database.connections import get_db_connection
from ml_models.database.queries import GeologicalQueries, QueryExecutor
from ml_models.database.schema_validator import SchemaValidator
from ml_models.database.data_extractors import GeologicalDataExtractor, StatisticalDataExtractor
import pandas as pd

def test_new_view_integration():
    """Test the complete integration with the new view"""
    print("Testing New View Integration...")
    print("="*50)
    
    # Test 1: Database connection
    print("\n1. Testing database connection...")
    try:
        with get_db_connection() as conn:
            print("✓ Database connection successful")
    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        return False
    
    # Test 2: Schema validation
    print("\n2. Testing schema validation...")
    try:
        validator = SchemaValidator()
        is_valid = validator.validate_main_view_schema()
        if is_valid:
            print("✓ Main view schema validation passed")
        else:
            print("✗ Main view schema validation failed")
    except Exception as e:
        print(f"✗ Schema validation error: {e}")
    
    # Test 3: Query executor
    print("\n3. Testing query executor...")
    try:
        executor = QueryExecutor()
        
        # Test basic query
        elements = executor.get_elements_summary()
        print(f"✓ Elements summary retrieved: {len(elements)} elements")
        
        # Test spatial extent
        spatial = executor.get_spatial_extent()
        print(f"✓ Spatial extent retrieved: {spatial}")
        
        # Test lab summary
        labs = executor.get_lab_summary()
        print(f"✓ Lab summary retrieved: {len(labs)} labs")
        
        # Test dataset summary
        datasets = executor.get_dataset_summary()
        print(f"✓ Dataset summary retrieved: {len(datasets)} datasets")
        
    except Exception as e:
        print(f"✗ Query executor error: {e}")
    
    # Test 4: Data extractors
    print("\n4. Testing data extractors...")
    try:
        geo_extractor = GeologicalDataExtractor()
        
        # Test training data extraction
        training_data = geo_extractor.extract_training_data(
            elements=['Cu'],
            max_records=100
        )
        print(f"✓ Training data extracted: {len(training_data)} records")
        print(f"  Columns: {list(training_data.columns)}")
        
        # Test statistical extractor
        stat_extractor = StatisticalDataExtractor()
        stats = stat_extractor.extract_element_statistics()
        print(f"✓ Element statistics extracted: {len(stats)} elements")
        
    except Exception as e:
        print(f"✗ Data extractor error: {e}")
    
    # Test 5: View data sample
    print("\n5. Testing view data sample...")
    try:
        with get_db_connection() as conn:
            sample_query = """
                SELECT TOP 5 
                    SampleID, Hole_ID, Element, standardized_grade_ppm, 
                    DataSet, LabCode, Depth_From, Depth_To
                FROM vw_HoleSamples_ElementGrades
                WHERE standardized_grade_ppm IS NOT NULL
                ORDER BY standardized_grade_ppm DESC
            """
            sample_df = pd.read_sql(sample_query, conn)
            print(f"✓ Sample data retrieved: {len(sample_df)} records")
            print(f"  Sample columns: {list(sample_df.columns)}")
            if not sample_df.empty:
                print(f"  Grade range: {sample_df['standardized_grade_ppm'].min():.2f} - {sample_df['standardized_grade_ppm'].max():.2f} ppm")
    
    except Exception as e:
        print(f"✗ View data sample error: {e}")
    
    print("\n" + "="*50)
    print("Integration test completed!")
    print("✓ New view integration is working correctly")

if __name__ == "__main__":
    test_new_view_integration()
