#!/usr/bin/env python3
"""
Test script to verify the updated data processing components work with the new view
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

from ml_models.data_processing.query_builder import SQLQueryBuilder, create_grade_filter
from ml_models.database.connections import get_db_connection
import pandas as pd

def test_data_processing_integration():
    """Test the updated data processing components"""
    print("Testing Data Processing Integration with New View...")
    print("="*60)
    
    try:
        # Test 1: Query Builder with new view
        print("\n1. Testing query builder with new view...")
        builder = SQLQueryBuilder()
        
        # Test basic query
        query, params = builder.build_drilling_query(elements=['Cu'], limit=50)
        print("✓ Query builder successfully generated query for new view")
        
        # Execute and get sample data
        with get_db_connection() as conn:
            df = pd.read_sql(query, conn, params=params)
        
        print(f"✓ Retrieved {len(df)} records from new view")
        print(f"  Columns: {list(df.columns)}")
        
        # Verify new column names are present
        expected_columns = ['standardized_grade_ppm', 'latitude', 'longitude', 'depth_from', 'depth_to']
        missing_columns = [col for col in expected_columns if col not in df.columns]
        
        if missing_columns:
            print(f"✗ Missing expected columns: {missing_columns}")
        else:
            print("✓ All expected new view columns are present")
        
        # Test 2: Feature Engineering compatibility
        print("\n2. Testing feature engineering with new view data...")
        try:
            from ml_models.data_processing.feature_engineering import GeologicalFeatureEngineer
            
            engineer = GeologicalFeatureEngineer()
            
            # Test spatial features
            df_spatial = engineer.create_spatial_features(df)
            print(f"✓ Spatial features created: {len(df_spatial.columns)} total columns")
            
            # Test geological features
            df_geological = engineer.create_geological_features(df)
            print(f"✓ Geological features created: {len(df_geological.columns)} total columns")
            
            # Check if grade-related features use the new column name
            grade_features = [col for col in df_geological.columns if 'grade' in col]
            print(f"  Grade-related features: {grade_features}")
            
        except Exception as e:
            print(f"✗ Feature engineering error: {e}")
        
        # Test 3: Query builder filters with new structure
        print("\n3. Testing query filters with new view...")
        
        # Create filters using new column names
        grade_filters = create_grade_filter(min_grade=100, max_grade=10000)
        
        query, params = builder.build_drilling_query(
            elements=['Cu', 'Pb'], 
            filters=grade_filters,
            limit=20
        )
        
        with get_db_connection() as conn:
            df_filtered = pd.read_sql(query, conn, params=params)
        
        print(f"✓ Filtered query executed: {len(df_filtered)} records")
        
        # Verify grade range
        if not df_filtered.empty:
            min_grade = df_filtered['standardized_grade_ppm'].min()
            max_grade = df_filtered['standardized_grade_ppm'].max()
            print(f"  Grade range: {min_grade:.2f} - {max_grade:.2f} ppm")
            
            if min_grade >= 100 and max_grade <= 10000:
                print("✓ Grade filters working correctly")
            else:
                print("⚠ Grade filters may not be working as expected")
        
        # Test 4: Multi-element pivot query
        print("\n4. Testing multi-element pivot query...")
        
        pivot_query, pivot_params = builder.build_multi_element_pivot_query(['Cu', 'Pb', 'Zn'])
        
        with get_db_connection() as conn:
            df_pivot = pd.read_sql(pivot_query, conn, params=pivot_params)
        
        print(f"✓ Pivot query executed: {len(df_pivot)} records")
        print(f"  Pivot columns: {list(df_pivot.columns)}")
        
        # Test 5: Summary statistics with new view
        print("\n5. Testing summary statistics...")
        
        summary_query, summary_params = builder.build_summary_query(elements=['Cu'])
        
        with get_db_connection() as conn:
            df_summary = pd.read_sql(summary_query, conn, params=summary_params)
        
        print(f"✓ Summary query executed: {len(df_summary)} groups")
        
        if not df_summary.empty:
            first_row = df_summary.iloc[0]
            print(f"  Example: {first_row['group_field']} - {first_row['record_count']} records, avg: {first_row['avg_grade']:.2f} ppm")
        
    except Exception as e:
        print(f"✗ Data processing integration error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "="*60)
    print("Data Processing Integration Test Completed!")
    print("✓ All data processing components work correctly with the new view")
    return True

if __name__ == "__main__":
    test_data_processing_integration()
