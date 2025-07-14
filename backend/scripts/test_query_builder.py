#!/usr/bin/env python3
"""
Test script to verify the query builder with the new view
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

from ml_models.data_processing.query_builder import SQLQueryBuilder, create_grade_filter, create_dataset_filter
from ml_models.database.connections import get_db_connection

def test_query_builder():
    """Test the query builder with the new view"""
    print("Testing Query Builder with New View...")
    print("="*50)
    
    try:
        # Initialize query builder
        builder = SQLQueryBuilder()
        
        # Test 1: Basic drilling query
        print("\n1. Testing basic drilling query...")
        query, params = builder.build_drilling_query(elements=['Cu', 'Pb'])
        print(f"✓ Query generated with {len(params)} parameters")
        print(f"  Parameters: {params}")
        
        # Test with database
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchmany(5)
            print(f"✓ Query executed successfully, retrieved {len(rows)} sample records")
            
            if rows:
                for row in rows:
                    print(f"  {row.Hole_ID} | {row.Element} | {row.standardized_grade_ppm:.2f} ppm | {row.DataSet}")
        
        # Test 2: Summary query
        print("\n2. Testing summary query...")
        query, params = builder.build_summary_query(elements=['Cu'])
        print(f"✓ Summary query generated with {len(params)} parameters")
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            print(f"✓ Summary query executed successfully, {len(rows)} groups found")
            
            if rows:
                for row in rows:
                    print(f"  {row.group_field}: {row.record_count} records, avg: {row.avg_grade:.2f} ppm")
        
        # Test 3: Quality check query
        print("\n3. Testing quality check query...")
        query, params = builder.build_quality_check_query()
        print(f"✓ Quality check query generated with {len(params)} parameters")
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            print(f"✓ Quality check executed successfully, {len(rows)} checks performed")
            
            if rows:
                for row in rows:
                    if row.percentage:
                        print(f"  {row.check_type}: {row.count_value} ({row.percentage}%)")
                    else:
                        print(f"  {row.check_type}: {row.count_value}")
        
        # Test 4: Query with filters
        print("\n4. Testing queries with filters...")
        grade_filters = create_grade_filter(min_grade=1000, max_grade=50000)
        dataset_filter = create_dataset_filter(['Drilling_BF', 'Drilling_INF'])
        all_filters = grade_filters + [dataset_filter]
        
        query, params = builder.build_drilling_query(
            elements=['Cu'], 
            filters=all_filters,
            limit=10
        )
        print(f"✓ Filtered query generated with {len(params)} parameters")
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            print(f"✓ Filtered query executed successfully, {len(rows)} records found")
            
            if rows:
                for row in rows:
                    print(f"  {row.Hole_ID} | {row.Element} | {row.standardized_grade_ppm:.2f} ppm | {row.DataSet}")
        
        # Test 5: Anomaly detection query
        print("\n5. Testing anomaly detection query...")
        query, params = builder.build_anomaly_detection_query(element='Cu', std_threshold=2.5)
        print(f"✓ Anomaly detection query generated with {len(params)} parameters")
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchmany(5)
            print(f"✓ Anomaly detection executed successfully, {len(rows)} top anomalies found")
            
            if rows:
                for row in rows:
                    print(f"  {row.Hole_ID} | {row.standardized_grade_ppm:.2f} ppm | Z-score: {row.z_score:.2f} | {row.anomaly_flag}")
        
    except Exception as e:
        print(f"✗ Query builder test error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "="*50)
    print("Query Builder test completed successfully!")
    print("✓ All query types work correctly with the new view")
    return True

if __name__ == "__main__":
    test_query_builder()
