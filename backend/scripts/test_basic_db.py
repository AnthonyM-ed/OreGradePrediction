#!/usr/bin/env python3
"""
Simple test script to verify the database integration with the new view
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

# Test basic database connection
def test_basic_connection():
    """Test basic database connection"""
    print("Testing Database Connection...")
    print("="*50)
    
    try:
        from ml_models.database.connections import get_db_connection
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Test view existence
            cursor.execute("SELECT COUNT(*) FROM INFORMATION_SCHEMA.VIEWS WHERE TABLE_NAME = 'vw_HoleSamples_ElementGrades'")
            view_exists = cursor.fetchone()[0]
            
            if view_exists:
                print("✓ View vw_HoleSamples_ElementGrades exists")
                
                # Test view data
                cursor.execute("SELECT COUNT(*) FROM vw_HoleSamples_ElementGrades")
                record_count = cursor.fetchone()[0]
                print(f"✓ View contains {record_count} records")
                
                # Test sample data
                cursor.execute("""
                    SELECT TOP 5 
                        SampleID, Hole_ID, Element, standardized_grade_ppm, 
                        DataSet, LabCode, Depth_From, Depth_To
                    FROM vw_HoleSamples_ElementGrades
                    WHERE standardized_grade_ppm IS NOT NULL
                    ORDER BY standardized_grade_ppm DESC
                """)
                
                rows = cursor.fetchall()
                print(f"✓ Retrieved {len(rows)} sample records")
                
                if rows:
                    print("Sample data:")
                    for row in rows:
                        print(f"  {row.SampleID} | {row.Hole_ID} | {row.Element} | {row.standardized_grade_ppm:.2f} ppm")
                
                # Test element distribution
                cursor.execute("""
                    SELECT Element, COUNT(*) as count, 
                           AVG(standardized_grade_ppm) as avg_grade
                    FROM vw_HoleSamples_ElementGrades
                    WHERE standardized_grade_ppm IS NOT NULL
                    GROUP BY Element
                    ORDER BY count DESC
                """)
                
                elements = cursor.fetchall()
                print(f"✓ Found {len(elements)} elements in the view")
                
                for elem in elements[:5]:  # Show top 5
                    print(f"  {elem.Element}: {elem.count} samples, avg: {elem.avg_grade:.2f} ppm")
                
                # Test dataset distribution
                cursor.execute("""
                    SELECT DataSet, COUNT(*) as count
                    FROM vw_HoleSamples_ElementGrades
                    WHERE standardized_grade_ppm IS NOT NULL
                    GROUP BY DataSet
                    ORDER BY count DESC
                """)
                
                datasets = cursor.fetchall()
                print(f"✓ Found {len(datasets)} datasets")
                
                for ds in datasets:
                    print(f"  {ds.DataSet}: {ds.count} samples")
                
            else:
                print("✗ View vw_HoleSamples_ElementGrades does not exist")
                
    except Exception as e:
        print(f"✗ Database connection error: {e}")
        return False
    
    print("\n" + "="*50)
    print("Basic database test completed!")
    return True

if __name__ == "__main__":
    test_basic_connection()
