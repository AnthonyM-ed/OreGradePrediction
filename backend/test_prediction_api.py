"""
Test Ore Grade Prediction API

This script tests the prediction API endpoint to ensure it works correctly
before integrating with the frontend.
"""

import requests
import json

def test_prediction_api():
    """Test the ore grade prediction API"""
    
    # API endpoint
    url = "http://127.0.0.1:8000/api/predict-ore-grade/"
    
    # Test data - using coordinates from your mining area
    test_data = {
        "element": "CU",
        "latitude": -19.123456,
        "longitude": -69.654321,
        "depth_from": 0.0,
        "depth_to": 10.0
    }
    
    print("🧪 TESTING ORE GRADE PREDICTION API")
    print("="*50)
    print(f"URL: {url}")
    print(f"Test data: {json.dumps(test_data, indent=2)}")
    print()
    
    try:
        # Make the request
        print("📡 Making API request...")
        response = requests.post(url, json=test_data, timeout=30)
        
        print(f"📊 Response status: {response.status_code}")
        print(f"📊 Response headers: {dict(response.headers)}")
        print()
        
        if response.status_code == 200:
            result = response.json()
            print("✅ SUCCESS! Prediction API working correctly")
            print("📋 Prediction results:")
            print(json.dumps(result, indent=2))
            
            # Validate response structure
            expected_fields = ['predicted_grade', 'element', 'latitude', 'longitude', 'depth_from', 'depth_to']
            missing_fields = [field for field in expected_fields if field not in result]
            
            if missing_fields:
                print(f"⚠️  Warning: Missing expected fields: {missing_fields}")
            else:
                print("✅ All expected fields present in response")
                
        else:
            print(f"❌ API request failed with status {response.status_code}")
            try:
                error_data = response.json()
                print("Error details:")
                print(json.dumps(error_data, indent=2))
            except:
                print("Response text:", response.text)
    
    except requests.exceptions.ConnectionError:
        print("❌ Connection error - is the Django server running?")
        print("💡 Start the server with: python manage.py runserver")
        
    except requests.exceptions.Timeout:
        print("❌ Request timeout - server took too long to respond")
        
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")

def test_available_models_api():
    """Test the available models API"""
    
    url = "http://127.0.0.1:8000/api/available-models/"
    
    print("\n🔍 TESTING AVAILABLE MODELS API")
    print("="*40)
    print(f"URL: {url}")
    
    try:
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Available models API working correctly")
            print("📋 Available models:")
            print(json.dumps(result, indent=2))
        else:
            print(f"❌ API request failed with status {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")

def test_multiple_elements():
    """Test predictions for multiple elements"""
    
    elements_to_test = ['CU', 'AU', 'AG']
    url = "http://127.0.0.1:8000/api/predict-ore-grade/"
    
    print("\n🔬 TESTING MULTIPLE ELEMENTS")
    print("="*35)
    
    for element in elements_to_test:
        print(f"\n🧪 Testing {element}...")
        
        test_data = {
            "element": element,
            "latitude": -19.123456,
            "longitude": -69.654321,
            "depth_from": 0.0,
            "depth_to": 10.0
        }
        
        try:
            response = requests.post(url, json=test_data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                predicted_grade = result.get('predicted_grade', 'N/A')
                print(f"   ✅ {element}: {predicted_grade} ppm")
            elif response.status_code == 404:
                print(f"   ⚠️  {element}: No trained model found")
            else:
                print(f"   ❌ {element}: Error {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ {element}: Error - {str(e)}")

if __name__ == "__main__":
    print("🔬 ORE GRADE PREDICTION API TESTS")
    print("="*60)
    print("Make sure the Django server is running before testing!")
    print("Start with: python manage.py runserver")
    print()
    
    # Run tests
    test_prediction_api()
    test_available_models_api()
    test_multiple_elements()
    
    print("\n🎯 TESTING COMPLETED!")
    print("If all tests pass, your frontend integration should work correctly.")
