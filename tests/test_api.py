"""
API tests for MLOps Error Prediction System.
File: tests/test_api.py
"""

import requests
import json
import sys
from pathlib import Path

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent))

def test_api():
    """Test the prediction API"""
    print("🧪 Testing MLOps Error Prediction API")
    print("=" * 50)
    
    base_url = "http://localhost:8000"
    
    # Test 1: Health check
    print("\n1. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            health_data = response.json()
            print("✅ Health check passed")
            print(f"   Status: {health_data['status']}")
            print(f"   Models loaded: {health_data['models_loaded']}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException:
        print("❌ API is not running. Start it first with: python main.py")
        return False
    
    # Test 2: Predictions with different scenarios
    test_scenarios = [
        {
            "name": "Normal System",
            "data": {
                "cpu_usage": 30,
                "memory_usage": 40,
                "disk_usage": 25,
                "network_latency_ms": 50,
                "error_count": 0,
                "response_time_ms": 200,
                "active_connections": 25
            }
        },
        {
            "name": "Critical System State",
            "data": {
                "cpu_usage": 95,
                "memory_usage": 92,
                "disk_usage": 88,
                "network_latency_ms": 300,
                "error_count": 15,
                "response_time_ms": 2500,
                "active_connections": 200
            }
        }
    ]
    
    print("\n2. Testing predictions...")
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n2.{i} {scenario['name']}:")
        try:
            response = requests.post(
                f"{base_url}/predict",
                json=scenario['data'],
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Prediction successful:")
                print(f"   Risk Level: {result['failure_risk']}")
                print(f"   Probability: {result['failure_probability']:.4f}")
                print(f"   Confidence: {result['confidence']:.3f}")
                print(f"   Model Used: {result['model_used']}")
                
            else:
                print(f"❌ Prediction failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Prediction error: {str(e)}")
    
    print("\n🎉 API testing completed!")
    return True

def main():
    """Main function"""
    try:
        success = test_api()
        if success:
            print("\n✅ All tests completed successfully!")
        else:
            print("\n❌ Some tests failed!")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n⏹️ Testing interrupted by user")
    except Exception as e:
        print(f"\n❌ Testing failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
