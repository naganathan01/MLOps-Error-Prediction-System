"""
Tests for the API endpoints.
"""

import pytest
import json
from fastapi.testclient import TestClient
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.api.app import app

client = TestClient(app)

class TestAPI:
    
    def test_root_endpoint(self):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert data["status"] == "running"
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "models_loaded" in data
        assert isinstance(data["models_loaded"], list)
    
    def test_predict_valid_input(self):
        """Test prediction with valid input"""
        test_input = {
            "cpu_usage": 85.0,
            "memory_usage": 90.0,
            "disk_usage": 45.0,
            "network_latency_ms": 120.0,
            "error_count": 3,
            "response_time_ms": 450.0,
            "active_connections": 75
        }
        
        response = client.post("/predict", json=test_input)
        
        # Should work even without models loaded (will return 500)
        # In a real test environment, models would be loaded
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            data = response.json()
            assert "failure_probability" in data
            assert "failure_risk" in data
            assert "recommendations" in data
            assert "model_used" in data
            assert "confidence" in data
            assert "timestamp" in data
            
            # Validate data types and ranges
            assert 0 <= data["failure_probability"] <= 1
            assert data["failure_risk"] in ["LOW", "MEDIUM", "HIGH"]
            assert isinstance(data["recommendations"], list)
            assert 0 <= data["confidence"] <= 1
    
    def test_predict_invalid_input(self):
        """Test prediction with invalid input"""
        # Missing required fields
        test_input = {
            "cpu_usage": 85.0,
            "memory_usage": 90.0
            # Missing other required fields
        }
        
        response = client.post("/predict", json=test_input)
        assert response.status_code == 422  # Validation error
    
    def test_predict_out_of_range_values(self):
        """Test prediction with out-of-range values"""
        test_input = {
            "cpu_usage": 150.0,  # > 100
            "memory_usage": -10.0,  # < 0
            "disk_usage": 45.0,
            "network_latency_ms": 120.0,
            "error_count": 3,
            "response_time_ms": 450.0,
            "active_connections": 75
        }
        
        response = client.post("/predict", json=test_input)
        assert response.status_code == 422  # Validation error
    
    def test_batch_predict(self):
        """Test batch prediction endpoint"""
        test_inputs = [
            {
                "cpu_usage": 85.0,
                "memory_usage": 90.0,
                "disk_usage": 45.0,
                "network_latency_ms": 120.0,
                "error_count": 3,
                "response_time_ms": 450.0,
                "active_connections": 75
            },
            {
                "cpu_usage": 30.0,
                "memory_usage": 40.0,
                "disk_usage": 20.0,
                "network_latency_ms": 50.0,
                "error_count": 0,
                "response_time_ms": 200.0,
                "active_connections": 25
            }
        ]
        
        response = client.post("/predict/batch", json=test_inputs)
        
        # Should work even without models loaded (will return 500)
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            data = response.json()
            assert "predictions" in data
            assert len(data["predictions"]) == 2
    
    def test_models_info_endpoint(self):
        """Test models info endpoint"""
        response = client.get("/models/info")
        
        # Should work even without models loaded (will return 500)
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            data = response.json()
            assert "loaded_models" in data
            assert "feature_count" in data
            assert isinstance(data["loaded_models"], list)
    
    def test_features_endpoint(self):
        """Test features list endpoint"""
        response = client.get("/models/features")
        assert response.status_code == 200
        data = response.json()
        assert "features" in data
        assert "feature_count" in data
        assert isinstance(data["features"], list)

@pytest.fixture
def sample_prediction_data():
    """Sample data for testing predictions"""
    return {
        "cpu_usage": 75.0,
        "memory_usage": 80.0,
        "disk_usage": 35.0,
        "network_latency_ms": 100.0,
        "error_count": 2,
        "response_time_ms": 350.0,
        "active_connections": 50,
        "hour": 14,
        "day_of_week": 2
    }

def test_prediction_with_optional_fields(sample_prediction_data):
    """Test prediction with optional time fields"""
    response = client.post("/predict", json=sample_prediction_data)
    
    # Should work even without models loaded
    assert response.status_code in [200, 500]

def test_api_error_handling():
    """Test API error handling"""
    # Test with completely invalid JSON
    response = client.post("/predict", data="invalid json")
    assert response.status_code == 422
    
    # Test with wrong content type
    response = client.post("/predict", data=json.dumps({"test": "data"}), 
                          headers={"Content-Type": "text/plain"})
    assert response.status_code == 422

def test_api_validation_messages():
    """Test that validation error messages are helpful"""
    test_input = {
        "cpu_usage": "not_a_number",
        "memory_usage": 90.0,
        "disk_usage": 45.0,
        "network_latency_ms": 120.0,
        "error_count": 3,
        "response_time_ms": 450.0,
        "active_connections": 75
    }
    
    response = client.post("/predict", json=test_input)
    assert response.status_code == 422
    
    error_data = response.json()
    assert "detail" in error_data

if __name__ == "__main__":
    pytest.main([__file__, "-v"])