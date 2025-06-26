"""
Complete enhanced pipeline runner that generates better data, engineers features, 
trains models, and starts the improved prediction API.
"""

import sys
import subprocess
import logging
from pathlib import Path
from datetime import datetime
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_enhanced_pipeline():
    """Run the complete enhanced MLOps pipeline"""
    
    print("🚀 Starting Enhanced MLOps Error Prediction Pipeline")
    print("=" * 80)
    
    start_time = datetime.now()
    
    try:
        # Step 1: Generate Enhanced Data
        print("\n📊 STEP 1: GENERATING ENHANCED DATA")
        print("-" * 50)
        
        from src.data.data_generator import EnhancedSystemDataGenerator
        
        generator = EnhancedSystemDataGenerator()
        metrics_df, logs_df = generator.generate_all_data(n_days=30, failure_rate=0.35)
        
        print(f"✅ Generated {len(metrics_df)} metrics and {len(logs_df)} logs")
        print(f"   Failure rate: {metrics_df['failure_within_hour'].mean():.2%}")
        
        # Step 2: Enhanced Feature Engineering
        print("\n🔧 STEP 2: ENHANCED FEATURE ENGINEERING")
        print("-" * 50)
        
        from src.features.feature_engineering import EnhancedFeatureEngineer
        
        engineer = EnhancedFeatureEngineer()
        features_df = engineer.process_enhanced_features()
        
        print(f"✅ Created {len(features_df.columns)} enhanced features")
        print(f"   Dataset shape: {features_df.shape}")
        
        # Step 3: Enhanced Model Training
        print("\n🤖 STEP 3: ENHANCED MODEL TRAINING")
        print("-" * 50)
        
        from src.models.training import EnhancedModelTrainer
        
        trainer = EnhancedModelTrainer()
        results, best_model_name, best_model = trainer.train_all_enhanced_models(use_search=True)
        
        if not results:
            logger.error("❌ Model training failed")
            return False
        
        print(f"✅ Training completed - Best model: {best_model_name}")
        print(f"   Best AUC: {results[best_model_name]['auc_score']:.4f}")
        
        # Step 4: Validate Models
        print("\n🔍 STEP 4: MODEL VALIDATION")
        print("-" * 50)
        
        validation_passed = validate_models(results)
        
        if not validation_passed:
            logger.warning("⚠️ Model validation has concerns but continuing...")
        
        # Step 5: Start Enhanced API
        print("\n🌐 STEP 5: STARTING ENHANCED API SERVER")
        print("-" * 50)
        
        print("Starting enhanced prediction API on http://localhost:8080")
        print("API Documentation: http://localhost:8080/docs")
        
        # Calculate total time
        end_time = datetime.now()
        duration = end_time - start_time
        
        print("\n" + "=" * 80)
        print("🎉 ENHANCED PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"⏱️  Total Duration: {duration}")
        print(f"🏆 Best Model: {best_model_name}")
        print(f"📊 Performance: {results[best_model_name]['auc_score']:.4f} AUC")
        
        # Print next steps
        print_next_steps()
        
        # Start the API server
        start_api_server()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {str(e)}")
        return False

def validate_models(results):
    """Validate that models meet minimum performance requirements"""
    print("Validating model performance...")
    
    validation_passed = True
    min_auc_threshold = 0.75
    min_f1_threshold = 0.60
    
    for model_name, metrics in results.items():
        auc = metrics.get('auc_score', 0)
        f1 = metrics.get('f1_score', 0)
        
        print(f"   {model_name}:")
        print(f"     AUC: {auc:.4f} {'✅' if auc >= min_auc_threshold else '❌'}")
        print(f"     F1:  {f1:.4f} {'✅' if f1 >= min_f1_threshold else '❌'}")
        
        if auc < min_auc_threshold or f1 < min_f1_threshold:
            validation_passed = False
            print(f"     ⚠️ {model_name} below minimum thresholds")
    
    if validation_passed:
        print("✅ All models meet performance requirements")
    else:
        print("⚠️ Some models below performance thresholds")
        print("   Consider: more data, feature engineering, or hyperparameter tuning")
    
    return validation_passed

def print_next_steps():
    """Print next steps and usage examples"""
    print("\n📋 NEXT STEPS:")
    print("1. Test the enhanced API:")
    print("   curl -X POST http://localhost:8080/predict \\")
    print("     -H 'Content-Type: application/json' \\")
    print("     -d '{")
    print('       "cpu_usage": 85,')
    print('       "memory_usage": 90,')
    print('       "disk_usage": 45,')
    print('       "network_latency_ms": 120,')
    print('       "error_count": 5,')
    print('       "response_time_ms": 800,')
    print('       "active_connections": 150')
    print("     }'")
    
    print("\n2. View API documentation:")
    print("   http://localhost:8080/docs")
    
    print("\n3. Monitor model performance:")
    print("   Check logs and metrics in the API responses")
    
    print("\n4. Test different scenarios:")
    print("   - Low risk: cpu_usage=30, memory_usage=40, error_count=0")
    print("   - Medium risk: cpu_usage=75, memory_usage=80, error_count=3")
    print("   - High risk: cpu_usage=95, memory_usage=95, error_count=15")
    print("   - Critical: cpu_usage=98, memory_usage=98, error_count=25")

def start_api_server():
    """Start the enhanced API server"""
    try:
        import uvicorn
        from src.models.prediction import app
        
        print("\n🌐 Starting Enhanced API Server...")
        print("   Server: http://localhost:8080")
        print("   Docs: http://localhost:8080/docs")
        print("   Health: http://localhost:8080/health")
        print("\nPress Ctrl+C to stop the server")
        
        # Start the server
        uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
        
    except KeyboardInterrupt:
        print("\n⏹️  Server stopped by user")
    except Exception as e:
        logger.error(f"❌ Failed to start API server: {str(e)}")
        print("\n🔧 Manual start command:")
        print("   python -m uvicorn src.models.prediction:app --host 0.0.0.0 --port 8080")

def test_api_predictions():
    """Test the API with various scenarios"""
    import requests
    import json
    
    print("\n🧪 TESTING API PREDICTIONS")
    print("-" * 50)
    
    test_scenarios = [
        {
            "name": "Low Risk Scenario",
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
            "name": "Medium Risk Scenario", 
            "data": {
                "cpu_usage": 75,
                "memory_usage": 80,
                "disk_usage": 45,
                "network_latency_ms": 120,
                "error_count": 3,
                "response_time_ms": 600,
                "active_connections": 100
            }
        },
        {
            "name": "High Risk Scenario",
            "data": {
                "cpu_usage": 95,
                "memory_usage": 92,
                "disk_usage": 88,
                "network_latency_ms": 300,
                "error_count": 12,
                "response_time_ms": 1500,
                "active_connections": 250
            }
        },
        {
            "name": "Critical Risk Scenario",
            "data": {
                "cpu_usage": 98,
                "memory_usage": 97,
                "disk_usage": 96,
                "network_latency_ms": 800,
                "error_count": 25,
                "response_time_ms": 3000,
                "active_connections": 500
            }
        }
    ]
    
    for scenario in test_scenarios:
        try:
            response = requests.post(
                "http://localhost:8080/predict",
                json=scenario["data"],
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ {scenario['name']}:")
                print(f"   Risk Level: {result['failure_risk']}")
                print(f"   Probability: {result['failure_probability']:.4f}")
                print(f"   Health Score: {result['system_health_score']:.1f}")
                print(f"   Confidence: {result['confidence']:.3f}")
                print(f"   Key Recommendations: {len(result['recommendations'])}")
            else:
                print(f"❌ {scenario['name']}: HTTP {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ {scenario['name']}: Connection error - {str(e)}")
        except Exception as e:
            print(f"❌ {scenario['name']}: Error - {str(e)}")

def main():
    """Main function to run the complete pipeline"""
    
    # Check dependencies
    required_packages = ['pandas', 'numpy', 'scikit-learn', 'fastapi', 'uvicorn']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing required packages: {missing_packages}")
        print("Please install with: pip install -r requirements.txt")
        return False
    
    # Create necessary directories
    dirs_to_create = ["data/raw", "data/processed", "models", "logs"]
    for dir_path in dirs_to_create:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # Run the enhanced pipeline
    success = run_enhanced_pipeline()
    
    if success:
        print("\n🎊 Pipeline completed successfully!")
        
        # Optionally test the API
        test_choice = input("\nWould you like to test the API with sample predictions? (y/n): ")
        if test_choice.lower() == 'y':
            import time
            print("Waiting 3 seconds for server to fully start...")
            time.sleep(3)
            test_api_predictions()
    else:
        print("\n❌ Pipeline failed!")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)