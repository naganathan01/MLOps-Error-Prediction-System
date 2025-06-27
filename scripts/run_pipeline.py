"""
Complete pipeline runner for MLOps Error Prediction System.
File: scripts/run_pipeline.py
"""

import sys
import logging
from pathlib import Path

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.data_generator import EnhancedDataGenerator
from src.features.feature_engineering import FeatureEngineer
from src.models.training import ModelTrainer
from src.models.prediction import PredictionEngine, SystemMetrics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_complete_pipeline():
    """Run the complete MLOps pipeline"""
    logger.info("🚀 Starting Complete MLOps Error Prediction Pipeline")
    logger.info("=" * 60)
    
    try:
        # Step 1: Generate data
        logger.info("\n📊 STEP 1: GENERATING DATA")
        logger.info("-" * 30)
        generator = EnhancedDataGenerator()
        generator.generate_realistic_data(n_days=30)
        
        # Step 2: Feature engineering
        logger.info("\n🔧 STEP 2: FEATURE ENGINEERING")
        logger.info("-" * 30)
        engineer = FeatureEngineer()
        engineer.create_features()
        
        # Step 3: Train models
        logger.info("\n🤖 STEP 3: TRAINING MODELS")
        logger.info("-" * 30)
        trainer = ModelTrainer()
        results, best_model_name, best_model = trainer.train_all_models()
        
        # Step 4: Test prediction
        logger.info("\n🧪 STEP 4: TESTING PREDICTIONS")
        logger.info("-" * 30)
        
        # Test with sample data
        test_metrics = SystemMetrics(
            cpu_usage=85.0,
            memory_usage=90.0,
            disk_usage=45.0,
            network_latency_ms=120.0,
            error_count=5,
            response_time_ms=800.0,
            active_connections=150
        )
        
        engine = PredictionEngine()
        prediction = engine.predict(test_metrics)
        
        logger.info(f"✅ Test Prediction Results:")
        logger.info(f"   Failure Probability: {prediction.failure_probability:.4f}")
        logger.info(f"   Risk Level: {prediction.failure_risk}")
        logger.info(f"   Confidence: {prediction.confidence:.3f}")
        logger.info(f"   Model Used: {prediction.model_used}")
        logger.info(f"   Recommendations: {len(prediction.recommendations)}")
        for i, rec in enumerate(prediction.recommendations, 1):
            logger.info(f"     {i}. {rec}")
        
        logger.info("\n" + "=" * 60)
        logger.info("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info(f"🏆 Best Model: {best_model_name}")
        logger.info(f"📊 Test Prediction: {prediction.failure_probability:.4f} ({prediction.failure_risk})")
        
        logger.info("\n📋 NEXT STEPS:")
        logger.info("1. Start the API server:")
        logger.info("   python main.py")
        logger.info("\n2. Test the API:")
        logger.info("   curl -X POST http://localhost:8000/predict \\")
        logger.info("     -H 'Content-Type: application/json' \\")
        logger.info("     -d '{")
        logger.info('       "cpu_usage": 85,')
        logger.info('       "memory_usage": 90,')
        logger.info('       "disk_usage": 45,')
        logger.info('       "network_latency_ms": 120,')
        logger.info('       "error_count": 5,')
        logger.info('       "response_time_ms": 800,')
        logger.info('       "active_connections": 150')
        logger.info("     }'")
        
        logger.info("\n3. View API documentation:")
        logger.info("   http://localhost:8000/docs")
        
        return True
        
    except Exception as e:
        logger.error(f"\n❌ Pipeline failed: {str(e)}")
        return False

def main():
    """Main function"""
    success = run_complete_pipeline()
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
