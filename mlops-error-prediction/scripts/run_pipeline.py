"""
Complete MLOps pipeline runner.
Executes the full data generation, feature engineering, and model training pipeline.
"""

import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime
import yaml

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.data_generator import SystemDataGenerator
from src.features.feature_engineering import FeatureEngineer
from src.models.training import ModelTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MLOpsPipeline:
    def __init__(self, config_path="config/config.yaml"):
        self.config_path = Path(config_path)
        self.config = self.load_config()
        self.start_time = datetime.now()
        
    def load_config(self):
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"✅ Loaded configuration from {self.config_path}")
            return config
        except FileNotFoundError:
            logger.warning(f"⚠️ Config file not found: {self.config_path}, using defaults")
            return self.get_default_config()
        except Exception as e:
            logger.error(f"❌ Failed to load config: {str(e)}")
            return self.get_default_config()
    
    def get_default_config(self):
        """Get default configuration"""
        return {
            'data': {
                'generation': {
                    'n_days': 30,
                    'samples_per_hour': 12,
                    'logs_per_hour': 100
                }
            },
            'features': {
                'rolling_windows': [5, 15, 30],
                'lag_periods': [1, 2, 3, 5]
            },
            'training': {
                'test_size': 0.2,
                'random_state': 42
            }
        }
    
    def run_data_generation(self, n_days=None):
        """Run data generation step"""
        logger.info("🚀 Starting data generation...")
        
        try:
            # Get parameters from config
            if n_days is None:
                n_days = self.config.get('data', {}).get('generation', {}).get('n_days', 30)
            
            generator = SystemDataGenerator()
            metrics_df, logs_df = generator.generate_all_data(n_days=n_days)
            
            logger.info(f"✅ Data generation completed: {len(metrics_df)} metrics, {len(logs_df)} logs")
            return True
            
        except Exception as e:
            logger.error(f"❌ Data generation failed: {str(e)}")
            return False
    
    def run_feature_engineering(self):
        """Run feature engineering step"""
        logger.info("🔧 Starting feature engineering...")
        
        try:
            engineer = FeatureEngineer()
            df = engineer.process_features()
            
            logger.info(f"✅ Feature engineering completed: {len(df)} samples, {len(df.columns)} features")
            return True
            
        except Exception as e:
            logger.error(f"❌ Feature engineering failed: {str(e)}")
            return False
    
    def run_model_training(self):
        """Run model training step"""
        logger.info("🤖 Starting model training...")
        
        try:
            trainer = ModelTrainer()
            results, best_model_name, best_model = trainer.train_all_models()
            
            logger.info(f"✅ Model training completed: Best model is {best_model_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Model training failed: {str(e)}")
            return False
    
    def validate_prerequisites(self):
        """Validate that all prerequisites are met"""
        logger.info("🔍 Validating prerequisites...")
        
        # Check required directories
        required_dirs = ["data", "models", "logs"]
        for dir_name in required_dirs:
            dir_path = Path(dir_name)
            if not dir_path.exists():
                logger.error(f"❌ Required directory missing: {dir_path}")
                return False
            
        # Check required files
        required_files = [
            "src/data/data_generator.py",
            "src/features/feature_engineering.py", 
            "src/models/training.py"
        ]
        
        for file_path in required_files:
            if not Path(file_path).exists():
                logger.error(f"❌ Required file missing: {file_path}")
                return False
        
        logger.info("✅ Prerequisites validated")
        return True
    
    def run_full_pipeline(self, n_days=None, skip_data=False, skip_features=False, skip_training=False):
        """Run the complete MLOps pipeline"""
        logger.info("🚀 Starting MLOps Error Prediction Pipeline...")
        logger.info(f"⏰ Pipeline started at: {self.start_time}")
        
        # Validate prerequisites
        if not self.validate_prerequisites():
            logger.error("❌ Prerequisites validation failed")
            return False
        
        success_steps = []
        failed_steps = []
        
        try:
            # Step 1: Data Generation
            if not skip_data:
                logger.info("\n" + "="*60)
                logger.info("📊 STEP 1: DATA GENERATION")
                logger.info("="*60)
                
                if self.run_data_generation(n_days):
                    success_steps.append("Data Generation")
                else:
                    failed_steps.append("Data Generation")
                    if not self.check_existing_data():
                        logger.error("❌ No existing data found, cannot continue")
                        return False
            else:
                logger.info("⏭️ Skipping data generation")
                
            # Step 2: Feature Engineering  
            if not skip_features:
                logger.info("\n" + "="*60)
                logger.info("🔧 STEP 2: FEATURE ENGINEERING")
                logger.info("="*60)
                
                if self.run_feature_engineering():
                    success_steps.append("Feature Engineering")
                else:
                    failed_steps.append("Feature Engineering")
                    if not self.check_existing_features():
                        logger.error("❌ No existing features found, cannot continue")
                        return False
            else:
                logger.info("⏭️ Skipping feature engineering")
                
            # Step 3: Model Training
            if not skip_training:
                logger.info("\n" + "="*60)
                logger.info("🤖 STEP 3: MODEL TRAINING")
                logger.info("="*60)
                
                if self.run_model_training():
                    success_steps.append("Model Training")
                else:
                    failed_steps.append("Model Training")
            else:
                logger.info("⏭️ Skipping model training")
            
            # Pipeline Summary
            end_time = datetime.now()
            duration = end_time - self.start_time
            
            logger.info("\n" + "="*60)
            logger.info("📋 PIPELINE SUMMARY")
            logger.info("="*60)
            logger.info(f"⏰ Started: {self.start_time}")
            logger.info(f"⏰ Finished: {end_time}")
            logger.info(f"⏱️ Duration: {duration}")
            logger.info(f"✅ Successful steps: {', '.join(success_steps) if success_steps else 'None'}")
            logger.info(f"❌ Failed steps: {', '.join(failed_steps) if failed_steps else 'None'}")
            
            if failed_steps:
                logger.warning(f"⚠️ Pipeline completed with {len(failed_steps)} failed steps")
                return False
            else:
                logger.info("🎉 Pipeline completed successfully!")
                self.print_next_steps()
                return True
                
        except KeyboardInterrupt:
            logger.info("\n⏹️ Pipeline interrupted by user")
            return False
        except Exception as e:
            logger.error(f"❌ Pipeline failed with unexpected error: {str(e)}")
            return False
    
    def check_existing_data(self):
        """Check if raw data exists"""
        data_files = [
            Path("data/raw/system_metrics.csv"),
            Path("data/raw/application_logs.csv")
        ]
        return all(f.exists() for f in data_files)
    
    def check_existing_features(self):
        """Check if processed features exist"""
        return Path("data/processed/features.csv").exists()
    
    def print_next_steps(self):
        """Print next steps after successful pipeline completion"""
        print("\n" + "="*60)
        print("🎉 MLOps Pipeline Completed Successfully!")
        print("="*60)
        print("\n📋 Next Steps:")
        print("1. Start the API server:")
        print("   uvicorn src.api.app:app --reload")
        print("\n2. Test the API:")
        print("   curl -X POST http://localhost:8000/predict \\")
        print("     -H 'Content-Type: application/json' \\")
        print("     -d '{\"cpu_usage\": 85, \"memory_usage\": 90, \"error_count\": 3}'")
        print("\n3. View API documentation:")
        print("   http://localhost:8000/docs")
        print("\n4. Check MLflow experiments:")
        print("   mlflow ui")
        print("\n5. Run tests:")
        print("   pytest tests/")
        print("\n🔧 Available commands:")
        print("   make api    # Start API server")
        print("   make test   # Run tests")
        print("   make clean  # Clean generated files")
        print("="*60)

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description="MLOps Error Prediction Pipeline")
    
    parser.add_argument("--config", default="config/config.yaml", 
                       help="Path to configuration file")
    parser.add_argument("--days", type=int, default=None,
                       help="Number of days of data to generate")
    parser.add_argument("--skip-data", action="store_true",
                       help="Skip data generation step")
    parser.add_argument("--skip-features", action="store_true", 
                       help="Skip feature engineering step")
    parser.add_argument("--skip-training", action="store_true",
                       help="Skip model training step")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize and run pipeline
    pipeline = MLOpsPipeline(config_path=args.config)
    
    success = pipeline.run_full_pipeline(
        n_days=args.days,
        skip_data=args.skip_data,
        skip_features=args.skip_features,
        skip_training=args.skip_training
    )
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()