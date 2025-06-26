"""
Standalone model training script.
Can be used to train models independently or retrain existing models.
"""

import sys
import logging
import argparse
from pathlib import Path
import yaml
import json
from datetime import datetime

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.training import ModelTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TrainingScript:
    def __init__(self, config_path="config/config.yaml"):
        self.config_path = Path(config_path)
        self.config = self.load_config()
        
    def load_config(self):
        """Load configuration"""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file not found: {self.config_path}")
            return {}
        except Exception as e:
            logger.error(f"Failed to load config: {str(e)}")
            return {}
    
    def validate_data(self):
        """Validate that required data exists"""
        logger.info("🔍 Validating training data...")
        
        required_files = [
            "data/processed/features.csv"
        ]
        
        missing_files = []
        for file_path in required_files:
            if not Path(file_path).exists():
                missing_files.append(file_path)
        
        if missing_files:
            logger.error(f"❌ Missing required files: {missing_files}")
            logger.info("💡 Run feature engineering first:")
            logger.info("   python src/features/feature_engineering.py")
            return False
        
        logger.info("✅ Training data validated")
        return True
    
    def train_single_model(self, model_type):
        """Train a single model type"""
        logger.info(f"🤖 Training {model_type} model...")
        
        try:
            trainer = ModelTrainer()
            trainer.load_data()
            trainer.prepare_data()
            
            if model_type.lower() == 'random_forest':
                model, score = trainer.train_random_forest()
            elif model_type.lower() == 'xgboost':
                model, score = trainer.train_xgboost()
            elif model_type.lower() == 'logistic_regression':
                model, score = trainer.train_logistic_regression()
            else:
                logger.error(f"❌ Unknown model type: {model_type}")
                return False
            
            # Save the single model
            trainer.models = {model_type.lower(): model}
            trainer.save_models()
            
            logger.info(f"✅ {model_type} training completed with AUC: {score:.4f}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Training failed: {str(e)}")
            return False
    
    def train_all_models(self):
        """Train all available models"""
        logger.info("🤖 Training all models...")
        
        try:
            trainer = ModelTrainer()
            results, best_model_name, best_model = trainer.train_all_models()
            
            # Print results summary
            logger.info("\n📊 Training Results Summary:")
            logger.info("-" * 50)
            
            for model_name, metrics in results.items():
                logger.info(f"{model_name.upper()}:")
                logger.info(f"  AUC Score: {metrics['auc_score']:.4f}")
                logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
                logger.info(f"  Precision: {metrics['precision']:.4f}")
                logger.info(f"  Recall:    {metrics['recall']:.4f}")
                logger.info(f"  F1-Score:  {metrics['f1_score']:.4f}")
                logger.info("")
            
            logger.info(f"🏆 Best Model: {best_model_name}")
            logger.info(f"🏆 Best AUC Score: {results[best_model_name]['auc_score']:.4f}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Training failed: {str(e)}")
            return False
    
    def retrain_models(self, model_types=None):
        """Retrain existing models with new data"""
        logger.info("🔄 Retraining models...")
        
        if model_types is None:
            model_types = ['random_forest', 'xgboost', 'logistic_regression']
        
        # Check if models directory exists and backup existing models
        models_dir = Path("models")
        if models_dir.exists():
            backup_dir = models_dir / f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            backup_dir.mkdir(exist_ok=True)
            
            # Backup existing models
            for model_file in models_dir.glob("*.joblib"):
                if model_file.is_file():
                    backup_file = backup_dir / model_file.name
                    model_file.rename(backup_file)
                    logger.info(f"📦 Backed up {model_file.name} to {backup_file}")
        
        # Train new models
        success = self.train_all_models()
        
        if success:
            logger.info("✅ Model retraining completed successfully")
        else:
            logger.error("❌ Model retraining failed")
        
        return success
    
    def evaluate_existing_models(self):
        """Evaluate existing trained models"""
        logger.info("📊 Evaluating existing models...")
        
        models_dir = Path("models")
        if not models_dir.exists():
            logger.error("❌ No models directory found")
            return False
        
        # Check for model files
        model_files = list(models_dir.glob("*_model.joblib"))
        if not model_files:
            logger.error("❌ No trained models found")
            return False
        
        try:
            trainer = ModelTrainer()
            trainer.load_data()
            trainer.prepare_data()
            
            # Load existing models
            import joblib
            for model_file in model_files:
                model_name = model_file.stem.replace('_model', '')
                trainer.models[model_name] = joblib.load(model_file)
                logger.info(f"📥 Loaded {model_name} model")
            
            # Evaluate models
            results, best_model_name, best_model = trainer.evaluate_models()
            
            logger.info("✅ Model evaluation completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Model evaluation failed: {str(e)}")
            return False
    
    def get_model_info(self):
        """Get information about trained models"""
        logger.info("ℹ️ Getting model information...")
        
        models_dir = Path("models")
        if not models_dir.exists():
            logger.warning("⚠️ No models directory found")
            return
        
        # Check metadata file
        metadata_file = models_dir / "training_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            logger.info("📋 Model Training Metadata:")
            logger.info(f"  Training Date: {metadata.get('training_date', 'Unknown')}")
            logger.info(f"  Total Samples: {metadata.get('total_samples', 'Unknown')}")
            logger.info(f"  Training Samples: {metadata.get('training_samples', 'Unknown')}")
            logger.info(f"  Testing Samples: {metadata.get('testing_samples', 'Unknown')}")
            logger.info(f"  Feature Count: {metadata.get('feature_count', 'Unknown')}")
        
        # List available models
        model_files = list(models_dir.glob("*_model.joblib"))
        if model_files:
            logger.info("🤖 Available Models:")
            for model_file in model_files:
                model_name = model_file.stem.replace('_model', '')
                file_size = model_file.stat().st_size / (1024 * 1024)  # MB
                logger.info(f"  - {model_name}: {file_size:.2f} MB")
        else:
            logger.warning("⚠️ No trained models found")

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description="MLOps Model Training Script")
    
    parser.add_argument("--config", default="config/config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--model", choices=['random_forest', 'xgboost', 'logistic_regression'],
                       help="Train only specific model type")
    parser.add_argument("--retrain", action="store_true",
                       help="Retrain existing models")
    parser.add_argument("--evaluate", action="store_true",
                       help="Evaluate existing models")
    parser.add_argument("--info", action="store_true",
                       help="Show model information")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize training script
    training_script = TrainingScript(config_path=args.config)
    
    # Validate data unless just showing info
    if not args.info and not training_script.validate_data():
        sys.exit(1)
    
    success = True
    
    try:
        if args.info:
            training_script.get_model_info()
        elif args.evaluate:
            success = training_script.evaluate_existing_models()
        elif args.retrain:
            success = training_script.retrain_models()
        elif args.model:
            success = training_script.train_single_model(args.model)
        else:
            success = training_script.train_all_models()
    
    except KeyboardInterrupt:
        logger.info("\n⏹️ Training interrupted by user")
        success = False
    except Exception as e:
        logger.error(f"❌ Training script failed: {str(e)}")
        success = False
    
    if not success:
        sys.exit(1)
    
    logger.info("🎉 Training script completed successfully!")

if __name__ == "__main__":
    main()