"""
Setup script for MLOps Error Prediction project.
Creates necessary directories and initializes the project structure.
"""

import os
import sys
from pathlib import Path
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProjectSetup:
    def __init__(self, project_root="."):
        self.project_root = Path(project_root)
        self.directories = [
            "data/raw",
            "data/processed", 
            "data/sample",
            "models",
            "logs",
            "mlruns",
            "notebooks",
            "tests",
            "scripts"
        ]
        
    def create_directories(self):
        """Create project directories"""
        logger.info("🏗️ Creating project directories...")
        
        for directory in self.directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"   Created: {dir_path}")
            
        logger.info("✅ Project directories created")
    
    def create_gitkeep_files(self):
        """Create .gitkeep files in empty directories"""
        logger.info("📁 Creating .gitkeep files...")
        
        empty_dirs = ["data/raw", "data/processed", "data/sample", "models", "logs"]
        
        for directory in empty_dirs:
            gitkeep_path = self.project_root / directory / ".gitkeep"
            gitkeep_path.touch()
            logger.info(f"   Created: {gitkeep_path}")
        
        logger.info("✅ .gitkeep files created")
    
    def create_init_files(self):
        """Create __init__.py files for Python packages"""
        logger.info("🐍 Creating __init__.py files...")
        
        python_dirs = [
            "src",
            "src/api",
            "src/data", 
            "src/features",
            "src/models",
            "src/monitoring",
            "tests"
        ]
        
        for directory in python_dirs:
            init_path = self.project_root / directory / "__init__.py"
            if not init_path.exists():
                init_path.touch()
                logger.info(f"   Created: {init_path}")
        
        logger.info("✅ __init__.py files created")
    
    def create_sample_config(self):
        """Create sample configuration files"""
        logger.info("⚙️ Creating sample configuration...")
        
        # Create sample data configuration
        sample_config = {
            "data_generation": {
                "n_days": 7,
                "samples_per_hour": 12,
                "logs_per_hour": 100
            },
            "feature_engineering": {
                "rolling_windows": [5, 15, 30],
                "lag_periods": [1, 2, 3, 5]
            },
            "model_training": {
                "test_size": 0.2,
                "random_state": 42,
                "cv_folds": 3
            }
        }
        
        config_path = self.project_root / "data" / "sample" / "sample_config.json"
        with open(config_path, 'w') as f:
            json.dump(sample_config, f, indent=2)
        
        logger.info(f"   Created: {config_path}")
        logger.info("✅ Sample configuration created")
    
    def create_readme_sections(self):
        """Create additional README sections"""
        logger.info("📖 Creating documentation...")
        
        # Create API documentation
        api_docs = """# API Documentation

## Endpoints

### POST /predict
Predict system failure probability

**Request Body:**
```json
{
  "cpu_usage": 85.0,
  "memory_usage": 90.0,
  "disk_usage": 45.0,
  "network_latency_ms": 120.0,
  "error_count": 3,
  "response_time_ms": 450.0,
  "active_connections": 75
}
```

**Response:**
```json
{
  "failure_probability": 0.8542,
  "failure_risk": "HIGH",
  "predicted_failure_time": "2024-01-15T14:30:00",
  "recommendations": [
    "🚨 URGENT: High failure risk detected",
    "⚡ Scale up CPU resources or optimize CPU-intensive processes"
  ],
  "model_used": "xgboost",
  "confidence": 0.875,
  "timestamp": "2024-01-15T14:15:00"
}
```

### GET /health
Health check endpoint

### GET /models/info
Get information about loaded models
"""
        
        docs_path = self.project_root / "docs" / "api.md"
        docs_path.parent.mkdir(exist_ok=True)
        with open(docs_path, 'w') as f:
            f.write(api_docs)
        
        logger.info(f"   Created: {docs_path}")
        logger.info("✅ Documentation created")
    
    def validate_requirements(self):
        """Validate that required files exist"""
        logger.info("🔍 Validating project requirements...")
        
        required_files = [
            "requirements.txt",
            "src/api/app.py",
            "src/data/data_generator.py",
            "src/features/feature_engineering.py",
            "src/models/training.py"
        ]
        
        missing_files = []
        for file_path in required_files:
            if not (self.project_root / file_path).exists():
                missing_files.append(file_path)
        
        if missing_files:
            logger.error(f"❌ Missing required files: {missing_files}")
            return False
        
        logger.info("✅ All required files present")
        return True
    
    def create_test_data(self):
        """Create minimal test data for testing"""
        logger.info("🧪 Creating test data...")
        
        test_data = {
            "system_metrics": [
                {
                    "timestamp": "2024-01-15T14:00:00",
                    "cpu_usage": 85.0,
                    "memory_usage": 90.0,
                    "disk_usage": 45.0,
                    "network_latency_ms": 120.0,
                    "error_count": 3,
                    "response_time_ms": 450.0,
                    "active_connections": 75,
                    "failure_within_hour": 1
                },
                {
                    "timestamp": "2024-01-15T14:05:00",
                    "cpu_usage": 45.0,
                    "memory_usage": 50.0,
                    "disk_usage": 35.0,
                    "network_latency_ms": 80.0,
                    "error_count": 0,
                    "response_time_ms": 200.0,
                    "active_connections": 45,
                    "failure_within_hour": 0
                }
            ]
        }
        
        test_path = self.project_root / "data" / "sample" / "test_data.json"
        with open(test_path, 'w') as f:
            json.dump(test_data, f, indent=2)
        
        logger.info(f"   Created: {test_path}")
        logger.info("✅ Test data created")
    
    def setup_project(self):
        """Run complete project setup"""
        logger.info("🚀 Starting MLOps Error Prediction project setup...")
        
        try:
            # Create directories
            self.create_directories()
            
            # Create .gitkeep files
            self.create_gitkeep_files()
            
            # Create __init__.py files
            self.create_init_files()
            
            # Create sample configuration
            self.create_sample_config()
            
            # Create documentation
            self.create_readme_sections()
            
            # Create test data
            self.create_test_data()
            
            # Validate requirements
            if not self.validate_requirements():
                logger.error("❌ Project setup validation failed")
                return False
            
            logger.info("🎉 Project setup completed successfully!")
            
            # Print next steps
            self.print_next_steps()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Project setup failed: {str(e)}")
            return False
    
    def print_next_steps(self):
        """Print next steps for the user"""
        print("\n" + "="*60)
        print("🎉 MLOps Error Prediction Project Setup Complete!")
        print("="*60)
        print("\n📋 Next Steps:")
        print("1. Install dependencies:")
        print("   pip install -r requirements.txt")
        print("\n2. Generate sample data:")  
        print("   python src/data/data_generator.py")
        print("\n3. Create features:")
        print("   python src/features/feature_engineering.py")
        print("\n4. Train models:")
        print("   python src/models/training.py")
        print("\n5. Start API server:")
        print("   uvicorn src.api.app:app --reload")
        print("\n6. Test the API:")
        print("   curl -X POST http://localhost:8000/predict \\")
        print("     -H 'Content-Type: application/json' \\")
        print("     -d '{\"cpu_usage\": 85, \"memory_usage\": 90, \"disk_usage\": 45, \"network_latency_ms\": 120, \"error_count\": 3, \"response_time_ms\": 450, \"active_connections\": 75}'")
        print("\n📖 Documentation:")
        print("   - API docs: http://localhost:8000/docs")
        print("   - Project README: README.md")
        print("   - Configuration: config/config.yaml")
        print("\n🔧 Use Makefile for easy commands:")
        print("   make help  # See all available commands")
        print("   make all   # Run complete pipeline")
        print("\n" + "="*60)

def main():
    """Main function"""
    if len(sys.argv) > 1:
        project_root = sys.argv[1]
    else:
        project_root = "."
    
    setup = ProjectSetup(project_root)
    success = setup.setup_project()
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()