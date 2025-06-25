# 🚀 Simple MLOps Error Prediction System

A complete, free, local MLOps pipeline for predicting system errors.

## 📁 Project Structure

```
mlops-error-prediction/
├── 📂 data/
│   ├── 📂 raw/                    # Raw log files
│   ├── 📂 processed/              # Processed datasets
│   └── 📂 sample/                 # Sample data for testing
├── 📂 src/
│   ├── 📂 data/
│   │   ├── 🐍 data_generator.py   # Generate synthetic log data
│   │   ├── 🐍 data_ingestion.py  # Ingest and parse logs
│   │   └── 🐍 preprocessing.py    # Clean and prepare data
│   ├── 📂 features/
│   │   ├── 🐍 feature_engineering.py # Create ML features
│   │   └── 🐍 feature_selection.py   # Select best features
│   ├── 📂 models/
│   │   ├── 🐍 training.py         # Train ML models
│   │   ├── 🐍 prediction.py       # Make predictions
│   │   └── 🐍 evaluation.py       # Evaluate model performance
│   ├── 📂 api/
│   │   ├── 🐍 app.py              # FastAPI application
│   │   └── 🐍 endpoints.py        # API endpoints
│   └── 📂 monitoring/
│       └── 🐍 model_monitor.py    # Monitor model performance
├── 📂 notebooks/
│   ├── 📓 01_data_exploration.ipynb
│   ├── 📓 02_feature_engineering.ipynb
│   └── 📓 03_model_development.ipynb
├── 📂 tests/
│   ├── 🐍 test_data_pipeline.py
│   ├── 🐍 test_models.py
│   └── 🐍 test_api.py
├── 📂 config/
│   ├── 📄 config.yaml             # Configuration settings
│   └── 📄 logging.yaml            # Logging configuration
├── 📂 docker/
│   ├── 📄 Dockerfile
│   └── 📄 docker-compose.yml
├── 📂 scripts/
│   ├── 🐍 setup.py               # Setup script
│   ├── 🐍 train_model.py         # Training script
│   └── 🐍 run_pipeline.py        # Full pipeline runner
├── 📂 models/                     # Saved models
├── 📂 logs/                       # Application logs
├── 📄 requirements.txt            # Python dependencies
├── 📄 README.md                   # Documentation
├── 📄 .env.example               # Environment variables example
└── 📄 Makefile                   # Automation commands
```

## 🛠️ Technology Stack (All Free & Open Source)

- **Python 3.8+** - Core language
- **scikit-learn** - Machine learning
- **pandas** - Data manipulation
- **FastAPI** - REST API
- **SQLite** - Local database
- **Jupyter** - Notebooks for exploration
- **Docker** - Containerization (optional)
- **Pytest** - Testing
- **MLflow** - Experiment tracking (local)

## 🚀 Quick Setup (5 Minutes)

### 1. Clone and Setup
```bash
git clone <your-repo>
cd mlops-error-prediction
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Generate Sample Data
```bash
python src/data/data_generator.py
```

### 3. Run Full Pipeline
```bash
python scripts/run_pipeline.py
```

### 4. Start API Server
```bash
uvicorn src.api.app:app --reload
```

### 5. Test Predictions
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"cpu_usage": 85, "memory_usage": 90, "error_count": 5}'
```

## 📊 Features Included

✅ **Data Generation** - Synthetic system logs and metrics  
✅ **Feature Engineering** - Time-based and statistical features  
✅ **Model Training** - Random Forest + XGBoost models  
✅ **Real-time Prediction** - REST API for predictions  
✅ **Model Monitoring** - Track model performance  
✅ **Experiment Tracking** - MLflow integration  
✅ **Testing** - Unit and integration tests  
✅ **Docker Support** - Containerized deployment  

## 🎯 What This System Predicts

- **System Failures** (High/Medium/Low risk)
- **Resource Bottlenecks** (CPU, Memory, Disk)
- **Application Errors** (Error probability)
- **Performance Degradation** (Response time issues)

## 📈 Sample Output

```json
{
  "prediction": {
    "failure_risk": "HIGH",
    "failure_probability": 0.87,
    "predicted_failure_time": "2024-01-15T14:30:00",
    "recommendations": [
      "Scale up CPU resources",
      "Clear log files",
      "Restart application service"
    ]
  }
}
```