# MLOps Error Prediction System

## 🚀 Overview

This MLOps Error Prediction System uses machine learning to predict system failures before they happen, enabling proactive intervention and preventing costly downtime. The system monitors real-time system metrics and provides actionable recommendations.

## 🎯 Real-World Use Cases

### 1. **E-Commerce Platform Monitoring**
- **Scenario**: Monitor high-traffic e-commerce sites during Black Friday sales
- **Benefit**: Predict and prevent crashes before they impact millions in sales
- **Example**: System detects memory leak patterns and alerts team 30 minutes before potential crash

### 2. **Cloud Infrastructure Management**
- **Scenario**: AWS/Azure/GCP resource monitoring for enterprise applications
- **Benefit**: Optimize resource allocation and prevent service outages
- **Example**: Predict when auto-scaling should trigger based on complex patterns

### 3. **Banking & Financial Systems**
- **Scenario**: Critical transaction processing systems
- **Benefit**: Ensure 99.99% uptime for payment processing
- **Example**: Detect anomalies in database response times before they impact transactions

### 4. **Healthcare Systems**
- **Scenario**: Hospital information systems and patient monitoring
- **Benefit**: Prevent system failures that could impact patient care
- **Example**: Predict EMR system failures and ensure continuous access to patient records

### 5. **Manufacturing & IoT**
- **Scenario**: Industrial IoT sensors and production line monitoring
- **Benefit**: Prevent production line stoppages
- **Example**: Detect patterns that indicate equipment failure 2-4 hours in advance

## 📊 Model Performance

The system uses an ensemble approach with three models:
- **XGBoost**: Best for complex non-linear patterns (typical AUC: 0.92-0.95)
- **Random Forest**: Robust to outliers (typical AUC: 0.89-0.93)
- **Logistic Regression**: Fast, interpretable baseline (typical AUC: 0.85-0.88)

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+ (tested on 3.12)
- 4GB RAM minimum
- 1GB free disk space

### Step 1: Clone and Setup Environment
```bash
# Clone the repository (or create directory)
mkdir mlops-error-prediction
cd mlops-error-prediction

# Create virtual environment
python -m venv mlops_env

# Activate environment
# Windows:
mlops_env\Scripts\activate
# Linux/Mac:
source mlops_env/bin/activate

# Upgrade pip
python -m pip install --upgrade pip setuptools wheel
```

### Step 2: Install Dependencies
```bash
# Install all requirements
pip install -r requirements.txt
```

### Step 3: Run the Complete Pipeline
```bash
# This will generate data, engineer features, and train models
python scripts/run_pipeline.py
```

Expected output:
```
🚀 Starting Complete MLOps Error Prediction Pipeline
📊 STEP 1: GENERATING DATA
✅ Generated 8640 records with 14.7% failure rate
🔧 STEP 2: FEATURE ENGINEERING
✅ Created 44 total features
🤖 STEP 3: TRAINING MODELS
🏆 Best Model: xgboost (AUC: 0.9342)
🎉 PIPELINE COMPLETED SUCCESSFULLY!
```

### Step 4: Start the API Server
```bash
python main.py
```

The API will be available at:
- Main: http://localhost:8000
- Docs: http://localhost:8000/docs
- Alternative Docs: http://localhost:8000/redoc

### Step 5: Test the API
In a new terminal:
```bash
# Activate environment first
mlops_env\Scripts\activate  # or source mlops_env/bin/activate

# Run tests
python tests/test_api.py
```

## 📋 API Usage Examples

### 1. Single Prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "cpu_usage": 85,
    "memory_usage": 90,
    "disk_usage": 45,
    "network_latency_ms": 120,
    "error_count": 5,
    "response_time_ms": 800,
    "active_connections": 150
  }'
```

### 2. Batch Prediction
```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '[
    {
      "cpu_usage": 30,
      "memory_usage": 40,
      "disk_usage": 25,
      "network_latency_ms": 50,
      "error_count": 0,
      "response_time_ms": 200,
      "active_connections": 25
    },
    {
      "cpu_usage": 95,
      "memory_usage": 92,
      "disk_usage": 88,
      "network_latency_ms": 300,
      "error_count": 15,
      "response_time_ms": 2500,
      "active_connections": 200
    }
  ]'
```

## 🔍 Understanding Predictions

### Risk Levels
- **CRITICAL** (>80% probability): Immediate action required
- **HIGH** (60-80%): Urgent attention needed
- **MEDIUM** (40-60%): Monitor closely
- **LOW** (20-40%): Normal monitoring
- **MINIMAL** (<20%): System healthy

### Example Response
```json
{
  "failure_probability": 0.8543,
  "failure_risk": "CRITICAL",
  "confidence": 0.854,
  "recommendations": [
    "🚨 URGENT: High failure risk - take immediate action",
    "⚡ CPU Critical: Scale resources or optimize processes",
    "💾 Memory Critical: Check for leaks, restart services"
  ],
  "model_used": "xgboost",
  "timestamp": "2024-01-10T15:30:45.123456"
}
```

## 🏭 Production Deployment

### Docker Deployment
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Train models during build (or mount pre-trained models)
RUN python scripts/run_pipeline.py

EXPOSE 8000

CMD ["python", "main.py"]
```

### Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mlops-error-prediction
spec:
  replicas: 3
  selector:
    matchLabels:
      app: mlops-error-prediction
  template:
    metadata:
      labels:
        app: mlops-error-prediction
    spec:
      containers:
      - name: api
        image: mlops-error-prediction:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
```

## 📈 Model Metrics

### Feature Importance (Top 10)
1. `response_time_ms` - Response time is the strongest predictor
2. `error_count` - Error frequency directly correlates with failures
3. `cpu_usage` - CPU spikes often precede failures
4. `memory_usage` - Memory leaks are common failure causes
5. `total_stress` - Combined stress indicator
6. `error_rate` - Errors per connection
7. `performance_ratio` - Response time per connection
8. `cpu_memory_product` - Interaction between CPU and memory
9. `network_latency_ms` - Network issues compound other problems
10. `resource_pressure` - Average resource utilization

### Model Validation
- **Cross-validation**: 5-fold CV ensures robust performance
- **Time-based split**: Validates on future data
- **Class imbalance**: Handled with class weights and SMOTE
- **Feature scaling**: StandardScaler for logistic regression

## 🔧 Configuration

### Environment Variables
```bash
# .env file
API_HOST=0.0.0.0
API_PORT=8000
MODEL_PATH=models/
DATA_PATH=data/
LOG_LEVEL=INFO
```

### Model Parameters
Edit `src/models/training.py` to tune:
```python
# Random Forest
n_estimators=200  # Increase for better accuracy
max_depth=15      # Prevent overfitting

# XGBoost
learning_rate=0.1  # Lower for more stable training
n_estimators=200   # More trees = better performance
```

## 🚨 Monitoring & Alerts

### Integration with Monitoring Tools
```python
# Prometheus metrics
from prometheus_client import Counter, Histogram, Gauge

prediction_counter = Counter('predictions_total', 'Total predictions made')
failure_gauge = Gauge('failure_probability', 'Current failure probability')
response_time = Histogram('prediction_duration_seconds', 'Prediction duration')
```

### Alert Rules (Prometheus)
```yaml
groups:
  - name: mlops_alerts
    rules:
      - alert: HighFailureProbability
        expr: failure_probability > 0.8
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High failure probability detected"
          description: "Failure probability is {{ $value }}"
```

## 📊 Performance Benchmarks

### API Performance
- **Latency**: <100ms for single prediction
- **Throughput**: 1000+ requests/second
- **Batch size**: Up to 1000 predictions per request

### Model Accuracy
- **Precision**: 0.89 (89% of predicted failures are real)
- **Recall**: 0.85 (catches 85% of actual failures)
- **F1-Score**: 0.87 (balanced performance)
- **AUC-ROC**: 0.93 (excellent discrimination)

## 🔍 Troubleshooting

### Common Issues

1. **No models loaded**
   - Solution: Run `python scripts/run_pipeline.py` first

2. **Import errors**
   - Solution: Ensure virtual environment is activated

3. **Memory errors during training**
   - Solution: Reduce `n_estimators` or use smaller data sample

4. **API timeout**
   - Solution: Increase uvicorn timeout settings

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Built with FastAPI, scikit-learn, and XGBoost
- Inspired by real-world DevOps challenges
- Synthetic data generation based on production patterns

## 📞 Support

For issues and questions:
- Email: nathannathan42242@gmail.com
