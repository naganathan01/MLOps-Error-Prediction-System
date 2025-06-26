"""
FastAPI application for error prediction service.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MLOps Error Prediction API",
    description="Predict system failures before they happen",
    version="1.0.0"
)

# Global variables for models and configurations
models = {}
scaler = None
feature_columns = []
model_metadata = {}

class SystemMetrics(BaseModel):
    """Input schema for system metrics"""
    cpu_usage: float = Field(..., ge=0, le=100, description="CPU usage percentage")
    memory_usage: float = Field(..., ge=0, le=100, description="Memory usage percentage")
    disk_usage: float = Field(..., ge=0, le=100, description="Disk usage percentage")
    network_latency_ms: float = Field(..., ge=0, description="Network latency in milliseconds")
    error_count: int = Field(..., ge=0, description="Number of errors in the last period")
    response_time_ms: float = Field(..., ge=0, description="Average response time in milliseconds")
    active_connections: int = Field(..., ge=0, description="Number of active connections")
    
    # Optional time-based features
    hour: Optional[int] = Field(default=None, ge=0, le=23, description="Hour of the day")
    day_of_week: Optional[int] = Field(default=None, ge=0, le=6, description="Day of the week (0=Monday)")

class PredictionResponse(BaseModel):
    """Response schema for predictions"""
    failure_probability: float = Field(..., description="Probability of failure (0-1)")
    failure_risk: str = Field(..., description="Risk level: LOW, MEDIUM, HIGH")
    predicted_failure_time: Optional[str] = Field(None, description="Estimated time of potential failure")
    recommendations: List[str] = Field(..., description="Recommended actions")
    model_used: str = Field(..., description="Model used for prediction")
    confidence: float = Field(..., description="Prediction confidence")
    timestamp: str = Field(..., description="Prediction timestamp")

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    models_loaded: List[str]
    uptime: str

@app.on_event("startup")
async def load_models():
    """Load trained models on startup"""
    try:
        model_dir = Path("models")
        
        if not model_dir.exists():
            logger.error(f"Models directory not found: {model_dir}")
            raise FileNotFoundError(f"Models directory not found: {model_dir}")
        
        # Load models
        global models, scaler, feature_columns, model_metadata
        
        model_files = {
            'random_forest': model_dir / "random_forest_model.joblib",
            'xgboost': model_dir / "xgboost_model.joblib",
            'logistic_regression': model_dir / "logistic_regression_model.joblib"
        }
        
        for name, file_path in model_files.items():
            if file_path.exists():
                models[name] = joblib.load(file_path)
                logger.info(f"Loaded {name} model")
            else:
                logger.warning(f"Model file not found: {file_path}")
        
        # Load scaler
        scaler_file = model_dir / "scaler.joblib"
        if scaler_file.exists():
            scaler = joblib.load(scaler_file)
            logger.info("Loaded scaler")
        
        # Load feature columns
        features_file = model_dir / "feature_columns.json"
        if features_file.exists():
            with open(features_file, 'r') as f:
                feature_columns = json.load(f)
            logger.info(f"Loaded {len(feature_columns)} feature columns")
        
        # Load metadata
        metadata_file = model_dir / "training_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                model_metadata = json.load(f)
            logger.info("Loaded model metadata")
        
        if not models:
            raise ValueError("No models were loaded successfully")
        
        logger.info(f"Successfully loaded {len(models)} models")
        
    except Exception as e:
        logger.error(f"Failed to load models: {str(e)}")
        raise

def create_features_from_metrics(metrics: SystemMetrics) -> pd.DataFrame:
    """Create feature vector from input metrics"""
    
    # Get current time info if not provided
    now = datetime.now()
    hour = metrics.hour if metrics.hour is not None else now.hour
    day_of_week = metrics.day_of_week if metrics.day_of_week is not None else now.weekday()
    
    # Create basic features
    features = {
        'cpu_usage': metrics.cpu_usage,
        'memory_usage': metrics.memory_usage,
        'disk_usage': metrics.disk_usage,
        'network_latency_ms': metrics.network_latency_ms,
        'error_count': metrics.error_count,
        'response_time_ms': metrics.response_time_ms,
        'active_connections': metrics.active_connections,
        'hour': hour,
        'day_of_week': day_of_week,
        'is_weekend': int(day_of_week >= 5),
        'is_business_hours': int(9 <= hour <= 17 and day_of_week < 5),
        'is_night': int(hour >= 22 or hour <= 6)
    }
    
    # Create interaction features
    features.update({
        'cpu_memory_product': metrics.cpu_usage * metrics.memory_usage,
        'resource_pressure': (metrics.cpu_usage + metrics.memory_usage + metrics.disk_usage) / 3,
        'performance_score': metrics.response_time_ms / (metrics.active_connections + 1),
        'error_per_connection': metrics.error_count / (metrics.active_connections + 1),
        'system_stress': int(metrics.cpu_usage > 80) + int(metrics.memory_usage > 80) + int(metrics.error_count > 5)
    })
    
    # Create anomaly features (simplified - using general thresholds)
    cpu_mean, cpu_std = 40, 20  # Approximate values
    memory_mean, memory_std = 50, 25
    response_mean, response_std = 300, 100
    
    features.update({
        'cpu_usage_zscore': (metrics.cpu_usage - cpu_mean) / cpu_std,
        'cpu_usage_is_anomaly': int(abs((metrics.cpu_usage - cpu_mean) / cpu_std) > 2),
        'memory_usage_zscore': (metrics.memory_usage - memory_mean) / memory_std,
        'memory_usage_is_anomaly': int(abs((metrics.memory_usage - memory_mean) / memory_std) > 2),
        'response_time_ms_zscore': (metrics.response_time_ms - response_mean) / response_std,
        'response_time_ms_is_anomaly': int(abs((metrics.response_time_ms - response_mean) / response_std) > 2)
    })
    
    # Add missing features with default values
    for col in feature_columns:
        if col not in features:
            if 'rolling' in col or 'lag' in col:
                # For rolling and lag features, use current value
                base_feature = col.split('_')[0] + '_' + col.split('_')[1]
                if base_feature in features:
                    features[col] = features[base_feature]
                else:
                    features[col] = 0
            else:
                features[col] = 0
    
    # Create DataFrame with correct column order
    df = pd.DataFrame([features])
    df = df.reindex(columns=feature_columns, fill_value=0)
    
    return df

def get_recommendations(failure_prob: float, metrics: SystemMetrics) -> List[str]:
    """Generate recommendations based on prediction and metrics"""
    recommendations = []
    
    if failure_prob > 0.7:
        recommendations.append("🚨 URGENT: High failure risk detected")
    
    if metrics.cpu_usage > 80:
        recommendations.append("⚡ Scale up CPU resources or optimize CPU-intensive processes")
    
    if metrics.memory_usage > 85:
        recommendations.append("💾 Increase memory allocation or investigate memory leaks")
    
    if metrics.disk_usage > 90:
        recommendations.append("💿 Free up disk space - clear logs, temporary files")
    
    if metrics.error_count > 10:
        recommendations.append("🐛 Investigate and fix recurring errors")
    
    if metrics.response_time_ms > 1000:
        recommendations.append("🚀 Optimize application performance - check database queries")
    
    if metrics.network_latency_ms > 200:
        recommendations.append("🌐 Check network connectivity and optimize network calls")
    
    if metrics.active_connections > 500:
        recommendations.append("🔗 Monitor connection pool - consider connection limits")
    
    if failure_prob > 0.5:
        recommendations.append("📊 Enable detailed monitoring and alerting")
        recommendations.append("🔄 Consider graceful service restart during low-traffic period")
    
    if not recommendations:
        recommendations.append("✅ System appears healthy - continue monitoring")
    
    return recommendations

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "MLOps Error Prediction API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    start_time = datetime.now()  # In production, track actual start time
    uptime = str(datetime.now() - start_time)
    
    return HealthResponse(
        status="healthy" if models else "unhealthy",
        timestamp=datetime.now().isoformat(),
        models_loaded=list(models.keys()),
        uptime=uptime
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_failure(metrics: SystemMetrics):
    """Predict system failure probability"""
    try:
        if not models:
            raise HTTPException(status_code=500, detail="No models loaded")
        
        # Create features
        features_df = create_features_from_metrics(metrics)
        
        # Use the best performing model (typically XGBoost or Random Forest)
        model_name = 'xgboost' if 'xgboost' in models else list(models.keys())[0]
        model = models[model_name]
        
        # Make prediction
        if model_name == 'logistic_regression' and scaler:
            features_scaled = scaler.transform(features_df)
            failure_prob = model.predict_proba(features_scaled)[0][1]
        else:
            failure_prob = model.predict_proba(features_df)[0][1]
        
        # Determine risk level
        if failure_prob >= 0.7:
            risk_level = "HIGH"
            failure_time = datetime.now() + timedelta(minutes=15)
        elif failure_prob >= 0.4:
            risk_level = "MEDIUM"
            failure_time = datetime.now() + timedelta(hours=1)
        else:
            risk_level = "LOW"
            failure_time = None
        
        # Generate recommendations
        recommendations = get_recommendations(failure_prob, metrics)
        
        # Calculate confidence (simplified)
        confidence = min(0.95, max(0.6, 1 - abs(failure_prob - 0.5) * 2))
        
        response = PredictionResponse(
            failure_probability=round(failure_prob, 4),
            failure_risk=risk_level,
            predicted_failure_time=failure_time.isoformat() if failure_time else None,
            recommendations=recommendations,
            model_used=model_name,
            confidence=round(confidence, 3),
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"Prediction made: {failure_prob:.4f} risk level: {risk_level}")
        
        return response
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch")
async def predict_batch(metrics_list: List[SystemMetrics]):
    """Batch prediction for multiple metrics"""
    try:
        if not models:
            raise HTTPException(status_code=500, detail="No models loaded")
        
        predictions = []
        for metrics in metrics_list:
            # Reuse single prediction logic
            prediction = await predict_failure(metrics)
            predictions.append(prediction)
        
        return {"predictions": predictions}
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/models/info")
async def get_model_info():
    """Get information about loaded models"""
    if not models:
        raise HTTPException(status_code=500, detail="No models loaded")
    
    return {
        "loaded_models": list(models.keys()),
        "feature_count": len(feature_columns),
        "metadata": model_metadata
    }

@app.get("/models/features")
async def get_feature_list():
    """Get list of features used by the model"""
    return {
        "features": feature_columns,
        "feature_count": len(feature_columns)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)