"""
Improved prediction logic with better risk thresholds and feature processing.
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
    title="MLOps Error Prediction API - Improved",
    description="Predict system failures with better accuracy",
    version="2.0.0"
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
    failure_risk: str = Field(..., description="Risk level: LOW, MEDIUM, HIGH, CRITICAL")
    predicted_failure_time: Optional[str] = Field(None, description="Estimated time of potential failure")
    recommendations: List[str] = Field(..., description="Recommended actions")
    model_used: str = Field(..., description="Model used for prediction")
    confidence: float = Field(..., description="Prediction confidence")
    risk_factors: Dict[str, float] = Field(..., description="Individual risk factor scores")
    timestamp: str = Field(..., description="Prediction timestamp")

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
    """Create enhanced feature vector from input metrics"""
    
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
    
    # Enhanced interaction features
    features.update({
        'cpu_memory_product': metrics.cpu_usage * metrics.memory_usage,
        'resource_pressure': (metrics.cpu_usage + metrics.memory_usage + metrics.disk_usage) / 3,
        'performance_score': metrics.response_time_ms / (metrics.active_connections + 1),
        'error_per_connection': metrics.error_count / (metrics.active_connections + 1),
        'system_stress': int(metrics.cpu_usage > 80) + int(metrics.memory_usage > 80) + int(metrics.error_count > 5),
        
        # Additional risk indicators
        'critical_cpu': int(metrics.cpu_usage > 90),
        'critical_memory': int(metrics.memory_usage > 90),
        'critical_disk': int(metrics.disk_usage > 95),
        'high_latency': int(metrics.network_latency_ms > 200),
        'high_errors': int(metrics.error_count > 10),
        'slow_response': int(metrics.response_time_ms > 1000),
        'high_connections': int(metrics.active_connections > 200),
        
        # Resource ratios
        'cpu_to_memory_ratio': metrics.cpu_usage / (metrics.memory_usage + 1),
        'memory_to_disk_ratio': metrics.memory_usage / (metrics.disk_usage + 1),
        'latency_to_response_ratio': metrics.network_latency_ms / (metrics.response_time_ms + 1),
        
        # Composite risk scores
        'resource_risk_score': (metrics.cpu_usage * 0.4 + metrics.memory_usage * 0.4 + metrics.disk_usage * 0.2) / 100,
        'performance_risk_score': (metrics.response_time_ms / 1000 + metrics.network_latency_ms / 500) / 2,
        'stability_risk_score': metrics.error_count / 20
    })
    
    # Enhanced anomaly features with updated thresholds
    cpu_mean, cpu_std = 35, 20  # Updated based on realistic data
    memory_mean, memory_std = 45, 25
    response_mean, response_std = 300, 150
    
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

def calculate_risk_factors(metrics: SystemMetrics) -> Dict[str, float]:
    """Calculate individual risk factor scores"""
    risk_factors = {}
    
    # CPU risk
    if metrics.cpu_usage >= 95:
        risk_factors['cpu'] = 1.0
    elif metrics.cpu_usage >= 85:
        risk_factors['cpu'] = 0.8
    elif metrics.cpu_usage >= 75:
        risk_factors['cpu'] = 0.6
    elif metrics.cpu_usage >= 65:
        risk_factors['cpu'] = 0.3
    else:
        risk_factors['cpu'] = 0.1
    
    # Memory risk
    if metrics.memory_usage >= 95:
        risk_factors['memory'] = 1.0
    elif metrics.memory_usage >= 85:
        risk_factors['memory'] = 0.8
    elif metrics.memory_usage >= 75:
        risk_factors['memory'] = 0.6
    elif metrics.memory_usage >= 65:
        risk_factors['memory'] = 0.3
    else:
        risk_factors['memory'] = 0.1
    
    # Disk risk
    if metrics.disk_usage >= 98:
        risk_factors['disk'] = 1.0
    elif metrics.disk_usage >= 90:
        risk_factors['disk'] = 0.8
    elif metrics.disk_usage >= 80:
        risk_factors['disk'] = 0.5
    else:
        risk_factors['disk'] = 0.1
    
    # Network risk
    if metrics.network_latency_ms >= 1000:
        risk_factors['network'] = 1.0
    elif metrics.network_latency_ms >= 500:
        risk_factors['network'] = 0.8
    elif metrics.network_latency_ms >= 200:
        risk_factors['network'] = 0.5
    elif metrics.network_latency_ms >= 100:
        risk_factors['network'] = 0.2
    else:
        risk_factors['network'] = 0.1
    
    # Error risk
    if metrics.error_count >= 20:
        risk_factors['errors'] = 1.0
    elif metrics.error_count >= 15:
        risk_factors['errors'] = 0.8
    elif metrics.error_count >= 10:
        risk_factors['errors'] = 0.6
    elif metrics.error_count >= 5:
        risk_factors['errors'] = 0.4
    else:
        risk_factors['errors'] = 0.1
    
    # Response time risk
    if metrics.response_time_ms >= 3000:
        risk_factors['response_time'] = 1.0
    elif metrics.response_time_ms >= 2000:
        risk_factors['response_time'] = 0.8
    elif metrics.response_time_ms >= 1000:
        risk_factors['response_time'] = 0.6
    elif metrics.response_time_ms >= 500:
        risk_factors['response_time'] = 0.3
    else:
        risk_factors['response_time'] = 0.1
    
    # Connection risk
    if metrics.active_connections >= 500:
        risk_factors['connections'] = 0.8
    elif metrics.active_connections >= 300:
        risk_factors['connections'] = 0.6
    elif metrics.active_connections >= 200:
        risk_factors['connections'] = 0.4
    elif metrics.active_connections >= 100:
        risk_factors['connections'] = 0.2
    else:
        risk_factors['connections'] = 0.1
    
    return risk_factors

def get_recommendations(failure_prob: float, metrics: SystemMetrics, risk_factors: Dict[str, float]) -> List[str]:
    """Generate enhanced recommendations based on prediction and risk factors"""
    recommendations = []
    
    # Critical alerts
    if failure_prob > 0.8:
        recommendations.append("🚨 CRITICAL: Immediate action required - system failure imminent")
        recommendations.append("📞 Alert operations team immediately")
        recommendations.append("🔄 Prepare for emergency failover")
    elif failure_prob > 0.6:
        recommendations.append("⚠️ HIGH RISK: System failure likely within 1 hour")
        recommendations.append("📊 Enable detailed monitoring")
    elif failure_prob > 0.4:
        recommendations.append("🔶 MEDIUM RISK: Monitor system closely")
    
    # Specific recommendations based on risk factors
    if risk_factors.get('cpu', 0) > 0.7:
        recommendations.append("⚡ URGENT: CPU usage critical - scale up resources or kill intensive processes")
        if metrics.cpu_usage > 95:
            recommendations.append("🔥 CPU at maximum - consider immediate load balancing")
    elif risk_factors.get('cpu', 0) > 0.5:
        recommendations.append("📈 High CPU usage - optimize processes or add capacity")
    
    if risk_factors.get('memory', 0) > 0.7:
        recommendations.append("💾 URGENT: Memory usage critical - investigate memory leaks")
        if metrics.memory_usage > 95:
            recommendations.append("🔥 Memory exhaustion imminent - restart services if necessary")
    elif risk_factors.get('memory', 0) > 0.5:
        recommendations.append("🧠 High memory usage - monitor for leaks and optimize allocation")
    
    if risk_factors.get('disk', 0) > 0.7:
        recommendations.append("💿 CRITICAL: Disk space critically low - free space immediately")
        recommendations.append("🗂️ Clean logs, temporary files, and old data")
    elif risk_factors.get('disk', 0) > 0.4:
        recommendations.append("📁 Disk usage high - schedule cleanup activities")
    
    if risk_factors.get('errors', 0) > 0.6:
        recommendations.append("🐛 URGENT: High error rate detected - investigate root cause")
        recommendations.append("📝 Review error logs for patterns")
    elif risk_factors.get('errors', 0) > 0.3:
        recommendations.append("⚠️ Elevated error rate - monitor error trends")
    
    if risk_factors.get('response_time', 0) > 0.6:
        recommendations.append("🚀 CRITICAL: Response time severely degraded - optimize immediately")
        recommendations.append("🔍 Check database queries, API calls, and bottlenecks")
    elif risk_factors.get('response_time', 0) > 0.3:
        recommendations.append("⏱️ Response time elevated - investigate performance issues")
    
    if risk_factors.get('network', 0) > 0.6:
        recommendations.append("🌐 URGENT: Network latency critical - check connectivity")
        recommendations.append("📡 Verify network infrastructure and routing")
    elif risk_factors.get('network', 0) > 0.3:
        recommendations.append("📶 Network latency elevated - monitor connection quality")
    
    if risk_factors.get('connections', 0) > 0.6:
        recommendations.append("🔗 High connection load - monitor connection pool")
        recommendations.append("⚙️ Consider connection limits and load balancing")
    
    # Preventive recommendations
    if failure_prob > 0.3:
        recommendations.append("📊 Enable detailed monitoring and alerting")
        recommendations.append("💾 Ensure recent backups are available")
        recommendations.append("📋 Review incident response procedures")
    
    # Composite risk recommendations
    high_risk_factors = [k for k, v in risk_factors.items() if v > 0.6]
    if len(high_risk_factors) >= 3:
        recommendations.append("🚨 Multiple critical systems affected - consider emergency maintenance")
    
    if not recommendations:
        recommendations.append("✅ System appears healthy - continue monitoring")
        recommendations.append("📈 Maintain current operational procedures")
    
    return recommendations

@app.post("/predict", response_model=PredictionResponse)
async def predict_failure(metrics: SystemMetrics):
    """Predict system failure probability with improved accuracy"""
    try:
        if not models:
            raise HTTPException(status_code=500, detail="No models loaded")
        
        # Create features
        features_df = create_features_from_metrics(metrics)
        
        # Calculate individual risk factors
        risk_factors = calculate_risk_factors(metrics)
        
        # Use ensemble approach for better accuracy
        predictions = []
        confidences = []
        
        for model_name, model in models.items():
            try:
                # Prepare data for specific model
                if model_name == 'logistic_regression' and scaler:
                    features_scaled = scaler.transform(features_df)
                    failure_prob = model.predict_proba(features_scaled)[0][1]
                else:
                    failure_prob = model.predict_proba(features_df)[0][1]
                
                predictions.append(failure_prob)
                
                # Calculate model-specific confidence
                confidence = min(0.95, max(0.6, 1 - abs(failure_prob - 0.5) * 2))
                confidences.append(confidence)
                
            except Exception as e:
                logger.warning(f"Model {model_name} prediction failed: {str(e)}")
                continue
        
        if not predictions:
            raise HTTPException(status_code=500, detail="All model predictions failed")
        
        # Ensemble prediction (weighted average)
        weights = np.array(confidences) / sum(confidences)
        ensemble_prob = np.average(predictions, weights=weights)
        ensemble_confidence = np.mean(confidences)
        
        # Adjust prediction based on risk factors (boost if high risk factors present)
        risk_boost = max(risk_factors.values()) * 0.3  # Up to 30% boost
        adjusted_prob = min(0.95, ensemble_prob + risk_boost)
        
        # Determine risk level with improved thresholds
        if adjusted_prob >= 0.8:
            risk_level = "CRITICAL"
            failure_time = datetime.now() + timedelta(minutes=5)
        elif adjusted_prob >= 0.6:
            risk_level = "HIGH"
            failure_time = datetime.now() + timedelta(minutes=30)
        elif adjusted_prob >= 0.4:
            risk_level = "MEDIUM"
            failure_time = datetime.now() + timedelta(hours=2)
        elif adjusted_prob >= 0.2:
            risk_level = "LOW"
            failure_time = datetime.now() + timedelta(hours=8)
        else:
            risk_level = "MINIMAL"
            failure_time = None
        
        # Generate recommendations
        recommendations = get_recommendations(adjusted_prob, metrics, risk_factors)
        
        # Select best model for reporting
        best_model_idx = np.argmax(confidences)
        best_model_name = list(models.keys())[best_model_idx]
        
        response = PredictionResponse(
            failure_probability=round(adjusted_prob, 4),
            failure_risk=risk_level,
            predicted_failure_time=failure_time.isoformat() if failure_time else None,
            recommendations=recommendations,
            model_used=f"ensemble_{best_model_name}",
            confidence=round(ensemble_confidence, 3),
            risk_factors=risk_factors,
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"Prediction made: {adjusted_prob:.4f} risk level: {risk_level}")
        logger.info(f"Risk factors: {risk_factors}")
        
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

@app.get("/health")
async def health_check():
    """Enhanced health check"""
    return {
        "status": "healthy" if models else "unhealthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": list(models.keys()),
        "features_count": len(feature_columns),
        "version": "2.0.0"
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "MLOps Error Prediction API - Improved",
        "version": "2.0.0",
        "status": "running",
        "improvements": [
            "Enhanced risk factor analysis",
            "Improved ensemble predictions",
            "Better threshold calibration",
            "More granular risk levels"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)