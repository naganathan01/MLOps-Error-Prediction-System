"""
Prediction module for MLOps Error Prediction System.
File: src/models/prediction.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
import logging
from datetime import datetime
from typing import List
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SystemMetrics(BaseModel):
    """Input schema for system metrics"""
    cpu_usage: float = Field(..., ge=0, le=100, description="CPU usage percentage")
    memory_usage: float = Field(..., ge=0, le=100, description="Memory usage percentage")
    disk_usage: float = Field(..., ge=0, le=100, description="Disk usage percentage")
    network_latency_ms: float = Field(..., ge=0, description="Network latency in milliseconds")
    error_count: int = Field(..., ge=0, description="Number of errors")
    response_time_ms: float = Field(..., ge=0, description="Response time in milliseconds")
    active_connections: int = Field(..., ge=0, description="Number of active connections")
    hour: int = Field(default=None, ge=0, le=23, description="Hour of the day")
    day_of_week: int = Field(default=None, ge=0, le=6, description="Day of the week")

class PredictionResponse(BaseModel):
    """Response schema for predictions"""
    failure_probability: float
    failure_risk: str
    confidence: float
    recommendations: List[str]
    model_used: str
    timestamp: str

class PredictionEngine:
    def __init__(self, model_dir="models"):
        self.model_dir = Path(model_dir)
        self.models = {}
        self.scaler = None
        self.feature_columns = []
        self.load_models()
    
    def load_models(self):
        """Load trained models and artifacts"""
        logger.info("📥 Loading models...")
        
        # Load Random Forest
        rf_path = self.model_dir / "random_forest_model.joblib"
        if rf_path.exists():
            self.models['random_forest'] = joblib.load(rf_path)
            logger.info("   Loaded Random Forest")
        
        # Load Logistic Regression
        lr_path = self.model_dir / "logistic_regression_model.joblib"
        if lr_path.exists():
            self.models['logistic_regression'] = joblib.load(lr_path)
            logger.info("   Loaded Logistic Regression")
        
        # Load XGBoost if available
        xgb_path = self.model_dir / "xgboost_model.joblib"
        if xgb_path.exists():
            self.models['xgboost'] = joblib.load(xgb_path)
            logger.info("   Loaded XGBoost")
        
        # Load scaler
        scaler_path = self.model_dir / "scaler.joblib"
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
            logger.info("   Loaded scaler")
        
        # Load feature columns
        features_path = self.model_dir / "feature_columns.json"
        if features_path.exists():
            with open(features_path, 'r') as f:
                self.feature_columns = json.load(f)
            logger.info(f"   Loaded {len(self.feature_columns)} feature columns")
        
        if not self.models:
            raise ValueError("No models loaded!")
    
    def create_features_from_metrics(self, metrics: SystemMetrics) -> pd.DataFrame:
        """Create feature vector from input metrics"""
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
        }
        
        # Create derived features (same as training)
        features.update({
            'cpu_memory_product': metrics.cpu_usage * metrics.memory_usage,
            'resource_pressure': (metrics.cpu_usage + metrics.memory_usage + metrics.disk_usage) / 3,
            'performance_ratio': metrics.response_time_ms / (metrics.active_connections + 1),
            'error_rate': metrics.error_count / (metrics.active_connections + 1),
            
            'cpu_high': int(metrics.cpu_usage > 80),
            'memory_high': int(metrics.memory_usage > 85),
            'response_slow': int(metrics.response_time_ms > 1000),
            'errors_high': int(metrics.error_count > 5),
        })
        
        features['total_stress'] = (features['cpu_high'] + features['memory_high'] + 
                                   features['response_slow'] + features['errors_high'])
        
        # Time features
        features.update({
            'hour_sin': np.sin(2 * np.pi * hour / 24),
            'hour_cos': np.cos(2 * np.pi * hour / 24),
            'day_sin': np.sin(2 * np.pi * day_of_week / 7),
            'day_cos': np.cos(2 * np.pi * day_of_week / 7),
        })
        
        # Add missing features with default values (for rolling/lag features)
        for col in self.feature_columns:
            if col not in features:
                if 'rolling_mean' in col:
                    base_col = col.split('_rolling_mean')[0]
                    features[col] = features.get(base_col, 0)
                elif 'rolling_max' in col:
                    base_col = col.split('_rolling_max')[0]
                    features[col] = features.get(base_col, 0)
                elif 'lag' in col:
                    base_col = col.split('_lag')[0]
                    features[col] = features.get(base_col, 0)
                else:
                    features[col] = 0
        
        # Create DataFrame with correct column order
        df = pd.DataFrame([features])
        df = df.reindex(columns=self.feature_columns, fill_value=0)
        
        return df
    
    def predict(self, metrics: SystemMetrics) -> PredictionResponse:
        """Make prediction"""
        if not self.models:
            raise ValueError("No models loaded")
        
        # Create features
        features_df = self.create_features_from_metrics(metrics)
        
        # Use best available model (priority: XGBoost > Random Forest > Logistic Regression)
        if 'xgboost' in self.models:
            model = self.models['xgboost']
            model_name = 'xgboost'
            failure_prob = model.predict_proba(features_df)[0][1]
        elif 'random_forest' in self.models:
            model = self.models['random_forest']
            model_name = 'random_forest'
            failure_prob = model.predict_proba(features_df)[0][1]
        elif 'logistic_regression' in self.models:
            model = self.models['logistic_regression']
            model_name = 'logistic_regression'
            # Scale features for logistic regression
            if self.scaler:
                features_scaled = self.scaler.transform(features_df)
                failure_prob = model.predict_proba(features_scaled)[0][1]
            else:
                failure_prob = model.predict_proba(features_df)[0][1]
        else:
            raise ValueError("No valid models available")
        
        # Determine risk level
        if failure_prob >= 0.8:
            risk_level = "CRITICAL"
        elif failure_prob >= 0.6:
            risk_level = "HIGH"
        elif failure_prob >= 0.4:
            risk_level = "MEDIUM"
        elif failure_prob >= 0.2:
            risk_level = "LOW"
        else:
            risk_level = "MINIMAL"
        
        # Generate recommendations
        recommendations = self.get_recommendations(failure_prob, metrics)
        
        # Calculate confidence (simplified)
        confidence = min(0.95, max(0.6, 1 - abs(failure_prob - 0.5)))
        
        return PredictionResponse(
            failure_probability=round(failure_prob, 4),
            failure_risk=risk_level,
            confidence=round(confidence, 3),
            recommendations=recommendations,
            model_used=model_name,
            timestamp=datetime.now().isoformat()
        )
    
    def get_recommendations(self, failure_prob: float, metrics: SystemMetrics) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        if failure_prob > 0.7:
            recommendations.append("🚨 URGENT: High failure risk - take immediate action")
        
        # Resource-specific recommendations
        if metrics.cpu_usage > 85:
            recommendations.append("⚡ CPU Critical: Scale resources or optimize processes")
        elif metrics.cpu_usage > 75:
            recommendations.append("⚠️ CPU High: Monitor and prepare to scale")
        
        if metrics.memory_usage > 90:
            recommendations.append("💾 Memory Critical: Check for leaks, restart services")
        elif metrics.memory_usage > 80:
            recommendations.append("💾 Memory High: Monitor memory usage closely")
        
        if metrics.error_count > 10:
            recommendations.append("🐛 Error Spike: Check logs and fix issues immediately")
        elif metrics.error_count > 5:
            recommendations.append("🐛 Errors Elevated: Investigate error patterns")
        
        if metrics.response_time_ms > 2000:
            recommendations.append("🚀 Performance Critical: Optimize slow operations")
        elif metrics.response_time_ms > 1000:
            recommendations.append("🚀 Performance Degraded: Check for bottlenecks")
        
        if metrics.disk_usage > 90:
            recommendations.append("💿 Disk Critical: Clean up space immediately")
        elif metrics.disk_usage > 80:
            recommendations.append("💿 Disk High: Plan cleanup activities")
        
        if metrics.network_latency_ms > 500:
            recommendations.append("🌐 Network Critical: Check connectivity and routing")
        elif metrics.network_latency_ms > 200:
            recommendations.append("🌐 Network Slow: Monitor network performance")
        
        if not recommendations:
            recommendations.append("✅ System healthy - continue normal monitoring")
        
        return recommendations

def main():
    """Main function for standalone testing"""
    logger.info("🚀 Testing prediction engine...")
    
    try:
        engine = PredictionEngine()
        
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
        
        prediction = engine.predict(test_metrics)
        
        logger.info("✅ Test Prediction Results:")
        logger.info(f"   Failure Probability: {prediction.failure_probability:.4f}")
        logger.info(f"   Risk Level: {prediction.failure_risk}")
        logger.info(f"   Confidence: {prediction.confidence:.3f}")
        logger.info(f"   Model Used: {prediction.model_used}")
        logger.info(f"   Recommendations:")
        for i, rec in enumerate(prediction.recommendations, 1):
            logger.info(f"     {i}. {rec}")
        
        return prediction
        
    except Exception as e:
        logger.error(f"❌ Prediction test failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
