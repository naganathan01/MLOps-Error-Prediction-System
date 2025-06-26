"""
Enhanced prediction API with improved risk thresholds, feature processing, and ensemble predictions.
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
    title="Enhanced MLOps Error Prediction API",
    description="Advanced system failure prediction with ensemble models",
    version="3.0.0"
)

# Global variables for models and configurations
models = {}
scalers = {}
selected_features = []
model_metadata = {}

class SystemMetrics(BaseModel):
    """Enhanced input schema for system metrics"""
    cpu_usage: float = Field(..., ge=0, le=100, description="CPU usage percentage")
    memory_usage: float = Field(..., ge=0, le=100, description="Memory usage percentage")
    disk_usage: float = Field(..., ge=0, le=100, description="Disk usage percentage")
    network_latency_ms: float = Field(..., ge=0, le=10000, description="Network latency in milliseconds")
    error_count: int = Field(..., ge=0, le=1000, description="Number of errors in the last period")
    response_time_ms: float = Field(..., ge=0, le=60000, description="Average response time in milliseconds")
    active_connections: int = Field(..., ge=0, le=10000, description="Number of active connections")
    
    # Optional time-based features
    hour: Optional[int] = Field(default=None, ge=0, le=23, description="Hour of the day")
    day_of_week: Optional[int] = Field(default=None, ge=0, le=6, description="Day of the week (0=Monday)")

class EnhancedPredictionResponse(BaseModel):
    """Enhanced response schema for predictions"""
    failure_probability: float = Field(..., description="Probability of failure (0-1)")
    failure_risk: str = Field(..., description="Risk level: MINIMAL, LOW, MEDIUM, HIGH, CRITICAL")
    confidence: float = Field(..., description="Prediction confidence")
    
    # Risk breakdown
    risk_factors: Dict[str, Dict[str, float]] = Field(..., description="Detailed risk factor analysis")
    
    # Predictions from multiple models
    model_predictions: Dict[str, float] = Field(..., description="Individual model predictions")
    ensemble_method: str = Field(..., description="Ensemble method used")
    
    # Actionable insights
    recommendations: List[str] = Field(..., description="Prioritized recommended actions")
    failure_indicators: List[str] = Field(..., description="Key failure indicators detected")
    
    # Time estimates
    predicted_failure_time: Optional[str] = Field(None, description="Estimated time of potential failure")
    time_to_action: Optional[str] = Field(None, description="Recommended time to take action")
    
    # Additional context
    system_health_score: float = Field(..., description="Overall system health score (0-100)")
    anomaly_score: float = Field(..., description="Anomaly detection score")
    trend_analysis: Dict[str, str] = Field(..., description="Trend analysis for key metrics")
    
    timestamp: str = Field(..., description="Prediction timestamp")

@app.on_event("startup")
async def load_enhanced_models():
    """Load enhanced trained models on startup"""
    try:
        model_dir = Path("models")
        
        if not model_dir.exists():
            logger.error(f"Models directory not found: {model_dir}")
            raise FileNotFoundError(f"Models directory not found: {model_dir}")
        
        global models, scalers, selected_features, model_metadata
        
        # Load enhanced models
        model_files = {
            'enhanced_random_forest': model_dir / "enhanced_random_forest_model.joblib",
            'enhanced_xgboost': model_dir / "enhanced_xgboost_model.joblib",
            'enhanced_lightgbm': model_dir / "enhanced_lightgbm_model.joblib",
            'enhanced_gradient_boosting': model_dir / "enhanced_gradient_boosting_model.joblib",
            'enhanced_logistic_regression': model_dir / "enhanced_logistic_regression_model.joblib",
            'voting_ensemble': model_dir / "voting_ensemble_model.joblib"
        }
        
        for name, file_path in model_files.items():
            if file_path.exists():
                models[name] = joblib.load(file_path)
                logger.info(f"Loaded {name} model")
        
        # Load scalers
        scaler_files = {
            'robust_scaler': model_dir / "robust_scaler.joblib",
            'standard_scaler': model_dir / "scaler.joblib"  # Fallback
        }
        
        for name, file_path in scaler_files.items():
            if file_path.exists():
                scalers[name] = joblib.load(file_path)
                logger.info(f"Loaded {name}")
        
        # Load selected features
        features_file = model_dir / "selected_features.json"
        if features_file.exists():
            with open(features_file, 'r') as f:
                selected_features = json.load(f)
            logger.info(f"Loaded {len(selected_features)} selected features")
        else:
            # Fallback to original feature columns
            fallback_features = model_dir / "feature_columns.json"
            if fallback_features.exists():
                with open(fallback_features, 'r') as f:
                    selected_features = json.load(f)
                logger.info(f"Loaded {len(selected_features)} fallback features")
        
        # Load enhanced metadata
        metadata_file = model_dir / "enhanced_training_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                model_metadata = json.load(f)
            logger.info("Loaded enhanced model metadata")
        
        if not models:
            raise ValueError("No enhanced models were loaded successfully")
        
        logger.info(f"Successfully loaded {len(models)} enhanced models")
        
    except Exception as e:
        logger.error(f"Failed to load enhanced models: {str(e)}")
        raise

def create_enhanced_features_from_metrics(metrics: SystemMetrics) -> pd.DataFrame:
    """Create enhanced feature vector with all advanced features"""
    
    # Get current time info if not provided
    now = datetime.now()
    hour = metrics.hour if metrics.hour is not None else now.hour
    day_of_week = metrics.day_of_week if metrics.day_of_week is not None else now.weekday()
    
    # Basic features
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
    }
    
    # Enhanced time features
    features.update({
        'is_weekend': int(day_of_week >= 5),
        'is_business_hours': int(9 <= hour <= 17 and day_of_week < 5),
        'is_night': int(hour >= 22 or hour <= 6),
        'is_peak_hours': int((10 <= hour <= 12) or (14 <= hour <= 16)),
        'is_off_hours': int(hour >= 18 or hour <= 8),
        
        # Cyclical encoding
        'hour_sin': np.sin(2 * np.pi * hour / 24),
        'hour_cos': np.cos(2 * np.pi * hour / 24),
        'day_sin': np.sin(2 * np.pi * day_of_week / 7),
        'day_cos': np.cos(2 * np.pi * day_of_week / 7),
    })
    
    # Advanced interaction features
    features.update({
        'cpu_memory_product': metrics.cpu_usage * metrics.memory_usage,
        'cpu_memory_max': max(metrics.cpu_usage, metrics.memory_usage),
        'resource_pressure': (metrics.cpu_usage + metrics.memory_usage + metrics.disk_usage) / 3,
        'resource_imbalance': np.std([metrics.cpu_usage, metrics.memory_usage, metrics.disk_usage]),
        
        # Performance indicators
        'performance_score': metrics.response_time_ms / (metrics.active_connections + 1),
        'throughput_estimate': metrics.active_connections / (metrics.response_time_ms / 1000 + 1),
        'error_per_connection': metrics.error_count / (metrics.active_connections + 1),
        'latency_response_ratio': metrics.network_latency_ms / (metrics.response_time_ms + 1),
        
        # Stress level indicators
        'cpu_stress_level': _get_stress_level(metrics.cpu_usage, [50, 70, 85]),
        'memory_stress_level': _get_stress_level(metrics.memory_usage, [60, 80, 90]),
        'disk_stress_level': _get_stress_level(metrics.disk_usage, [70, 85, 95]),
        
        # Critical thresholds
        'cpu_critical': int(metrics.cpu_usage > 90),
        'memory_critical': int(metrics.memory_usage > 85),
        'disk_critical': int(metrics.disk_usage > 95),
        'response_critical': int(metrics.response_time_ms > 2000),
        'error_critical': int(metrics.error_count > 10),
        'latency_critical': int(metrics.network_latency_ms > 500),
        
        # Workload characterization
        'high_cpu_low_memory': int((metrics.cpu_usage > 80) and (metrics.memory_usage < 60)),
        'high_memory_low_cpu': int((metrics.memory_usage > 80) and (metrics.cpu_usage < 60)),
        'balanced_high_load': int((metrics.cpu_usage > 70) and (metrics.memory_usage > 70)),
        'io_bound_workload': int((metrics.disk_usage > 80) or (metrics.network_latency_ms > 200)),
    })
    
    # Calculate composite scores
    features['total_stress_score'] = (features['cpu_stress_level'] + 
                                    features['memory_stress_level'] + 
                                    features['disk_stress_level'])
    
    features['critical_indicators_count'] = (features['cpu_critical'] + 
                                           features['memory_critical'] + 
                                           features['disk_critical'] + 
                                           features['response_critical'] + 
                                           features['error_critical'] + 
                                           features['latency_critical'])
    
    # Enhanced anomaly features
    features.update(_calculate_anomaly_features(metrics))
    
    # Add rolling/lag features with estimated values (since we don't have history)
    features.update(_estimate_temporal_features(metrics))
    
    # Ensure we have all selected features
    for feature in selected_features:
        if feature not in features:
            features[feature] = 0
    
    # Create DataFrame with correct column order
    df = pd.DataFrame([features])
    
    # Reindex to match training features
    if selected_features:
        df = df.reindex(columns=selected_features, fill_value=0)
    
    return df

def _get_stress_level(value: float, thresholds: List[float]) -> int:
    """Get stress level (0-3) based on thresholds"""
    for i, threshold in enumerate(thresholds):
        if value <= threshold:
            return i
    return len(thresholds)

def _calculate_anomaly_features(metrics: SystemMetrics) -> Dict[str, float]:
    """Calculate anomaly detection features"""
    # Historical statistics (estimated from typical system behavior)
    stats = {
        'cpu_usage': {'mean': 35, 'std': 20, 'median': 30, 'mad': 15},
        'memory_usage': {'mean': 50, 'std': 25, 'median': 45, 'mad': 20},
        'response_time_ms': {'mean': 300, 'std': 150, 'median': 250, 'mad': 100},
        'error_count': {'mean': 2, 'std': 3, 'median': 1, 'mad': 2},
        'network_latency_ms': {'mean': 80, 'std': 50, 'median': 70, 'mad': 30}
    }
    
    anomaly_features = {}
    
    for metric_name, metric_value in [
        ('cpu_usage', metrics.cpu_usage),
        ('memory_usage', metrics.memory_usage),
        ('response_time_ms', metrics.response_time_ms),
        ('error_count', metrics.error_count),
        ('network_latency_ms', metrics.network_latency_ms)
    ]:
        stat = stats[metric_name]
        
        # Z-score
        zscore = (metric_value - stat['mean']) / (stat['std'] + 1e-8)
        anomaly_features[f'{metric_name}_zscore'] = zscore
        anomaly_features[f'{metric_name}_zscore_abs'] = abs(zscore)
        anomaly_features[f'{metric_name}_is_anomaly_zscore'] = int(abs(zscore) > 2.5)
        
        # Modified Z-score
        modified_zscore = 0.6745 * (metric_value - stat['median']) / (stat['mad'] + 1e-8)
        anomaly_features[f'{metric_name}_modified_zscore'] = modified_zscore
        anomaly_features[f'{metric_name}_is_anomaly_modified'] = int(abs(modified_zscore) > 3.5)
        
        # Percentile estimation
        if metric_value > stat['mean'] + 2 * stat['std']:
            percentile = 0.98
        elif metric_value > stat['mean'] + stat['std']:
            percentile = 0.85
        elif metric_value > stat['mean']:
            percentile = 0.65
        else:
            percentile = 0.35
        
        anomaly_features[f'{metric_name}_percentile'] = percentile
        anomaly_features[f'{metric_name}_is_top_5pct'] = int(percentile > 0.95)
    
    # Composite anomaly score
    anomaly_indicators = [v for k, v in anomaly_features.items() if '_is_anomaly_' in k]
    anomaly_features['total_anomaly_score'] = sum(anomaly_indicators)
    anomaly_features['is_multi_anomaly'] = int(sum(anomaly_indicators) >= 3)
    
    return anomaly_features

def _estimate_temporal_features(metrics: SystemMetrics) -> Dict[str, float]:
    """Estimate temporal features when historical data is not available"""
    temporal_features = {}
    
    # Simulate some rolling/lag features with current values
    # In production, these would be calculated from actual historical data
    base_metrics = ['cpu_usage', 'memory_usage', 'error_count', 'response_time_ms']
    
    for metric in base_metrics:
        value = getattr(metrics, metric)
        
        # Estimate rolling means (assume some stability)
        for window in [3, 5, 10]:
            temporal_features[f'{metric}_rolling_mean_{window}'] = value * (0.9 + np.random.uniform(-0.1, 0.1))
            temporal_features[f'{metric}_rolling_std_{window}'] = value * 0.1
            temporal_features[f'{metric}_diff_from_mean_{window}'] = value * np.random.uniform(-0.1, 0.1)
        
        # Estimate lag features (assume some temporal correlation)
        for lag in [1, 2, 3]:
            temporal_features[f'{metric}_lag_{lag}'] = value * (0.95 + np.random.uniform(-0.1, 0.1))
            temporal_features[f'{metric}_lag_diff_{lag}'] = value * np.random.uniform(-0.05, 0.05)
    
    return temporal_features

def calculate_detailed_risk_factors(metrics: SystemMetrics) -> Dict[str, Dict[str, float]]:
    """Calculate detailed risk factor analysis"""
    risk_factors = {
        'resource_utilization': {
            'cpu_risk': _calculate_resource_risk(metrics.cpu_usage, [70, 80, 90, 95]),
            'memory_risk': _calculate_resource_risk(metrics.memory_usage, [75, 85, 90, 95]),
            'disk_risk': _calculate_resource_risk(metrics.disk_usage, [80, 90, 95, 98])
        },
        'performance_degradation': {
            'response_time_risk': _calculate_performance_risk(metrics.response_time_ms, [500, 1000, 2000, 5000]),
            'network_latency_risk': _calculate_performance_risk(metrics.network_latency_ms, [100, 200, 500, 1000]),
            'throughput_risk': _calculate_throughput_risk(metrics.active_connections, metrics.response_time_ms)
        },
        'stability_indicators': {
            'error_rate_risk': _calculate_error_risk(metrics.error_count, [3, 5, 10, 20]),
            'connection_risk': _calculate_connection_risk(metrics.active_connections, [100, 200, 400, 800]),
            'anomaly_risk': _calculate_anomaly_risk(metrics)
        },
        'system_patterns': {
            'cascade_failure_risk': _calculate_cascade_risk(metrics),
            'memory_leak_risk': _calculate_memory_leak_risk(metrics),
            'overload_risk': _calculate_overload_risk(metrics)
        }
    }
    
    return risk_factors

def _calculate_resource_risk(value: float, thresholds: List[float]) -> float:
    """Calculate risk based on resource utilization thresholds"""
    for i, threshold in enumerate(thresholds):
        if value <= threshold:
            return i / len(thresholds)
    return 1.0

def _calculate_performance_risk(value: float, thresholds: List[float]) -> float:
    """Calculate risk based on performance metrics"""
    for i, threshold in enumerate(thresholds):
        if value <= threshold:
            return i / len(thresholds)
    return 1.0

def _calculate_throughput_risk(connections: int, response_time: float) -> float:
    """Calculate throughput-based risk"""
    if response_time <= 0:
        return 0.0
    
    throughput = connections / (response_time / 1000)
    
    if throughput > 100:
        return 0.1
    elif throughput > 50:
        return 0.3
    elif throughput > 20:
        return 0.6
    else:
        return 0.9

def _calculate_error_risk(error_count: int, thresholds: List[int]) -> float:
    """Calculate error-based risk"""
    for i, threshold in enumerate(thresholds):
        if error_count <= threshold:
            return i / len(thresholds)
    return 1.0

def _calculate_connection_risk(connections: int, thresholds: List[int]) -> float:
    """Calculate connection-based risk"""
    for i, threshold in enumerate(thresholds):
        if connections <= threshold:
            return i / len(thresholds)
    return 1.0

def _calculate_anomaly_risk(metrics: SystemMetrics) -> float:
    """Calculate anomaly-based risk"""
    anomaly_features = _calculate_anomaly_features(metrics)
    anomaly_score = anomaly_features.get('total_anomaly_score', 0)
    return min(1.0, anomaly_score / 10)

def _calculate_cascade_risk(metrics: SystemMetrics) -> float:
    """Calculate cascade failure risk"""
    high_cpu = metrics.cpu_usage > 80
    high_memory = metrics.memory_usage > 80
    high_errors = metrics.error_count > 5
    slow_response = metrics.response_time_ms > 1000
    
    cascade_indicators = sum([high_cpu, high_memory, high_errors, slow_response])
    return cascade_indicators / 4

def _calculate_memory_leak_risk(metrics: SystemMetrics) -> float:
    """Calculate memory leak risk"""
    if metrics.memory_usage > 85 and metrics.response_time_ms > 500:
        return 0.8
    elif metrics.memory_usage > 75:
        return 0.4
    else:
        return 0.1

def _calculate_overload_risk(metrics: SystemMetrics) -> float:
    """Calculate system overload risk"""
    resource_pressure = (metrics.cpu_usage + metrics.memory_usage) / 2
    performance_impact = min(1.0, metrics.response_time_ms / 1000)
    
    return (resource_pressure / 100 + performance_impact) / 2

def get_enhanced_recommendations(failure_prob: float, metrics: SystemMetrics, 
                               risk_factors: Dict) -> List[str]:
    """Generate enhanced, prioritized recommendations"""
    recommendations = []
    
    # Critical alerts based on failure probability
    if failure_prob > 0.85:
        recommendations.append("🚨 CRITICAL: System failure imminent - initiate emergency protocols")
        recommendations.append("📞 Alert operations team and management immediately")
        recommendations.append("🔄 Prepare for immediate failover to backup systems")
        recommendations.append("💾 Ensure all critical data is backed up")
    elif failure_prob > 0.7:
        recommendations.append("⚠️ HIGH RISK: System failure likely within 15-30 minutes")
        recommendations.append("📊 Enable maximum monitoring and alerting")
        recommendations.append("🔧 Prepare maintenance window for immediate fixes")
    elif failure_prob > 0.5:
        recommendations.append("🔶 MEDIUM RISK: Increased monitoring required")
        recommendations.append("📈 Review system metrics and trends")
    
    # Specific recommendations based on risk factors
    resource_risks = risk_factors.get('resource_utilization', {})
    
    if resource_risks.get('cpu_risk', 0) > 0.7:
        recommendations.append("⚡ URGENT: CPU usage critical - scale horizontally or kill non-essential processes")
        if metrics.cpu_usage > 95:
            recommendations.append("🔥 CPU at maximum - consider immediate load balancing or circuit breaker activation")
    
    if resource_risks.get('memory_risk', 0) > 0.7:
        recommendations.append("💾 URGENT: Memory usage critical - investigate memory leaks and restart services")
        if metrics.memory_usage > 95:
            recommendations.append("🔥 Memory exhaustion imminent - emergency memory cleanup required")
    
    if resource_risks.get('disk_risk', 0) > 0.8:
        recommendations.append("💿 CRITICAL: Disk space critically low - immediate cleanup required")
        recommendations.append("🗂️ Emergency actions: clear logs, temp files, and old data")
    
    performance_risks = risk_factors.get('performance_degradation', {})
    
    if performance_risks.get('response_time_risk', 0) > 0.6:
        recommendations.append("🚀 URGENT: Response time severely degraded")
        recommendations.append("🔍 Immediate actions: check database queries, API timeouts, and network bottlenecks")
    
    if performance_risks.get('network_latency_risk', 0) > 0.6:
        recommendations.append("🌐 URGENT: Network latency critical - check connectivity and routing")
        recommendations.append("📡 Verify network infrastructure and consider CDN activation")
    
    stability_risks = risk_factors.get('stability_indicators', {})
    
    if stability_risks.get('error_rate_risk', 0) > 0.6:
        recommendations.append("🐛 URGENT: High error rate detected - investigate root cause immediately")
        recommendations.append("📝 Check error logs for patterns and implement circuit breakers")
    
    system_patterns = risk_factors.get('system_patterns', {})
    
    if system_patterns.get('cascade_failure_risk', 0) > 0.7:
        recommendations.append("🌊 CASCADE FAILURE RISK: Multiple systems affected")
        recommendations.append("🛡️ Activate all circuit breakers and reduce system load")
    
    if system_patterns.get('memory_leak_risk', 0) > 0.7:
        recommendations.append("🧠 MEMORY LEAK DETECTED: Plan service restart during low-traffic period")
    
    # Preventive recommendations
    if failure_prob > 0.3:
        recommendations.append("📊 Enable detailed monitoring and create incident response plan")
        recommendations.append("💾 Verify backup systems and recovery procedures")
    
    if not recommendations:
        recommendations.append("✅ System appears healthy - continue monitoring")
        recommendations.append("📈 Maintain current operational procedures")
        recommendations.append("🔍 Consider proactive optimization during low-traffic periods")
    
    return recommendations

def detect_failure_indicators(metrics: SystemMetrics, risk_factors: Dict) -> List[str]:
    """Detect key failure indicators"""
    indicators = []
    
    # Resource-based indicators
    if metrics.cpu_usage > 90:
        indicators.append(f"CPU usage critical: {metrics.cpu_usage:.1f}%")
    
    if metrics.memory_usage > 85:
        indicators.append(f"Memory usage high: {metrics.memory_usage:.1f}%")
    
    if metrics.disk_usage > 95:
        indicators.append(f"Disk space critical: {metrics.disk_usage:.1f}%")
    
    # Performance indicators
    if metrics.response_time_ms > 2000:
        indicators.append(f"Response time degraded: {metrics.response_time_ms:.0f}ms")
    
    if metrics.network_latency_ms > 500:
        indicators.append(f"Network latency high: {metrics.network_latency_ms:.0f}ms")
    
    if metrics.error_count > 10:
        indicators.append(f"Error rate elevated: {metrics.error_count} errors")
    
    # Pattern-based indicators
    if risk_factors.get('system_patterns', {}).get('cascade_failure_risk', 0) > 0.7:
        indicators.append("Cascade failure pattern detected")
    
    if risk_factors.get('system_patterns', {}).get('memory_leak_risk', 0) > 0.7:
        indicators.append("Memory leak pattern detected")
    
    # Anomaly indicators
    anomaly_features = _calculate_anomaly_features(metrics)
    if anomaly_features.get('is_multi_anomaly', 0):
        indicators.append("Multiple anomalies detected")
    
    return indicators

def calculate_system_health_score(metrics: SystemMetrics, failure_prob: float) -> float:
    """Calculate overall system health score (0-100)"""
    # Resource health (40% weight)
    resource_health = (
        (100 - metrics.cpu_usage) * 0.4 +
        (100 - metrics.memory_usage) * 0.4 +
        (100 - metrics.disk_usage) * 0.2
    ) / 100
    
    # Performance health (30% weight)
    response_health = max(0, 1 - metrics.response_time_ms / 5000)
    latency_health = max(0, 1 - metrics.network_latency_ms / 1000)
    performance_health = (response_health + latency_health) / 2
    
    # Stability health (20% weight)
    error_health = max(0, 1 - metrics.error_count / 20)
    
    # Failure probability impact (10% weight)
    failure_health = 1 - failure_prob
    
    # Weighted average
    health_score = (
        resource_health * 0.4 +
        performance_health * 0.3 +
        error_health * 0.2 +
        failure_health * 0.1
    ) * 100
    
    return max(0, min(100, health_score))

def analyze_trends(metrics: SystemMetrics) -> Dict[str, str]:
    """Analyze trends for key metrics (simplified without historical data)"""
    trends = {}
    
    # Simple trend analysis based on current values relative to thresholds
    if metrics.cpu_usage > 80:
        trends['cpu_usage'] = 'increasing' if metrics.cpu_usage > 90 else 'elevated'
    elif metrics.cpu_usage < 30:
        trends['cpu_usage'] = 'stable_low'
    else:
        trends['cpu_usage'] = 'stable'
    
    if metrics.memory_usage > 80:
        trends['memory_usage'] = 'increasing' if metrics.memory_usage > 90 else 'elevated'
    elif metrics.memory_usage < 40:
        trends['memory_usage'] = 'stable_low'
    else:
        trends['memory_usage'] = 'stable'
    
    if metrics.response_time_ms > 1000:
        trends['response_time'] = 'degrading'
    elif metrics.response_time_ms < 200:
        trends['response_time'] = 'optimal'
    else:
        trends['response_time'] = 'acceptable'
    
    if metrics.error_count > 5:
        trends['error_rate'] = 'increasing'
    elif metrics.error_count == 0:
        trends['error_rate'] = 'none'
    else:
        trends['error_rate'] = 'low'
    
    return trends

@app.post("/predict", response_model=EnhancedPredictionResponse)
async def enhanced_predict_failure(metrics: SystemMetrics):
    """Enhanced system failure prediction with comprehensive analysis"""
    try:
        if not models:
            raise HTTPException(status_code=500, detail="No enhanced models loaded")
        
        # Create enhanced features
        features_df = create_enhanced_features_from_metrics(metrics)
        
        # Get predictions from all available models
        model_predictions = {}
        ensemble_predictions = []
        
        for model_name, model in models.items():
            try:
                if 'logistic_regression' in model_name and 'robust_scaler' in scalers:
                    # Use proper scaling for logistic regression
                    prediction_prob = model.predict_proba(features_df)[0][1]
                else:
                    prediction_prob = model.predict_proba(features_df)[0][1]
                
                model_predictions[model_name] = float(prediction_prob)
                ensemble_predictions.append(prediction_prob)
                
            except Exception as e:
                logger.warning(f"Model {model_name} prediction failed: {str(e)}")
                continue
        
        if not ensemble_predictions:
            raise HTTPException(status_code=500, detail="All model predictions failed")
        
        # Ensemble prediction (weighted by model performance if available)
        if 'voting_ensemble' in model_predictions:
            # Use ensemble model if available
            failure_prob = model_predictions['voting_ensemble']
            ensemble_method = "voting_ensemble"
        else:
            # Simple average of available models
            failure_prob = np.mean(ensemble_predictions)
            ensemble_method = "simple_average"
        
        # Calculate confidence based on model agreement
        prediction_std = np.std(ensemble_predictions)
        confidence = max(0.6, min(0.95, 1 - prediction_std * 2))
        
        # Determine risk level with enhanced thresholds
        if failure_prob >= 0.85:
            risk_level = "CRITICAL"
            failure_time = datetime.now() + timedelta(minutes=5)
            action_time = "immediately"
        elif failure_prob >= 0.70:
            risk_level = "HIGH"
            failure_time = datetime.now() + timedelta(minutes=15)
            action_time = "within 5 minutes"
        elif failure_prob >= 0.50:
            risk_level = "MEDIUM"
            failure_time = datetime.now() + timedelta(hours=1)
            action_time = "within 15 minutes"
        elif failure_prob >= 0.25:
            risk_level = "LOW"
            failure_time = datetime.now() + timedelta(hours=4)
            action_time = "within 1 hour"
        else:
            risk_level = "MINIMAL"
            failure_time = None
            action_time = "routine monitoring"
        
        # Calculate detailed risk factors
        risk_factors = calculate_detailed_risk_factors(metrics)
        
        # Generate recommendations and failure indicators
        recommendations = get_enhanced_recommendations(failure_prob, metrics, risk_factors)
        failure_indicators = detect_failure_indicators(metrics, risk_factors)
        
        # Calculate additional metrics
        system_health_score = calculate_system_health_score(metrics, failure_prob)
        anomaly_score = _calculate_anomaly_features(metrics).get('total_anomaly_score', 0) / 10
        trend_analysis = analyze_trends(metrics)
        
        response = EnhancedPredictionResponse(
            failure_probability=round(failure_prob, 4),
            failure_risk=risk_level,
            confidence=round(confidence, 3),
            risk_factors=risk_factors,
            model_predictions=model_predictions,
            ensemble_method=ensemble_method,
            recommendations=recommendations,
            failure_indicators=failure_indicators,
            predicted_failure_time=failure_time.isoformat() if failure_time else None,
            time_to_action=action_time,
            system_health_score=round(system_health_score, 1),
            anomaly_score=round(anomaly_score, 3),
            trend_analysis=trend_analysis,
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"Enhanced prediction: {failure_prob:.4f} risk: {risk_level} health: {system_health_score:.1f}")
        
        return response
        
    except Exception as e:
        logger.error(f"Enhanced prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Enhanced prediction failed: {str(e)}")

@app.get("/health")
async def enhanced_health_check():
    """Enhanced health check with model information"""
    return {
        "status": "healthy" if models else "unhealthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": list(models.keys()),
        "features_count": len(selected_features),
        "scalers_loaded": list(scalers.keys()),
        "version": "3.0.0",
        "enhancements": [
            "Ensemble predictions",
            "Detailed risk factor analysis", 
            "Enhanced feature engineering",
            "Comprehensive recommendations",
            "System health scoring",
            "Trend analysis"
        ]
    }

@app.get("/")
async def root():
    """Root endpoint with enhanced information"""
    return {
        "message": "Enhanced MLOps Error Prediction API",
        "version": "3.0.0",
        "status": "running",
        "capabilities": [
            "Multi-model ensemble predictions",
            "Real-time risk factor analysis",
            "Comprehensive failure detection",
            "Actionable recommendations",
            "System health assessment",
            "Advanced anomaly detection"
        ],
        "models_available": len(models),
        "features_engineered": len(selected_features)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)