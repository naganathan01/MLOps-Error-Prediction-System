"""
Prediction module for making predictions with trained models.
Handles model loading, preprocessing, and prediction generation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelPredictor:
    """Main prediction class for loading models and making predictions"""
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.models = {}
        self.scaler = None
        self.feature_columns = []
        self.model_metadata = {}
        
        # Load models and artifacts
        self._load_models()
    
    def _load_models(self):
        """Load trained models and preprocessing artifacts"""
        logger.info("📥 Loading models for prediction...")
        
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Models directory not found: {self.model_dir}")
        
        # Load models
        model_files = {
            'random_forest': self.model_dir / "random_forest_model.joblib",
            'xgboost': self.model_dir / "xgboost_model.joblib",
            'logistic_regression': self.model_dir / "logistic_regression_model.joblib"
        }
        
        for name, file_path in model_files.items():
            if file_path.exists():
                try:
                    self.models[name] = joblib.load(file_path)
                    logger.info(f"   ✅ Loaded {name} model")
                except Exception as e:
                    logger.error(f"   ❌ Failed to load {name}: {str(e)}")
        
        # Load scaler
        scaler_file = self.model_dir / "scaler.joblib"
        if scaler_file.exists():
            try:
                self.scaler = joblib.load(scaler_file)
                logger.info("   ✅ Loaded scaler")
            except Exception as e:
                logger.error(f"   ❌ Failed to load scaler: {str(e)}")
        
        # Load feature columns
        features_file = self.model_dir / "feature_columns.json"
        if features_file.exists():
            try:
                with open(features_file, 'r') as f:
                    self.feature_columns = json.load(f)
                logger.info(f"   ✅ Loaded {len(self.feature_columns)} feature columns")
            except Exception as e:
                logger.error(f"   ❌ Failed to load feature columns: {str(e)}")
        
        # Load metadata
        metadata_file = self.model_dir / "training_metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    self.model_metadata = json.load(f)
                logger.info("   ✅ Loaded model metadata")
            except Exception as e:
                logger.error(f"   ❌ Failed to load metadata: {str(e)}")
        
        if not self.models:
            raise ValueError("No models were loaded successfully")
        
        logger.info(f"🎯 Ready for prediction with {len(self.models)} models")
    
    def preprocess_features(self, raw_features: Dict[str, Any]) -> pd.DataFrame:
        """Preprocess raw features into model-ready format"""
        
        # Get current time info if not provided
        now = datetime.now()
        hour = raw_features.get('hour', now.hour)
        day_of_week = raw_features.get('day_of_week', now.weekday())
        
        # Create basic features
        features = {
            'cpu_usage': float(raw_features.get('cpu_usage', 0)),
            'memory_usage': float(raw_features.get('memory_usage', 0)),
            'disk_usage': float(raw_features.get('disk_usage', 0)),
            'network_latency_ms': float(raw_features.get('network_latency_ms', 0)),
            'error_count': int(raw_features.get('error_count', 0)),
            'response_time_ms': float(raw_features.get('response_time_ms', 0)),
            'active_connections': int(raw_features.get('active_connections', 0)),
            'hour': hour,
            'day_of_week': day_of_week,
            'is_weekend': int(day_of_week >= 5),
            'is_business_hours': int(9 <= hour <= 17 and day_of_week < 5),
            'is_night': int(hour >= 22 or hour <= 6)
        }
        
        # Create interaction features
        features.update({
            'cpu_memory_product': features['cpu_usage'] * features['memory_usage'],
            'resource_pressure': (features['cpu_usage'] + features['memory_usage'] + features['disk_usage']) / 3,
            'performance_score': features['response_time_ms'] / (features['active_connections'] + 1),
            'error_per_connection': features['error_count'] / (features['active_connections'] + 1),
            'system_stress': int(features['cpu_usage'] > 80) + int(features['memory_usage'] > 80) + int(features['error_count'] > 5)
        })
        
        # Create anomaly features (simplified - using general thresholds)
        cpu_mean, cpu_std = 40, 20  # Approximate values from training
        memory_mean, memory_std = 50, 25
        response_mean, response_std = 300, 100
        
        features.update({
            'cpu_usage_zscore': (features['cpu_usage'] - cpu_mean) / cpu_std,
            'cpu_usage_is_anomaly': int(abs((features['cpu_usage'] - cpu_mean) / cpu_std) > 2),
            'memory_usage_zscore': (features['memory_usage'] - memory_mean) / memory_std,
            'memory_usage_is_anomaly': int(abs((features['memory_usage'] - memory_mean) / memory_std) > 2),
            'response_time_ms_zscore': (features['response_time_ms'] - response_mean) / response_std,
            'response_time_ms_is_anomaly': int(abs((features['response_time_ms'] - response_mean) / response_std) > 2)
        })
        
        # Add missing features with default values
        for col in self.feature_columns:
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
        df = df.reindex(columns=self.feature_columns, fill_value=0)
        
        return df
    
    def predict_single(self, features: Dict[str, Any], 
                      model_name: Optional[str] = None) -> Dict[str, Any]:
        """Make a single prediction"""
        
        # Preprocess features
        processed_features = self.preprocess_features(features)
        
        # Select model
        if model_name is None:
            # Use the best performing model (typically XGBoost or Random Forest)
            model_name = 'xgboost' if 'xgboost' in self.models else list(self.models.keys())[0]
        
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not available. Available models: {list(self.models.keys())}")
        
        model = self.models[model_name]
        
        # Prepare data for prediction
        if model_name == 'logistic_regression' and self.scaler:
            X = self.scaler.transform(processed_features)
        else:
            X = processed_features
        
        # Make prediction
        failure_prob = model.predict_proba(X)[0][1]
        failure_pred = int(failure_prob > 0.5)
        
        # Determine risk level and timing
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
        recommendations = self._generate_recommendations(failure_prob, features)
        
        # Calculate confidence (simplified)
        confidence = self._calculate_confidence(failure_prob, model_name)
        
        result = {
            'prediction': {
                'failure_probability': round(failure_prob, 4),
                'failure_prediction': failure_pred,
                'failure_risk': risk_level,
                'predicted_failure_time': failure_time.isoformat() if failure_time else None
            },
            'model_info': {
                'model_used': model_name,
                'confidence': round(confidence, 3),
                'features_used': len(self.feature_columns)
            },
            'recommendations': recommendations,
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'input_features': features
            }
        }
        
        return result
    
    def predict_batch(self, features_list: List[Dict[str, Any]], 
                     model_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Make batch predictions"""
        
        results = []
        for features in features_list:
            try:
                result = self.predict_single(features, model_name)
                results.append(result)
            except Exception as e:
                logger.error(f"Batch prediction failed for features {features}: {str(e)}")
                # Add error result
                results.append({
                    'error': str(e),
                    'input_features': features,
                    'timestamp': datetime.now().isoformat()
                })
        
        return results
    
    def predict_from_dataframe(self, df: pd.DataFrame, 
                              model_name: Optional[str] = None) -> pd.DataFrame:
        """Make predictions from a pandas DataFrame"""
        
        # Convert DataFrame to list of dictionaries
        features_list = df.to_dict('records')
        
        # Make batch predictions
        results = self.predict_batch(features_list, model_name)
        
        # Extract predictions and add to DataFrame
        predictions = []
        confidences = []
        risk_levels = []
        
        for result in results:
            if 'error' in result:
                predictions.append(None)
                confidences.append(None)
                risk_levels.append(None)
            else:
                predictions.append(result['prediction']['failure_probability'])
                confidences.append(result['model_info']['confidence'])
                risk_levels.append(result['prediction']['failure_risk'])
        
        # Add results to DataFrame
        result_df = df.copy()
        result_df['failure_probability'] = predictions
        result_df['confidence'] = confidences
        result_df['risk_level'] = risk_levels
        result_df['prediction_timestamp'] = datetime.now().isoformat()
        
        return result_df
    
    def compare_models(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Compare predictions across all available models"""
        
        results = {}
        processed_features = self.preprocess_features(features)
        
        for model_name, model in self.models.items():
            try:
                # Prepare data for specific model
                if model_name == 'logistic_regression' and self.scaler:
                    X = self.scaler.transform(processed_features)
                else:
                    X = processed_features
                
                # Make prediction
                failure_prob = model.predict_proba(X)[0][1]
                failure_pred = int(failure_prob > 0.5)
                confidence = self._calculate_confidence(failure_prob, model_name)
                
                results[model_name] = {
                    'failure_probability': round(failure_prob, 4),
                    'failure_prediction': failure_pred,
                    'confidence': round(confidence, 3)
                }
                
            except Exception as e:
                logger.error(f"Model comparison failed for {model_name}: {str(e)}")
                results[model_name] = {'error': str(e)}
        
        # Add ensemble prediction (average of all models)
        valid_probs = [r['failure_probability'] for r in results.values() 
                      if 'failure_probability' in r]
        
        if valid_probs:
            ensemble_prob = np.mean(valid_probs)
            ensemble_std = np.std(valid_probs)
            
            results['ensemble'] = {
                'failure_probability': round(ensemble_prob, 4),
                'failure_prediction': int(ensemble_prob > 0.5),
                'confidence': round(1 - min(ensemble_std, 0.5), 3),  # Higher std = lower confidence
                'model_agreement': round(1 - ensemble_std, 3)
            }
        
        return {
            'model_predictions': results,
            'input_features': features,
            'timestamp': datetime.now().isoformat()
        }
    
    def _generate_recommendations(self, failure_prob: float, 
                                features: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on prediction and features"""
        recommendations = []
        
        if failure_prob > 0.7:
            recommendations.append("🚨 URGENT: High failure risk detected - immediate action required")
        elif failure_prob > 0.4:
            recommendations.append("⚠️ MEDIUM: Elevated failure risk - monitor closely")
        
        # CPU recommendations
        cpu_usage = features.get('cpu_usage', 0)
        if cpu_usage > 80:
            recommendations.append("⚡ Scale up CPU resources or optimize CPU-intensive processes")
        elif cpu_usage > 70:
            recommendations.append("📊 Monitor CPU usage - approaching high threshold")
        
        # Memory recommendations
        memory_usage = features.get('memory_usage', 0)
        if memory_usage > 85:
            recommendations.append("💾 Increase memory allocation or investigate memory leaks")
        elif memory_usage > 75:
            recommendations.append("🔍 Monitor memory usage - consider cleanup")
        
        # Disk recommendations
        disk_usage = features.get('disk_usage', 0)
        if disk_usage > 90:
            recommendations.append("💿 CRITICAL: Free up disk space immediately")
        elif disk_usage > 80:
            recommendations.append("🗂️ Clean up disk space - clear logs and temporary files")
        
        # Error recommendations
        error_count = features.get('error_count', 0)
        if error_count > 10:
            recommendations.append("🐛 Investigate and fix recurring errors")
        elif error_count > 5:
            recommendations.append("📝 Review error logs for patterns")
        
        # Performance recommendations
        response_time = features.get('response_time_ms', 0)
        if response_time > 1000:
            recommendations.append("🚀 Critical: Optimize application performance")
        elif response_time > 500:
            recommendations.append("⏱️ Consider performance optimization")
        
        # Network recommendations
        network_latency = features.get('network_latency_ms', 0)
        if network_latency > 200:
            recommendations.append("🌐 Check network connectivity and optimize network calls")
        
        # Connection recommendations
        active_connections = features.get('active_connections', 0)
        if active_connections > 500:
            recommendations.append("🔗 Monitor connection pool - consider connection limits")
        
        # General recommendations based on risk level
        if failure_prob > 0.5:
            recommendations.append("📊 Enable detailed monitoring and alerting")
            recommendations.append("🔄 Consider graceful service restart during low-traffic period")
            recommendations.append("📞 Notify operations team")
        
        if not recommendations:
            recommendations.append("✅ System appears healthy - continue monitoring")
        
        return recommendations
    
    def _calculate_confidence(self, failure_prob: float, model_name: str) -> float:
        """Calculate prediction confidence based on probability and model characteristics"""
        # Basic confidence calculation
        # Higher confidence for predictions closer to 0 or 1
        base_confidence = 1 - 2 * abs(failure_prob - 0.5)
        
        # Model-specific adjustments
        model_adjustments = {
            'random_forest': 0.05,    # Generally more confident
            'xgboost': 0.03,          # Slightly more confident
            'logistic_regression': 0.0  # Baseline
        }
        
        adjustment = model_adjustments.get(model_name, 0.0)
        confidence = min(0.95, max(0.5, base_confidence + adjustment))
        
        return confidence
    
    def get_feature_importance(self, model_name: Optional[str] = None) -> Dict[str, float]:
        """Get feature importance from trained models"""
        
        if model_name is None:
            model_name = 'random_forest' if 'random_forest' in self.models else list(self.models.keys())[0]
        
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not available")
        
        model = self.models[model_name]
        
        if not hasattr(model, 'feature_importances_'):
            raise ValueError(f"Model '{model_name}' doesn't have feature importance")
        
        importance_dict = dict(zip(self.feature_columns, model.feature_importances_))
        
        # Sort by importance
        sorted_importance = dict(sorted(importance_dict.items(), 
                                      key=lambda x: x[1], reverse=True))
        
        return sorted_importance
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        
        info = {
            'models_loaded': list(self.models.keys()),
            'feature_count': len(self.feature_columns),
            'has_scaler': self.scaler is not None,
            'metadata': self.model_metadata,
            'model_details': {}
        }
        
        for model_name, model in self.models.items():
            model_info = {
                'type': type(model).__name__,
                'has_feature_importance': hasattr(model, 'feature_importances_'),
                'parameters': model.get_params() if hasattr(model, 'get_params') else {}
            }
            info['model_details'][model_name] = model_info
        
        return info
    
    def save_predictions(self, predictions: List[Dict[str, Any]], 
                        filepath: str):
        """Save predictions to file"""
        
        # Convert to DataFrame for easier saving
        prediction_records = []
        
        for pred in predictions:
            if 'error' not in pred:
                record = {
                    'timestamp': pred['metadata']['timestamp'],
                    'failure_probability': pred['prediction']['failure_probability'],
                    'failure_risk': pred['prediction']['failure_risk'],
                    'model_used': pred['model_info']['model_used'],
                    'confidence': pred['model_info']['confidence']
                }
                
                # Add input features
                for key, value in pred['metadata']['input_features'].items():
                    record[f'input_{key}'] = value
                
                prediction_records.append(record)
        
        # Save to CSV
        df = pd.DataFrame(prediction_records)
        df.to_csv(filepath, index=False)
        
        logger.info(f"💾 Saved {len(prediction_records)} predictions to {filepath}")

def main():
    """Example usage of ModelPredictor"""
    
    try:
        # Initialize predictor
        predictor = ModelPredictor()
        
        # Example prediction
        sample_features = {
            'cpu_usage': 85.0,
            'memory_usage': 90.0,
            'disk_usage': 45.0,
            'network_latency_ms': 120.0,
            'error_count': 3,
            'response_time_ms': 450.0,
            'active_connections': 75
        }
        
        # Single prediction
        result = predictor.predict_single(sample_features)
        
        print("🎯 Prediction Result:")
        print(f"Failure Probability: {result['prediction']['failure_probability']}")
        print(f"Risk Level: {result['prediction']['failure_risk']}")
        print(f"Model Used: {result['model_info']['model_used']}")
        print(f"Confidence: {result['model_info']['confidence']}")
        
        print("\n💡 Recommendations:")
        for rec in result['recommendations']:
            print(f"   • {rec}")
        
        # Model comparison
        print("\n📊 Model Comparison:")
        comparison = predictor.compare_models(sample_features)
        for model_name, pred in comparison['model_predictions'].items():
            if 'error' not in pred:
                print(f"   {model_name}: {pred['failure_probability']:.4f} (confidence: {pred.get('confidence', 'N/A')})")
        
        # Feature importance
        print("\n🔍 Top 10 Important Features:")
        importance = predictor.get_feature_importance()
        for i, (feature, score) in enumerate(list(importance.items())[:10]):
            print(f"   {i+1:2d}. {feature}: {score:.4f}")
        
    except Exception as e:
        logger.error(f"❌ Example failed: {str(e)}")

if __name__ == "__main__":
    main()