"""
Model monitoring module for tracking model performance,
data drift, and system health in production.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import joblib
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelMetrics:
    """Data class for model performance metrics"""
    model_name: str
    timestamp: datetime
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    prediction_count: int
    average_confidence: float

@dataclass
class DriftMetrics:
    """Data class for data drift metrics"""
    feature_name: str
    timestamp: datetime
    drift_score: float
    drift_detected: bool
    reference_mean: float
    current_mean: float
    reference_std: float
    current_std: float

class ModelMonitor:
    """Main model monitoring class"""
    
    def __init__(self, db_path: str = "data/monitoring.db", 
                 models_dir: str = "models",
                 monitoring_config: Dict = None):
        self.db_path = Path(db_path)
        self.models_dir = Path(models_dir)
        self.config = monitoring_config or self._get_default_config()
        
        # Create database and tables
        self._init_database()
        
        # Load models and reference data
        self.models = {}
        self.reference_data = {}
        self.feature_columns = []
        self.scaler = None
        
        self._load_models()
        self._load_reference_data()
    
    def _get_default_config(self) -> Dict:
        """Get default monitoring configuration"""
        return {
            'drift_detection': {
                'method': 'ks_test',
                'threshold': 0.05,
                'window_size': 1000,
                'reference_window': 5000
            },
            'performance_monitoring': {
                'min_accuracy': 0.70,
                'min_precision': 0.65,
                'min_recall': 0.60,
                'min_auc': 0.75,
                'evaluation_window': 500
            },
            'alerting': {
                'enable_alerts': True,
                'alert_thresholds': {
                    'performance_degradation': 0.05,
                    'high_drift_score': 0.3,
                    'low_confidence': 0.6
                }
            }
        }
    
    def _init_database(self):
        """Initialize SQLite database for monitoring data"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            # Predictions table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    model_name TEXT NOT NULL,
                    prediction REAL NOT NULL,
                    confidence REAL NOT NULL,
                    actual_outcome INTEGER,
                    features TEXT NOT NULL
                )
            ''')
            
            # Model performance table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    model_name TEXT NOT NULL,
                    accuracy REAL,
                    precision REAL,
                    recall REAL,
                    f1_score REAL,
                    roc_auc REAL,
                    prediction_count INTEGER,
                    average_confidence REAL
                )
            ''')
            
            # Data drift table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS data_drift (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    feature_name TEXT NOT NULL,
                    drift_score REAL NOT NULL,
                    drift_detected BOOLEAN NOT NULL,
                    reference_mean REAL,
                    current_mean REAL,
                    reference_std REAL,
                    current_std REAL
                )
            ''')
            
            # Alerts table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    alert_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    message TEXT NOT NULL,
                    resolved BOOLEAN DEFAULT FALSE,
                    resolved_at DATETIME
                )
            ''')
            
            conn.commit()
        
        logger.info(f"✅ Monitoring database initialized at {self.db_path}")
    
    def _load_models(self):
        """Load trained models"""
        try:
            if not self.models_dir.exists():
                logger.warning(f"Models directory not found: {self.models_dir}")
                return
            
            model_files = {
                'random_forest': self.models_dir / "random_forest_model.joblib",
                'xgboost': self.models_dir / "xgboost_model.joblib",
                'logistic_regression': self.models_dir / "logistic_regression_model.joblib"
            }
            
            for name, file_path in model_files.items():
                if file_path.exists():
                    self.models[name] = joblib.load(file_path)
                    logger.info(f"   Loaded {name} model")
            
            # Load scaler
            scaler_file = self.models_dir / "scaler.joblib"
            if scaler_file.exists():
                self.scaler = joblib.load(scaler_file)
            
            # Load feature columns
            features_file = self.models_dir / "feature_columns.json"
            if features_file.exists():
                with open(features_file, 'r') as f:
                    self.feature_columns = json.load(f)
            
            logger.info(f"✅ Loaded {len(self.models)} models for monitoring")
            
        except Exception as e:
            logger.error(f"Failed to load models: {str(e)}")
    
    def _load_reference_data(self):
        """Load reference data for drift detection"""
        try:
            # Load training data as reference
            data_file = Path("data/processed/features.csv")
            if data_file.exists():
                df = pd.read_csv(data_file)
                
                # Prepare reference data (same as training data preparation)
                exclude_cols = ['timestamp', 'system_state', 'failure_within_hour']
                feature_cols = [col for col in df.columns if col not in exclude_cols]
                
                self.reference_data = df[feature_cols].fillna(0)
                logger.info(f"✅ Loaded reference data: {len(self.reference_data)} samples")
                
        except Exception as e:
            logger.error(f"Failed to load reference data: {str(e)}")
    
    def log_prediction(self, model_name: str, prediction: float, confidence: float,
                      features: Dict, actual_outcome: Optional[int] = None):
        """Log a prediction for monitoring"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO predictions 
                    (model_name, prediction, confidence, actual_outcome, features)
                    VALUES (?, ?, ?, ?, ?)
                ''', (model_name, prediction, confidence, actual_outcome, json.dumps(features)))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to log prediction: {str(e)}")
    
    def update_actual_outcome(self, prediction_id: int, actual_outcome: int):
        """Update the actual outcome for a logged prediction"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    UPDATE predictions 
                    SET actual_outcome = ? 
                    WHERE id = ?
                ''', (actual_outcome, prediction_id))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to update actual outcome: {str(e)}")
    
    def calculate_model_performance(self, model_name: str, 
                                  lookback_hours: int = 24) -> Optional[ModelMetrics]:
        """Calculate model performance metrics"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=lookback_hours)
            
            with sqlite3.connect(self.db_path) as conn:
                query = '''
                    SELECT prediction, confidence, actual_outcome
                    FROM predictions 
                    WHERE model_name = ? 
                    AND actual_outcome IS NOT NULL
                    AND timestamp > ?
                '''
                
                df = pd.read_sql_query(query, conn, params=(model_name, cutoff_time))
            
            if len(df) == 0:
                logger.warning(f"No predictions with outcomes found for {model_name}")
                return None
            
            # Calculate metrics
            y_true = df['actual_outcome'].values
            y_pred = (df['prediction'] > 0.5).astype(int)
            y_pred_proba = df['prediction'].values
            
            metrics = ModelMetrics(
                model_name=model_name,
                timestamp=datetime.now(),
                accuracy=accuracy_score(y_true, y_pred),
                precision=precision_score(y_true, y_pred, zero_division=0),
                recall=recall_score(y_true, y_pred, zero_division=0),
                f1_score=f1_score(y_true, y_pred, zero_division=0),
                roc_auc=roc_auc_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else 0.0,
                prediction_count=len(df),
                average_confidence=df['confidence'].mean()
            )
            
            # Store metrics
            self._store_performance_metrics(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to calculate performance for {model_name}: {str(e)}")
            return None
    
    def _store_performance_metrics(self, metrics: ModelMetrics):
        """Store performance metrics in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO model_performance 
                    (model_name, accuracy, precision, recall, f1_score, roc_auc, 
                     prediction_count, average_confidence)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    metrics.model_name, metrics.accuracy, metrics.precision,
                    metrics.recall, metrics.f1_score, metrics.roc_auc,
                    metrics.prediction_count, metrics.average_confidence
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to store performance metrics: {str(e)}")
    
    def detect_data_drift(self, current_data: pd.DataFrame, 
                         features: Optional[List[str]] = None) -> List[DriftMetrics]:
        """Detect data drift using statistical tests"""
        if self.reference_data is None or len(self.reference_data) == 0:
            logger.warning("No reference data available for drift detection")
            return []
        
        if features is None:
            features = self.feature_columns[:10]  # Monitor top 10 features
        
        drift_results = []
        
        for feature in features:
            if feature not in self.reference_data.columns or feature not in current_data.columns:
                continue
            
            try:
                # Get reference and current distributions
                reference_values = self.reference_data[feature].dropna()
                current_values = current_data[feature].dropna()
                
                if len(reference_values) == 0 or len(current_values) == 0:
                    continue
                
                # Perform Kolmogorov-Smirnov test
                ks_statistic, p_value = stats.ks_2samp(reference_values, current_values)
                
                # Determine if drift is detected
                drift_threshold = self.config['drift_detection']['threshold']
                drift_detected = p_value < drift_threshold
                
                drift_metrics = DriftMetrics(
                    feature_name=feature,
                    timestamp=datetime.now(),
                    drift_score=ks_statistic,
                    drift_detected=drift_detected,
                    reference_mean=float(reference_values.mean()),
                    current_mean=float(current_values.mean()),
                    reference_std=float(reference_values.std()),
                    current_std=float(current_values.std())
                )
                
                drift_results.append(drift_metrics)
                
                # Store drift metrics
                self._store_drift_metrics(drift_metrics)
                
                if drift_detected:
                    logger.warning(f"Data drift detected in {feature}: KS={ks_statistic:.3f}, p={p_value:.3f}")
                
            except Exception as e:
                logger.error(f"Failed to detect drift for {feature}: {str(e)}")
        
        return drift_results
    
    def _store_drift_metrics(self, metrics: DriftMetrics):
        """Store drift metrics in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO data_drift 
                    (feature_name, drift_score, drift_detected, reference_mean, 
                     current_mean, reference_std, current_std)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    metrics.feature_name, metrics.drift_score, metrics.drift_detected,
                    metrics.reference_mean, metrics.current_mean,
                    metrics.reference_std, metrics.current_std
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to store drift metrics: {str(e)}")
    
    def check_alerts(self) -> List[Dict]:
        """Check for alert conditions and create alerts"""
        alerts = []
        
        try:
            # Check model performance alerts
            for model_name in self.models.keys():
                metrics = self.calculate_model_performance(model_name, lookback_hours=24)
                
                if metrics:
                    perf_config = self.config['performance_monitoring']
                    
                    # Check accuracy
                    if metrics.accuracy < perf_config['min_accuracy']:
                        alert = self._create_alert(
                            'performance_degradation',
                            'HIGH',
                            f"Model {model_name} accuracy below threshold: {metrics.accuracy:.3f}"
                        )
                        alerts.append(alert)
                    
                    # Check AUC
                    if metrics.roc_auc < perf_config['min_auc']:
                        alert = self._create_alert(
                            'performance_degradation',
                            'HIGH',
                            f"Model {model_name} AUC below threshold: {metrics.roc_auc:.3f}"
                        )
                        alerts.append(alert)
                    
                    # Check confidence
                    if metrics.average_confidence < self.config['alerting']['alert_thresholds']['low_confidence']:
                        alert = self._create_alert(
                            'low_confidence',
                            'MEDIUM',
                            f"Model {model_name} average confidence low: {metrics.average_confidence:.3f}"
                        )
                        alerts.append(alert)
            
            # Check drift alerts
            recent_drift = self._get_recent_drift_alerts()
            for drift in recent_drift:
                if drift['drift_detected'] and drift['drift_score'] > self.config['alerting']['alert_thresholds']['high_drift_score']:
                    alert = self._create_alert(
                        'data_drift',
                        'MEDIUM',
                        f"High data drift detected in {drift['feature_name']}: {drift['drift_score']:.3f}"
                    )
                    alerts.append(alert)
            
        except Exception as e:
            logger.error(f"Failed to check alerts: {str(e)}")
        
        return alerts
    
    def _create_alert(self, alert_type: str, severity: str, message: str) -> Dict:
        """Create and store an alert"""
        alert = {
            'timestamp': datetime.now().isoformat(),
            'alert_type': alert_type,
            'severity': severity,
            'message': message,
            'resolved': False
        }
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    INSERT INTO alerts (alert_type, severity, message)
                    VALUES (?, ?, ?)
                ''', (alert_type, severity, message))
                alert['id'] = cursor.lastrowid
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to store alert: {str(e)}")
        
        return alert
    
    def _get_recent_drift_alerts(self, hours: int = 1) -> List[Dict]:
        """Get recent drift detection results"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            with sqlite3.connect(self.db_path) as conn:
                query = '''
                    SELECT feature_name, drift_score, drift_detected
                    FROM data_drift 
                    WHERE timestamp > ?
                    ORDER BY timestamp DESC
                '''
                
                df = pd.read_sql_query(query, conn, params=(cutoff_time,))
                return df.to_dict('records')
                
        except Exception as e:
            logger.error(f"Failed to get recent drift alerts: {str(e)}")
            return []
    
    def generate_monitoring_report(self, days: int = 7) -> Dict:
        """Generate comprehensive monitoring report"""
        cutoff_time = datetime.now() - timedelta(days=days)
        
        report = {
            'report_period': f"{days} days",
            'generated_at': datetime.now().isoformat(),
            'model_performance': {},
            'data_drift_summary': {},
            'alert_summary': {},
            'recommendations': []
        }
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Model performance summary
                perf_query = '''
                    SELECT model_name, AVG(accuracy) as avg_accuracy, 
                           AVG(precision) as avg_precision, AVG(recall) as avg_recall,
                           AVG(roc_auc) as avg_auc, COUNT(*) as measurements
                    FROM model_performance 
                    WHERE timestamp > ?
                    GROUP BY model_name
                '''
                
                perf_df = pd.read_sql_query(perf_query, conn, params=(cutoff_time,))
                report['model_performance'] = perf_df.to_dict('records')
                
                # Data drift summary
                drift_query = '''
                    SELECT feature_name, AVG(drift_score) as avg_drift_score,
                           SUM(CASE WHEN drift_detected THEN 1 ELSE 0 END) as drift_detections,
                           COUNT(*) as total_measurements
                    FROM data_drift 
                    WHERE timestamp > ?
                    GROUP BY feature_name
                '''
                
                drift_df = pd.read_sql_query(drift_query, conn, params=(cutoff_time,))
                report['data_drift_summary'] = drift_df.to_dict('records')
                
                # Alert summary
                alert_query = '''
                    SELECT alert_type, severity, COUNT(*) as count
                    FROM alerts 
                    WHERE timestamp > ?
                    GROUP BY alert_type, severity
                '''
                
                alert_df = pd.read_sql_query(alert_query, conn, params=(cutoff_time,))
                report['alert_summary'] = alert_df.to_dict('records')
        
        except Exception as e:
            logger.error(f"Failed to generate monitoring report: {str(e)}")
        
        # Generate recommendations
        report['recommendations'] = self._generate_recommendations(report)
        
        return report
    
    def _generate_recommendations(self, report: Dict) -> List[str]:
        """Generate recommendations based on monitoring report"""
        recommendations = []
        
        # Check model performance
        for model_perf in report.get('model_performance', []):
            if model_perf.get('avg_accuracy', 0) < 0.75:
                recommendations.append(f"Consider retraining {model_perf['model_name']} - accuracy is low")
            
            if model_perf.get('avg_auc', 0) < 0.8:
                recommendations.append(f"Review {model_perf['model_name']} features - AUC could be improved")
        
        # Check data drift
        high_drift_features = [
            drift['feature_name'] for drift in report.get('data_drift_summary', [])
            if drift.get('avg_drift_score', 0) > 0.3
        ]
        
        if high_drift_features:
            recommendations.append(f"Investigate data drift in features: {', '.join(high_drift_features)}")
        
        # Check alerts
        alert_counts = {alert['alert_type']: alert['count'] for alert in report.get('alert_summary', [])}
        
        if alert_counts.get('performance_degradation', 0) > 5:
            recommendations.append("High number of performance alerts - consider model retraining")
        
        if alert_counts.get('data_drift', 0) > 10:
            recommendations.append("Frequent data drift detected - review data pipeline")
        
        if not recommendations:
            recommendations.append("System is performing well - continue monitoring")
        
        return recommendations
    
    def visualize_model_performance(self, model_name: str, days: int = 30, 
                                   save_path: Optional[str] = None):
        """Create visualizations for model performance"""
        try:
            cutoff_time = datetime.now() - timedelta(days=days)
            
            with sqlite3.connect(self.db_path) as conn:
                query = '''
                    SELECT timestamp, accuracy, precision, recall, f1_score, roc_auc
                    FROM model_performance 
                    WHERE model_name = ? AND timestamp > ?
                    ORDER BY timestamp
                '''
                
                df = pd.read_sql_query(query, conn, params=(model_name, cutoff_time))
            
            if len(df) == 0:
                logger.warning(f"No performance data found for {model_name}")
                return
            
            # Convert timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'Model Performance Over Time - {model_name}', fontsize=16)
            
            # Plot metrics
            metrics = ['accuracy', 'precision', 'recall', 'roc_auc']
            colors = ['blue', 'green', 'red', 'orange']
            
            for i, (metric, color) in enumerate(zip(metrics, colors)):
                ax = axes[i//2, i%2]
                ax.plot(df['timestamp'], df[metric], color=color, marker='o', linewidth=2)
                ax.set_title(f'{metric.replace("_", " ").title()}')
                ax.set_ylabel(metric.replace("_", " ").title())
                ax.grid(True, alpha=0.3)
                ax.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Performance visualization saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Failed to create performance visualization: {str(e)}")
    
    def run_monitoring_cycle(self):
        """Run a complete monitoring cycle"""
        logger.info("🔍 Starting monitoring cycle...")
        
        try:
            # Calculate performance for all models
            for model_name in self.models.keys():
                metrics = self.calculate_model_performance(model_name)
                if metrics:
                    logger.info(f"   {model_name}: AUC={metrics.roc_auc:.3f}, Accuracy={metrics.accuracy:.3f}")
            
            # Check for alerts
            alerts = self.check_alerts()
            if alerts:
                logger.warning(f"⚠️ {len(alerts)} alerts generated")
                for alert in alerts:
                    logger.warning(f"   {alert['severity']}: {alert['message']}")
            else:
                logger.info("✅ No alerts detected")
            
            logger.info("✅ Monitoring cycle completed")
            
        except Exception as e:
            logger.error(f"❌ Monitoring cycle failed: {str(e)}")

def main():
    """Example usage of ModelMonitor"""
    monitor = ModelMonitor()
    
    # Run monitoring cycle
    monitor.run_monitoring_cycle()
    
    # Generate report
    report = monitor.generate_monitoring_report(days=7)
    
    print("\n📊 Monitoring Report Summary:")
    print(f"Report period: {report['report_period']}")
    print(f"Models monitored: {len(report['model_performance'])}")
    print(f"Features with drift monitoring: {len(report['data_drift_summary'])}")
    print(f"Total alerts: {sum(alert['count'] for alert in report['alert_summary'])}")
    
    print("\n💡 Recommendations:")
    for rec in report['recommendations']:
        print(f"   • {rec}")

if __name__ == "__main__":
    main()