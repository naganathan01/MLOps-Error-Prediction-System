"""
Complete setup script that creates all files for MLOps Error Prediction System.
Save this as: complete_setup.py
Run with: python complete_setup.py
"""

import os
from pathlib import Path

def create_file(file_path, content):
    """Create a file with given content"""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"   Created: {file_path}")

def create_all_files():
    """Create all project files"""
    print("📁 Creating all project files...")
    
    # 1. requirements.txt
    requirements_content = """# Core ML and Data Science
pandas==2.1.4
numpy==1.24.4
scikit-learn==1.3.2
xgboost==2.0.3
joblib==1.3.2

# API Framework
fastapi==0.104.1
uvicorn==0.24.0
pydantic==2.5.2

# Data Processing
matplotlib==3.8.2
seaborn==0.13.0

# Testing
requests==2.31.0
pytest==7.4.4

# Utilities
python-dotenv==1.0.0
"""
    create_file("requirements.txt", requirements_content)
    
    # 2. src/data/data_generator.py
    data_generator_content = '''"""
Enhanced data generator for MLOps Error Prediction System.
File: src/data/data_generator.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedDataGenerator:
    def __init__(self, output_dir="data/raw"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_realistic_data(self, n_days=30, samples_per_hour=12):
        """Generate realistic system metrics with proper failure patterns"""
        logger.info(f"🔄 Generating {n_days} days of realistic system data...")
        
        total_samples = n_days * 24 * samples_per_hour
        start_time = datetime.now() - timedelta(days=n_days)
        
        data = []
        
        for i in range(total_samples):
            timestamp = start_time + timedelta(minutes=i*5)
            hour = timestamp.hour
            day_of_week = timestamp.weekday()
            
            # Time-based load factors
            business_hours = 1.3 if 9 <= hour <= 17 and day_of_week < 5 else 1.0
            weekend_factor = 0.7 if day_of_week >= 5 else 1.0
            night_factor = 0.5 if hour < 6 or hour > 22 else 1.0
            
            base_load = business_hours * weekend_factor * night_factor
            
            # Generate base metrics with realistic correlations
            cpu_base = np.random.normal(40, 15) * base_load
            memory_base = np.random.normal(50, 20) * base_load
            
            # Add realistic correlations
            if cpu_base > 80:
                memory_base += np.random.normal(10, 5)  # High CPU often means high memory
            
            # Generate failure scenarios (15% of time)
            is_failure_scenario = np.random.random() < 0.15
            failure_within_hour = 0
            
            if is_failure_scenario:
                # Create realistic failure patterns
                scenario_type = np.random.choice(['cpu_spike', 'memory_leak', 'cascade_failure'])
                
                if scenario_type == 'cpu_spike':
                    cpu_usage = np.random.uniform(85, 98)
                    memory_usage = memory_base + np.random.uniform(0, 20)
                    error_count = np.random.poisson(8)
                    response_time = np.random.uniform(1500, 4000)
                    failure_within_hour = 1 if np.random.random() < 0.8 else 0
                    
                elif scenario_type == 'memory_leak':
                    cpu_usage = cpu_base + np.random.uniform(0, 15)
                    memory_usage = np.random.uniform(88, 97)
                    error_count = np.random.poisson(5)
                    response_time = np.random.uniform(800, 2500)
                    failure_within_hour = 1 if np.random.random() < 0.85 else 0
                    
                else:  # cascade_failure
                    cpu_usage = np.random.uniform(80, 95)
                    memory_usage = np.random.uniform(85, 95)
                    error_count = np.random.poisson(12)
                    response_time = np.random.uniform(2000, 5000)
                    failure_within_hour = 1 if np.random.random() < 0.9 else 0
            else:
                # Normal operation
                cpu_usage = np.clip(cpu_base, 5, 75)
                memory_usage = np.clip(memory_base, 10, 80)
                error_count = np.random.poisson(2)
                response_time = np.random.normal(300, 100)
                failure_within_hour = 1 if np.random.random() < 0.02 else 0  # 2% background failure rate
            
            # Generate other metrics with realistic relationships
            disk_usage = np.clip(np.random.normal(45, 20), 10, 95)
            network_latency = np.random.exponential(50) + 20
            active_connections = np.random.poisson(50 * base_load)
            
            # Ensure response time is realistic
            response_time = max(100, response_time + network_latency * 0.3)
            if cpu_usage > 80:
                response_time *= 1.5
            if memory_usage > 85:
                response_time *= 1.3
            
            record = {
                'timestamp': timestamp,
                'cpu_usage': round(np.clip(cpu_usage, 0, 100), 2),
                'memory_usage': round(np.clip(memory_usage, 0, 100), 2),
                'disk_usage': round(disk_usage, 2),
                'network_latency_ms': round(network_latency, 2),
                'error_count': max(0, int(error_count)),
                'response_time_ms': round(max(50, response_time), 2),
                'active_connections': max(1, int(active_connections)),
                'hour': hour,
                'day_of_week': day_of_week,
                'is_weekend': int(day_of_week >= 5),
                'is_business_hours': int(9 <= hour <= 17 and day_of_week < 5),
                'failure_within_hour': failure_within_hour
            }
            
            data.append(record)
        
        df = pd.DataFrame(data)
        
        # Save data
        output_file = self.output_dir / "system_metrics.csv"
        df.to_csv(output_file, index=False)
        
        failure_rate = df['failure_within_hour'].mean()
        logger.info(f"✅ Generated {len(df)} records with {failure_rate:.1%} failure rate")
        logger.info(f"💾 Saved to {output_file}")
        
        return df

def main():
    """Main function for standalone execution"""
    logger.info("🚀 Starting data generation...")
    
    generator = EnhancedDataGenerator()
    df = generator.generate_realistic_data(n_days=30)
    
    logger.info("🎉 Data generation completed successfully!")
    return df

if __name__ == "__main__":
    main()
'''
    create_file("src/data/data_generator.py", data_generator_content)
    
    # 3. src/features/feature_engineering.py
    feature_engineering_content = '''"""
Feature engineering for MLOps Error Prediction System.
File: src/features/feature_engineering.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        
    def create_features(self):
        """Create comprehensive features for ML"""
        logger.info("🔧 Creating features...")
        
        # Load data
        input_file = self.data_dir / "raw" / "system_metrics.csv"
        if not input_file.exists():
            raise FileNotFoundError(f"Data file not found: {input_file}")
            
        df = pd.read_csv(input_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Basic features already exist
        logger.info(f"   Starting with {len(df.columns)} basic features")
        
        # Create interaction features
        df['cpu_memory_product'] = df['cpu_usage'] * df['memory_usage']
        df['resource_pressure'] = (df['cpu_usage'] + df['memory_usage'] + df['disk_usage']) / 3
        df['performance_ratio'] = df['response_time_ms'] / (df['active_connections'] + 1)
        df['error_rate'] = df['error_count'] / (df['active_connections'] + 1)
        
        # Create stress indicators
        df['cpu_high'] = (df['cpu_usage'] > 80).astype(int)
        df['memory_high'] = (df['memory_usage'] > 85).astype(int)
        df['response_slow'] = (df['response_time_ms'] > 1000).astype(int)
        df['errors_high'] = (df['error_count'] > 5).astype(int)
        
        df['total_stress'] = df['cpu_high'] + df['memory_high'] + df['response_slow'] + df['errors_high']
        
        # Create rolling features (simplified)
        for window in [3, 5]:
            for col in ['cpu_usage', 'memory_usage', 'error_count', 'response_time_ms']:
                df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window=window, min_periods=1).mean()
                df[f'{col}_rolling_max_{window}'] = df[col].rolling(window=window, min_periods=1).max()
        
        # Create lag features
        for lag in [1, 2]:
            for col in ['cpu_usage', 'memory_usage', 'error_count']:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag).fillna(df[col].mean())
        
        # Time features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Fill any NaN values
        df = df.fillna(method='forward').fillna(0)
        
        # Save processed features
        processed_dir = self.data_dir / "processed"
        processed_dir.mkdir(exist_ok=True)
        
        output_file = processed_dir / "features.csv"
        df.to_csv(output_file, index=False)
        
        logger.info(f"✅ Created {len(df.columns)} total features")
        logger.info(f"💾 Saved to {output_file}")
        
        return df

def main():
    """Main function for standalone execution"""
    logger.info("🚀 Starting feature engineering...")
    
    engineer = FeatureEngineer()
    df = engineer.create_features()
    
    logger.info("🎉 Feature engineering completed successfully!")
    return df

if __name__ == "__main__":
    main()
'''
    create_file("src/features/feature_engineering.py", feature_engineering_content)
    
    # 4. src/models/training.py
    training_content = '''"""
Model training for MLOps Error Prediction System.
File: src/models/training.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Optional XGBoost support
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, data_dir="data", model_dir="models"):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        self.models = {}
        self.scaler = None
        self.feature_columns = []
        
    def prepare_data(self):
        """Prepare data for training"""
        logger.info("📂 Loading and preparing data...")
        
        # Load features
        features_file = self.data_dir / "processed" / "features.csv"
        if not features_file.exists():
            raise FileNotFoundError(f"Features file not found: {features_file}")
            
        df = pd.read_csv(features_file)
        
        # Define target and features
        target_col = 'failure_within_hour'
        exclude_cols = ['timestamp', target_col]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        self.feature_columns = feature_cols
        
        X = df[feature_cols]
        y = df[target_col]
        
        # Handle any remaining issues
        X = X.fillna(0)
        X = X.replace([np.inf, -np.inf], 0)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"✅ Data prepared:")
        logger.info(f"   Features: {len(feature_cols)}")
        logger.info(f"   Training samples: {len(self.X_train)}")
        logger.info(f"   Testing samples: {len(self.X_test)}")
        logger.info(f"   Failure rate (train): {self.y_train.mean():.2%}")
        logger.info(f"   Failure rate (test): {self.y_test.mean():.2%}")
        
    def train_random_forest(self):
        """Train Random Forest model"""
        logger.info("🌲 Training Random Forest...")
        
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        rf.fit(self.X_train, self.y_train)
        auc = self._evaluate_model(rf, 'Random Forest')
        
        self.models['random_forest'] = rf
        return rf, auc
    
    def train_logistic_regression(self):
        """Train Logistic Regression with proper scaling"""
        logger.info("📊 Training Logistic Regression...")
        
        # Scale features for logistic regression
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(self.X_train)
        X_test_scaled = self.scaler.transform(self.X_test)
        
        lr = LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        )
        
        lr.fit(X_train_scaled, self.y_train)
        
        # Evaluate on scaled test data
        y_pred = lr.predict(X_test_scaled)
        y_pred_proba = lr.predict_proba(X_test_scaled)[:, 1]
        
        auc = roc_auc_score(self.y_test, y_pred_proba)
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, zero_division=0)
        recall = recall_score(self.y_test, y_pred, zero_division=0)
        f1 = f1_score(self.y_test, y_pred, zero_division=0)
        
        logger.info(f"   Logistic Regression Results:")
        logger.info(f"   AUC: {auc:.4f}")
        logger.info(f"   Accuracy: {accuracy:.4f}")
        logger.info(f"   Precision: {precision:.4f}")
        logger.info(f"   Recall: {recall:.4f}")
        logger.info(f"   F1-Score: {f1:.4f}")
        
        self.models['logistic_regression'] = lr
        return lr, auc
    
    def train_xgboost(self):
        """Train XGBoost model if available"""
        if not XGBOOST_AVAILABLE:
            logger.warning("⚠️ XGBoost not available - skipping")
            return None, 0.0
        
        logger.info("🚀 Training XGBoost...")
        
        # Calculate scale_pos_weight for class imbalance
        scale_pos_weight = (self.y_train == 0).sum() / (self.y_train == 1).sum()
        
        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric='logloss',
            use_label_encoder=False
        )
        
        xgb_model.fit(self.X_train, self.y_train)
        auc = self._evaluate_model(xgb_model, 'XGBoost')
        
        self.models['xgboost'] = xgb_model
        return xgb_model, auc
    
    def _evaluate_model(self, model, model_name):
        """Evaluate model performance"""
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        
        auc = roc_auc_score(self.y_test, y_pred_proba)
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, zero_division=0)
        recall = recall_score(self.y_test, y_pred, zero_division=0)
        f1 = f1_score(self.y_test, y_pred, zero_division=0)
        
        logger.info(f"   {model_name} Results:")
        logger.info(f"   AUC: {auc:.4f}")
        logger.info(f"   Accuracy: {accuracy:.4f}")
        logger.info(f"   Precision: {precision:.4f}")
        logger.info(f"   Recall: {recall:.4f}")
        logger.info(f"   F1-Score: {f1:.4f}")
        
        return auc
    
    def train_all_models(self):
        """Train all models"""
        self.prepare_data()
        
        results = {}
        
        # Train Random Forest
        rf_model, rf_auc = self.train_random_forest()
        results['random_forest'] = {
            'model': rf_model,
            'auc_score': rf_auc
        }
        
        # Train Logistic Regression
        lr_model, lr_auc = self.train_logistic_regression()
        results['logistic_regression'] = {
            'model': lr_model,
            'auc_score': lr_auc
        }
        
        # Train XGBoost if available
        if XGBOOST_AVAILABLE:
            xgb_model, xgb_auc = self.train_xgboost()
            if xgb_model is not None:
                results['xgboost'] = {
                    'model': xgb_model,
                    'auc_score': xgb_auc
                }
        
        # Find best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['auc_score'])
        best_model = results[best_model_name]['model']
        
        logger.info(f"\\n🏆 Best Model: {best_model_name} (AUC: {results[best_model_name]['auc_score']:.4f})")
        
        # Save models
        self.save_models()
        
        return results, best_model_name, best_model
    
    def save_models(self):
        """Save trained models and artifacts"""
        logger.info("💾 Saving models...")
        
        # Save models
        for name, model in self.models.items():
            model_path = self.model_dir / f"{name}_model.joblib"
            joblib.dump(model, model_path)
            logger.info(f"   Saved {name} to {model_path}")
        
        # Save scaler
        if self.scaler:
            scaler_path = self.model_dir / "scaler.joblib"
            joblib.dump(self.scaler, scaler_path)
            logger.info(f"   Saved scaler to {scaler_path}")
        
        # Save feature columns
        features_path = self.model_dir / "feature_columns.json"
        with open(features_path, 'w') as f:
            json.dump(self.feature_columns, f)
        logger.info(f"   Saved feature columns to {features_path}")
        
        # Save metadata
        metadata = {
            'training_date': datetime.now().isoformat(),
            'feature_count': len(self.feature_columns),
            'models_trained': list(self.models.keys())
        }
        
        metadata_path = self.model_dir / "training_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"   Saved metadata to {metadata_path}")

def main():
    """Main function for standalone execution"""
    logger.info("🚀 Starting model training...")
    
    trainer = ModelTrainer()
    results, best_model_name, best_model = trainer.train_all_models()
    
    logger.info("🎉 Model training completed successfully!")
    return results, best_model_name, best_model

if __name__ == "__main__":
    main()
'''
    create_file("src/models/training.py", training_content)
    
    # 5. src/models/prediction.py
    prediction_content = '''"""
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
'''
    create_file("src/models/prediction.py", prediction_content)
    
    # 6. src/api/app.py
    api_content = '''"""
FastAPI application for MLOps Error Prediction System.
File: src/api/app.py
"""

from fastapi import FastAPI, HTTPException
from datetime import datetime
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.prediction import PredictionEngine, SystemMetrics, PredictionResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="MLOps Error Prediction API",
    description="Predict system failures before they happen",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Global prediction engine
prediction_engine = None

@app.on_event("startup")
async def startup_event():
    """Initialize the prediction engine on startup"""
    global prediction_engine
    try:
        prediction_engine = PredictionEngine()
        logger.info("✅ Prediction engine initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize prediction engine: {str(e)}")
        raise

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "MLOps Error Prediction API",
        "version": "2.0.0",
        "status": "running",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    global prediction_engine
    
    return {
        "status": "healthy" if prediction_engine else "unhealthy",
        "models_loaded": list(prediction_engine.models.keys()) if prediction_engine else [],
        "feature_count": len(prediction_engine.feature_columns) if prediction_engine else 0,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_failure(metrics: SystemMetrics):
    """Predict system failure probability"""
    global prediction_engine
    
    if not prediction_engine:
        raise HTTPException(status_code=500, detail="Prediction engine not initialized")
    
    try:
        prediction = prediction_engine.predict(metrics)
        logger.info(f"Prediction made: {prediction.failure_probability:.4f} ({prediction.failure_risk})")
        return prediction
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/models/info")
async def get_model_info():
    """Get information about loaded models"""
    global prediction_engine
    
    if not prediction_engine:
        raise HTTPException(status_code=500, detail="Prediction engine not initialized")
    
    return {
        "models_loaded": list(prediction_engine.models.keys()),
        "feature_count": len(prediction_engine.feature_columns),
        "features": prediction_engine.feature_columns[:20],  # First 20 features
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict/batch")
async def predict_batch(metrics_list: list[SystemMetrics]):
    """Batch prediction for multiple metrics"""
    global prediction_engine
    
    if not prediction_engine:
        raise HTTPException(status_code=500, detail="Prediction engine not initialized")
    
    try:
        predictions = []
        for metrics in metrics_list:
            prediction = prediction_engine.predict(metrics)
            predictions.append(prediction)
        
        return {
            "predictions": predictions,
            "count": len(predictions),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

# Error handlers
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
'''
    create_file("src/api/app.py", api_content)
    
    # 7. main.py
    main_content = '''"""
Main entry point for MLOps Error Prediction System.
File: main.py
"""

import sys
import logging
from pathlib import Path

# Add src to Python path
sys.path.append(str(Path(__file__).parent))

from src.api.app import app

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    import uvicorn
    
    logger.info("🚀 Starting MLOps Error Prediction API")
    logger.info("🌐 API will be available at: http://localhost:8000")
    logger.info("📖 Documentation at: http://localhost:8000/docs")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
'''
    create_file("main.py", main_content)
    
    # 8. scripts/run_pipeline.py
    pipeline_content = '''"""
Complete pipeline runner for MLOps Error Prediction System.
File: scripts/run_pipeline.py
"""

import sys
import logging
from pathlib import Path

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.data_generator import EnhancedDataGenerator
from src.features.feature_engineering import FeatureEngineer
from src.models.training import ModelTrainer
from src.models.prediction import PredictionEngine, SystemMetrics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_complete_pipeline():
    """Run the complete MLOps pipeline"""
    logger.info("🚀 Starting Complete MLOps Error Prediction Pipeline")
    logger.info("=" * 60)
    
    try:
        # Step 1: Generate data
        logger.info("\\n📊 STEP 1: GENERATING DATA")
        logger.info("-" * 30)
        generator = EnhancedDataGenerator()
        generator.generate_realistic_data(n_days=30)
        
        # Step 2: Feature engineering
        logger.info("\\n🔧 STEP 2: FEATURE ENGINEERING")
        logger.info("-" * 30)
        engineer = FeatureEngineer()
        engineer.create_features()
        
        # Step 3: Train models
        logger.info("\\n🤖 STEP 3: TRAINING MODELS")
        logger.info("-" * 30)
        trainer = ModelTrainer()
        results, best_model_name, best_model = trainer.train_all_models()
        
        # Step 4: Test prediction
        logger.info("\\n🧪 STEP 4: TESTING PREDICTIONS")
        logger.info("-" * 30)
        
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
        
        engine = PredictionEngine()
        prediction = engine.predict(test_metrics)
        
        logger.info(f"✅ Test Prediction Results:")
        logger.info(f"   Failure Probability: {prediction.failure_probability:.4f}")
        logger.info(f"   Risk Level: {prediction.failure_risk}")
        logger.info(f"   Confidence: {prediction.confidence:.3f}")
        logger.info(f"   Model Used: {prediction.model_used}")
        logger.info(f"   Recommendations: {len(prediction.recommendations)}")
        for i, rec in enumerate(prediction.recommendations, 1):
            logger.info(f"     {i}. {rec}")
        
        logger.info("\\n" + "=" * 60)
        logger.info("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info(f"🏆 Best Model: {best_model_name}")
        logger.info(f"📊 Test Prediction: {prediction.failure_probability:.4f} ({prediction.failure_risk})")
        
        logger.info("\\n📋 NEXT STEPS:")
        logger.info("1. Start the API server:")
        logger.info("   python main.py")
        logger.info("\\n2. Test the API:")
        logger.info("   curl -X POST http://localhost:8000/predict \\\\")
        logger.info("     -H 'Content-Type: application/json' \\\\")
        logger.info("     -d '{")
        logger.info('       "cpu_usage": 85,')
        logger.info('       "memory_usage": 90,')
        logger.info('       "disk_usage": 45,')
        logger.info('       "network_latency_ms": 120,')
        logger.info('       "error_count": 5,')
        logger.info('       "response_time_ms": 800,')
        logger.info('       "active_connections": 150')
        logger.info("     }'")
        
        logger.info("\\n3. View API documentation:")
        logger.info("   http://localhost:8000/docs")
        
        return True
        
    except Exception as e:
        logger.error(f"\\n❌ Pipeline failed: {str(e)}")
        return False

def main():
    """Main function"""
    success = run_complete_pipeline()
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
    create_file("scripts/run_pipeline.py", pipeline_content)
    
    # 9. tests/test_api.py
    test_content = '''"""
API tests for MLOps Error Prediction System.
File: tests/test_api.py
"""

import requests
import json
import sys
from pathlib import Path

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent))

def test_api():
    """Test the prediction API"""
    print("🧪 Testing MLOps Error Prediction API")
    print("=" * 50)
    
    base_url = "http://localhost:8000"
    
    # Test 1: Health check
    print("\\n1. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            health_data = response.json()
            print("✅ Health check passed")
            print(f"   Status: {health_data['status']}")
            print(f"   Models loaded: {health_data['models_loaded']}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException:
        print("❌ API is not running. Start it first with: python main.py")
        return False
    
    # Test 2: Predictions with different scenarios
    test_scenarios = [
        {
            "name": "Normal System",
            "data": {
                "cpu_usage": 30,
                "memory_usage": 40,
                "disk_usage": 25,
                "network_latency_ms": 50,
                "error_count": 0,
                "response_time_ms": 200,
                "active_connections": 25
            }
        },
        {
            "name": "Critical System State",
            "data": {
                "cpu_usage": 95,
                "memory_usage": 92,
                "disk_usage": 88,
                "network_latency_ms": 300,
                "error_count": 15,
                "response_time_ms": 2500,
                "active_connections": 200
            }
        }
    ]
    
    print("\\n2. Testing predictions...")
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\\n2.{i} {scenario['name']}:")
        try:
            response = requests.post(
                f"{base_url}/predict",
                json=scenario['data'],
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Prediction successful:")
                print(f"   Risk Level: {result['failure_risk']}")
                print(f"   Probability: {result['failure_probability']:.4f}")
                print(f"   Confidence: {result['confidence']:.3f}")
                print(f"   Model Used: {result['model_used']}")
                
            else:
                print(f"❌ Prediction failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Prediction error: {str(e)}")
    
    print("\\n🎉 API testing completed!")
    return True

def main():
    """Main function"""
    try:
        success = test_api()
        if success:
            print("\\n✅ All tests completed successfully!")
        else:
            print("\\n❌ Some tests failed!")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\\n⏹️ Testing interrupted by user")
    except Exception as e:
        print(f"\\n❌ Testing failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
    create_file("tests/test_api.py", test_content)
    
    # 10. __init__.py files
    init_files = [
        ("src/__init__.py", '"""MLOps Error Prediction System"""\\n__version__ = "2.0.0"'),
        ("src/data/__init__.py", '"""Data generation and processing modules"""'),
        ("src/features/__init__.py", '"""Feature engineering modules"""'),
        ("src/models/__init__.py", '"""Model training and prediction modules"""'),
        ("src/api/__init__.py", '"""API modules"""'),
        ("tests/__init__.py", '"""Test modules"""'),
    ]
    
    for file_path, content in init_files:
        create_file(file_path, content)

def run_setup():
    """Run the complete setup"""
    print("🚀 MLOps Error Prediction System - Complete Setup")
    print("=" * 60)
    
    # Create all files
    create_all_files()
    
    print("\n✅ All files created successfully!")
    print("\n📋 Next Steps:")
    print("1. Install dependencies:")
    print("   pip install -r requirements.txt")
    print("\n2. Run the complete pipeline:")
    print("   python scripts/run_pipeline.py")
    print("\n3. Start the API server:")
    print("   python main.py")
    print("\n4. Test the API (in another terminal):")
    print("   python tests/test_api.py")
    print("\n5. View API documentation:")
    print("   http://localhost:8000/docs")

if __name__ == "__main__":
    run_setup()