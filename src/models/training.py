"""
Model training module for the error prediction system.
Trains multiple ML models and selects the best performer.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
import logging
from datetime import datetime
from typing import Dict, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report
)

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not available. Install with: pip install xgboost")

try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logging.warning("MLflow not available. Install with: pip install mlflow")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, data_dir="data", model_dir="models"):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        # Data attributes
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        
        # Model storage
        self.models = {}
        self.results = {}
        
        # MLflow setup
        if MLFLOW_AVAILABLE:
            mlflow.set_tracking_uri("file:./mlruns")
            mlflow.set_experiment("error_prediction")
    
    def load_data(self, file_path: str = None):
        """Load processed data for training"""
        if file_path is None:
            file_path = self.data_dir / "processed" / "features.csv"
        
        logger.info(f"📂 Loading data from {file_path}")
        
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        df = pd.read_csv(file_path)
        logger.info(f"✅ Loaded {len(df)} samples with {len(df.columns)} columns")
        
        return df
    
    def prepare_data(self, df=None, test_size=0.2, random_state=42):
        """Prepare data for training"""
        logger.info("🔧 Preparing data for training...")
        
        if df is None:
            df = self.load_data()
        
        # Define target and feature columns
        target_col = 'failure_within_hour'
        exclude_cols = ['timestamp', 'system_state', target_col]
        
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")
        
        # Prepare features and target
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        self.feature_columns = feature_cols
        
        X = df[feature_cols]
        y = df[target_col]
        
        # Handle any remaining missing values
        X = X.fillna(0)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"✅ Data prepared:")
        logger.info(f"   Features: {len(self.feature_columns)}")
        logger.info(f"   Training samples: {len(self.X_train)}")
        logger.info(f"   Testing samples: {len(self.X_test)}")
        logger.info(f"   Positive class rate: {self.y_train.mean():.2%}")
    
    def train_random_forest(self, use_grid_search=True):
        """Train Random Forest model"""
        logger.info("🌲 Training Random Forest model...")
        
        if use_grid_search:
            # Grid search for best parameters
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'class_weight': ['balanced']
            }
            
            rf = RandomForestClassifier(random_state=42, n_jobs=-1)
            grid_search = GridSearchCV(
                rf, param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=1
            )
            grid_search.fit(self.X_train, self.y_train)
            
            best_rf = grid_search.best_estimator_
            logger.info(f"   Best parameters: {grid_search.best_params_}")
            
        else:
            # Use default parameters with some optimization
            best_rf = RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=2,
                min_samples_leaf=1,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
            best_rf.fit(self.X_train, self.y_train)
        
        # Evaluate model
        y_pred = best_rf.predict(self.X_test)
        y_pred_proba = best_rf.predict_proba(self.X_test)[:, 1]
        
        auc_score = roc_auc_score(self.y_test, y_pred_proba)
        
        self.models['random_forest'] = best_rf
        
        logger.info(f"✅ Random Forest trained with AUC: {auc_score:.4f}")
        
        return best_rf, auc_score
    
    def train_xgboost(self, use_grid_search=True):
        """Train XGBoost model"""
        if not XGBOOST_AVAILABLE:
            logger.error("❌ XGBoost not available")
            return None, 0.0
        
        logger.info("🚀 Training XGBoost model...")
        
        if use_grid_search:
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [3, 6, 10],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 1.0],
                'eval_metric': ['logloss']
            }
            
            xgb_model = xgb.XGBClassifier(random_state=42, n_jobs=-1)
            grid_search = GridSearchCV(
                xgb_model, param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=1
            )
            grid_search.fit(self.X_train, self.y_train)
            
            best_xgb = grid_search.best_estimator_
            logger.info(f"   Best parameters: {grid_search.best_params_}")
            
        else:
            best_xgb = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                eval_metric='logloss',
                random_state=42,
                n_jobs=-1
            )
            best_xgb.fit(self.X_train, self.y_train)
        
        # Evaluate model
        y_pred = best_xgb.predict(self.X_test)
        y_pred_proba = best_xgb.predict_proba(self.X_test)[:, 1]
        
        auc_score = roc_auc_score(self.y_test, y_pred_proba)
        
        self.models['xgboost'] = best_xgb
        
        logger.info(f"✅ XGBoost trained with AUC: {auc_score:.4f}")
        
        return best_xgb, auc_score
    
    def train_logistic_regression(self, use_grid_search=True):
        """Train Logistic Regression model"""
        logger.info("📈 Training Logistic Regression model...")
        
        # Scale features for logistic regression
        X_train_scaled = self.scaler.fit_transform(self.X_train)
        X_test_scaled = self.scaler.transform(self.X_test)
        
        if use_grid_search:
            param_grid = {
                'C': [0.1, 1.0, 10.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear'],
                'class_weight': ['balanced'],
                'max_iter': [1000]
            }
            
            lr = LogisticRegression(random_state=42)
            grid_search = GridSearchCV(
                lr, param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=1
            )
            grid_search.fit(X_train_scaled, self.y_train)
            
            best_lr = grid_search.best_estimator_
            logger.info(f"   Best parameters: {grid_search.best_params_}")
            
        else:
            best_lr = LogisticRegression(
                C=1.0,
                penalty='l2',
                solver='liblinear',
                class_weight='balanced',
                max_iter=1000,
                random_state=42
            )
            best_lr.fit(X_train_scaled, self.y_train)
        
        # Evaluate model
        y_pred = best_lr.predict(X_test_scaled)
        y_pred_proba = best_lr.predict_proba(X_test_scaled)[:, 1]
        
        auc_score = roc_auc_score(self.y_test, y_pred_proba)
        
        self.models['logistic_regression'] = best_lr
        
        logger.info(f"✅ Logistic Regression trained with AUC: {auc_score:.4f}")
        
        return best_lr, auc_score
    
    def evaluate_models(self):
        """Evaluate all trained models"""
        logger.info("📊 Evaluating all models...")
        
        results = {}
        
        for model_name, model in self.models.items():
            logger.info(f"   Evaluating {model_name}...")
            
            # Prepare test data
            if model_name == 'logistic_regression':
                X_test_model = self.scaler.transform(self.X_test)
            else:
                X_test_model = self.X_test
            
            # Make predictions
            y_pred = model.predict(X_test_model)
            y_pred_proba = model.predict_proba(X_test_model)[:, 1]
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(self.y_test, y_pred),
                'precision': precision_score(self.y_test, y_pred),
                'recall': recall_score(self.y_test, y_pred),
                'f1_score': f1_score(self.y_test, y_pred),
                'auc_score': roc_auc_score(self.y_test, y_pred_proba)
            }
            
            results[model_name] = metrics
            
            logger.info(f"     AUC: {metrics['auc_score']:.4f}")
            logger.info(f"     F1-Score: {metrics['f1_score']:.4f}")
        
        self.results = results
        
        # Find best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['auc_score'])
        best_model = self.models[best_model_name]
        
        logger.info(f"🏆 Best model: {best_model_name} (AUC: {results[best_model_name]['auc_score']:.4f})")
        
        return results, best_model_name, best_model
    
    def save_models(self):
        """Save trained models and artifacts"""
        logger.info("💾 Saving models and artifacts...")
        
        # Save individual models
        for model_name, model in self.models.items():
            model_path = self.model_dir / f"{model_name}_model.joblib"
            joblib.dump(model, model_path)
            logger.info(f"   Saved {model_name} to {model_path}")
        
        # Save scaler
        scaler_path = self.model_dir / "scaler.joblib"
        joblib.dump(self.scaler, scaler_path)
        logger.info(f"   Saved scaler to {scaler_path}")
        
        # Save feature columns
        features_path = self.model_dir / "feature_columns.json"
        with open(features_path, 'w') as f:
            json.dump(self.feature_columns, f, indent=2)
        logger.info(f"   Saved feature columns to {features_path}")
        
        # Save training metadata
        metadata = {
            'training_date': datetime.now().isoformat(),
            'total_samples': len(self.X_train) + len(self.X_test),
            'training_samples': len(self.X_train),
            'testing_samples': len(self.X_test),
            'feature_count': len(self.feature_columns),
            'positive_class_rate': float(self.y_train.mean()),
            'models_trained': list(self.models.keys()),
            'results': self.results
        }
        
        metadata_path = self.model_dir / "training_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info(f"   Saved metadata to {metadata_path}")
        
        logger.info("✅ All models and artifacts saved")
    
    def log_to_mlflow(self, model_name: str, model, metrics: Dict[str, float]):
        """Log model and metrics to MLflow"""
        if not MLFLOW_AVAILABLE:
            return
        
        with mlflow.start_run(run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log parameters
            if hasattr(model, 'get_params'):
                mlflow.log_params(model.get_params())
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log model
            mlflow.sklearn.log_model(model, model_name)
            
            # Log feature importance if available
            if hasattr(model, 'feature_importances_'):
                feature_importance = dict(zip(self.feature_columns, model.feature_importances_))
                mlflow.log_dict(feature_importance, f"{model_name}_feature_importance.json")
    
    def train_all_models(self, use_grid_search=False):
        """Train all available models"""
        logger.info("🚀 Starting complete model training pipeline...")
        
        # Prepare data if not already done
        if self.X_train is None:
            self.prepare_data()
        
        training_methods = [
            ('random_forest', self.train_random_forest),
            ('logistic_regression', self.train_logistic_regression)
        ]
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            training_methods.append(('xgboost', self.train_xgboost))
        
        # Train each model
        for model_name, train_method in training_methods:
            try:
                logger.info(f"\n{'='*60}")
                logger.info(f"Training {model_name.upper()}")
                logger.info('='*60)
                
                model, score = train_method(use_grid_search=use_grid_search)
                
                if model is not None and MLFLOW_AVAILABLE:
                    # Get metrics for MLflow
                    if model_name in self.results:
                        self.log_to_mlflow(model_name, model, self.results[model_name])
                
            except Exception as e:
                logger.error(f"❌ Failed to train {model_name}: {str(e)}")
        
        # Evaluate all models
        if self.models:
            results, best_model_name, best_model = self.evaluate_models()
            
            # Save everything
            self.save_models()
            
            return results, best_model_name, best_model
        else:
            logger.error("❌ No models were trained successfully")
            return {}, None, None

def main():
    """Main training function"""
    logger.info("🚀 Starting MLOps Error Prediction Model Training")
    
    try:
        trainer = ModelTrainer()
        results, best_model_name, best_model = trainer.train_all_models()
        
        if results:
            logger.info("\n" + "="*60)
            logger.info("🎉 TRAINING COMPLETED SUCCESSFULLY!")
            logger.info("="*60)
            logger.info(f"🏆 Best Model: {best_model_name}")
            logger.info(f"🏆 Best AUC Score: {results[best_model_name]['auc_score']:.4f}")
            
            logger.info("\n📊 All Model Results:")
            for model_name, metrics in results.items():
                logger.info(f"\n{model_name.upper()}:")
                for metric_name, value in metrics.items():
                    logger.info(f"  {metric_name}: {value:.4f}")
            
            logger.info("\n🎯 Next Steps:")
            logger.info("1. Start the API server: uvicorn src.api.app:app --reload")
            logger.info("2. Test predictions: curl -X POST http://localhost:8000/predict \\")
            logger.info("   -H 'Content-Type: application/json' \\")
            logger.info("   -d '{\"cpu_usage\": 85, \"memory_usage\": 90, \"error_count\": 3}'")
            
        else:
            logger.error("❌ Training failed - no models were trained")
            return False
            
    except Exception as e:
        logger.error(f"❌ Training pipeline failed: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)