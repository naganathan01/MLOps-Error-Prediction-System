"""
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
        
        logger.info(f"\n🏆 Best Model: {best_model_name} (AUC: {results[best_model_name]['auc_score']:.4f})")
        
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
