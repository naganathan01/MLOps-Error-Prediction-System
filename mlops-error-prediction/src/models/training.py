"""
Model training pipeline for error prediction.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb

import mlflow
import mlflow.sklearn
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    def __init__(self, data_dir="data", model_dir="models"):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        # Initialize MLflow
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment("error_prediction")
        
        self.models = {}
        self.scalers = {}
        self.feature_columns = None
        
    def load_data(self):
        """Load processed features"""
        print("📂 Loading processed data...")
        
        features_file = self.data_dir / "processed" / "features.csv"
        if not features_file.exists():
            raise FileNotFoundError(f"Features file not found: {features_file}")
        
        self.df = pd.read_csv(features_file)
        print(f"✅ Loaded {len(self.df)} samples with {len(self.df.columns)} features")
        
        # Prepare features and target
        target_col = 'failure_within_hour'
        exclude_cols = ['timestamp', 'system_state', target_col]
        
        self.feature_columns = [col for col in self.df.columns if col not in exclude_cols]
        self.X = self.df[self.feature_columns]
        self.y = self.df[target_col]
        
        print(f"🎯 Target distribution:")
        print(f"   No failure: {(self.y == 0).sum()} ({(self.y == 0).mean():.2%})")
        print(f"   Failure: {(self.y == 1).sum()} ({(self.y == 1).mean():.2%})")
        
    def prepare_data(self, test_size=0.2, random_state=42):
        """Split and scale data"""
        print("🔄 Preparing data for training...")
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, stratify=self.y
        )
        
        # Scale features
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"📊 Data split:")
        print(f"   Training: {len(self.X_train)} samples")
        print(f"   Testing: {len(self.X_test)} samples")
        
    def train_random_forest(self):
        """Train Random Forest model"""
        print("🌲 Training Random Forest model...")
        
        with mlflow.start_run(run_name="RandomForest"):
            # Hyperparameter tuning
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
            
            rf = RandomForestClassifier(random_state=42, class_weight='balanced')
            grid_search = GridSearchCV(
                rf, param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=1
            )
            
            grid_search.fit(self.X_train, self.y_train)
            
            # Best model
            best_rf = grid_search.best_estimator_
            
            # Predictions
            y_pred = best_rf.predict(self.X_test)
            y_pred_proba = best_rf.predict_proba(self.X_test)[:, 1]
            
            # Metrics
            auc_score = roc_auc_score(self.y_test, y_pred_proba)
            
            # Log parameters and metrics
            mlflow.log_params(grid_search.best_params_)
            mlflow.log_metric("auc_score", auc_score)
            mlflow.log_metric("accuracy", best_rf.score(self.X_test, self.y_test))
            
            # Log model
            mlflow.sklearn.log_model(best_rf, "random_forest_model")
            
            self.models['random_forest'] = best_rf
            
            print(f"✅ Random Forest trained - AUC: {auc_score:.4f}")
            print(f"   Best params: {grid_search.best_params_}")
            
            return best_rf, auc_score
    
    def train_xgboost(self):
        """Train XGBoost model"""
        print("🚀 Training XGBoost model...")
        
        with mlflow.start_run(run_name="XGBoost"):
            # Hyperparameter tuning
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [3, 6, 10],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 1.0]
            }
            
            # Calculate scale_pos_weight for imbalanced data
            scale_pos_weight = (self.y_train == 0).sum() / (self.y_train == 1).sum()
            
            xgb_model = xgb.XGBClassifier(
                random_state=42,
                scale_pos_weight=scale_pos_weight,
                eval_metric='logloss'
            )
            
            grid_search = GridSearchCV(
                xgb_model, param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=1
            )
            
            grid_search.fit(self.X_train, self.y_train)
            
            # Best model
            best_xgb = grid_search.best_estimator_
            
            # Predictions
            y_pred = best_xgb.predict(self.X_test)
            y_pred_proba = best_xgb.predict_proba(self.X_test)[:, 1]
            
            # Metrics
            auc_score = roc_auc_score(self.y_test, y_pred_proba)
            
            # Log parameters and metrics
            mlflow.log_params(grid_search.best_params_)
            mlflow.log_metric("auc_score", auc_score)
            mlflow.log_metric("accuracy", best_xgb.score(self.X_test, self.y_test))
            
            # Log model
            mlflow.sklearn.log_model(best_xgb, "xgboost_model")
            
            self.models['xgboost'] = best_xgb
            
            print(f"✅ XGBoost trained - AUC: {auc_score:.4f}")
            print(f"   Best params: {grid_search.best_params_}")
            
            return best_xgb, auc_score
    
    def train_logistic_regression(self):
        """Train Logistic Regression model"""
        print("📈 Training Logistic Regression model...")
        
        with mlflow.start_run(run_name="LogisticRegression"):
            # Hyperparameter tuning
            param_grid = {
                'C': [0.1, 1.0, 10.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            }
            
            lr = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
            grid_search = GridSearchCV(
                lr, param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=1
            )
            
            grid_search.fit(self.X_train_scaled, self.y_train)
            
            # Best model
            best_lr = grid_search.best_estimator_
            
            # Predictions
            y_pred = best_lr.predict(self.X_test_scaled)
            y_pred_proba = best_lr.predict_proba(self.X_test_scaled)[:, 1]
            
            # Metrics
            auc_score = roc_auc_score(self.y_test, y_pred_proba)
            
            # Log parameters and metrics
            mlflow.log_params(grid_search.best_params_)
            mlflow.log_metric("auc_score", auc_score)
            mlflow.log_metric("accuracy", best_lr.score(self.X_test_scaled, self.y_test))
            
            # Log model
            mlflow.sklearn.log_model(best_lr, "logistic_regression_model")
            
            self.models['logistic_regression'] = best_lr
            
            print(f"✅ Logistic Regression trained - AUC: {auc_score:.4f}")
            print(f"   Best params: {grid_search.best_params_}")
            
            return best_lr, auc_score
    
    def evaluate_models(self):
        """Evaluate all trained models"""
        print("📊 Evaluating all models...")
        
        results = {}
        
        for name, model in self.models.items():
            print(f"\n🔍 Evaluating {name}...")
            
            # Use scaled data for logistic regression
            if name == 'logistic_regression':
                X_test_eval = self.X_test_scaled
            else:
                X_test_eval = self.X_test
            
            # Predictions
            y_pred = model.predict(X_test_eval)
            y_pred_proba = model.predict_proba(X_test_eval)[:, 1]
            
            # Metrics
            auc_score = roc_auc_score(self.y_test, y_pred_proba)
            accuracy = model.score(X_test_eval, self.y_test)
            
            # Classification report
            report = classification_report(self.y_test, y_pred, output_dict=True)
            
            results[name] = {
                'auc_score': auc_score,
                'accuracy': accuracy,
                'precision': report['1']['precision'],
                'recall': report['1']['recall'],
                'f1_score': report['1']['f1-score']
            }
            
            print(f"   AUC: {auc_score:.4f}")
            print(f"   Accuracy: {accuracy:.4f}")
            print(f"   Precision: {report['1']['precision']:.4f}")
            print(f"   Recall: {report['1']['recall']:.4f}")
            print(f"   F1-Score: {report['1']['f1-score']:.4f}")
        
        # Find best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['auc_score'])
        best_model = self.models[best_model_name]
        
        print(f"\n🏆 Best model: {best_model_name} (AUC: {results[best_model_name]['auc_score']:.4f})")
        
        return results, best_model_name, best_model
    
    def save_models(self):
        """Save trained models and metadata"""
        print("💾 Saving models...")
        
        # Save each model
        for name, model in self.models.items():
            model_file = self.model_dir / f"{name}_model.joblib"
            joblib.dump(model, model_file)
            print(f"   Saved {name} to {model_file}")
        
        # Save scaler
        scaler_file = self.model_dir / "scaler.joblib"
        joblib.dump(self.scaler, scaler_file)
        
        # Save feature columns
        features_file = self.model_dir / "feature_columns.json"
        with open(features_file, 'w') as f:
            json.dump(self.feature_columns, f, indent=2)
        
        # Save metadata
        metadata = {
            'training_date': datetime.now().isoformat(),
            'total_samples': len(self.df),
            'training_samples': len(self.X_train),
            'testing_samples': len(self.X_test),
            'feature_count': len(self.feature_columns),
            'target_distribution': {
                'no_failure': int((self.y == 0).sum()),
                'failure': int((self.y == 1).sum())
            }
        }
        
        metadata_file = self.model_dir / "training_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ All models and metadata saved to {self.model_dir}")
    
    def get_feature_importance(self, model_name='random_forest'):
        """Get feature importance from tree-based models"""
        if model_name not in self.models:
            print(f"Model {model_name} not found")
            return None
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\n🔥 Top 10 features for {model_name}:")
            print(importance_df.head(10).to_string(index=False))
            
            # Save feature importance
            importance_file = self.model_dir / f"{model_name}_feature_importance.csv"
            importance_df.to_csv(importance_file, index=False)
            
            return importance_df
        else:
            print(f"Model {model_name} doesn't have feature_importances_ attribute")
            return None
    
    def train_all_models(self):
        """Train all models in the pipeline"""
        print("🚀 Starting model training pipeline...")
        
        # Load and prepare data
        self.load_data()
        self.prepare_data()
        
        # Train models
        print("\n" + "="*50)
        self.train_random_forest()
        
        print("\n" + "="*50)
        self.train_xgboost()
        
        print("\n" + "="*50)
        self.train_logistic_regression()
        
        # Evaluate models
        print("\n" + "="*50)
        results, best_model_name, best_model = self.evaluate_models()
        
        # Get feature importance
        print("\n" + "="*50)
        self.get_feature_importance('random_forest')
        self.get_feature_importance('xgboost')
        
        # Save models
        print("\n" + "="*50)
        self.save_models()
        
        print(f"\n🎉 Training pipeline completed!")
        print(f"🏆 Best model: {best_model_name}")
        
        return results, best_model_name, best_model

if __name__ == "__main__":
    trainer = ModelTrainer()
    results, best_model_name, best_model = trainer.train_all_models()