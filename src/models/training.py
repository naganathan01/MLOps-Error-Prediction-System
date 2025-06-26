"""
Enhanced model training with better algorithms, hyperparameter tuning, and ensemble methods.
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

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectFromModel, RFE

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedModelTrainer:
    def __init__(self, data_dir="data", model_dir="models"):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        # Data attributes
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scalers = {}
        self.feature_columns = []
        self.selected_features = None
        
        # Model storage
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        
        # MLflow setup
        if MLFLOW_AVAILABLE:
            mlflow.set_tracking_uri("file:./mlruns")
            mlflow.set_experiment("enhanced_error_prediction")
    
    def load_data(self, file_path: str = None):
        """Load enhanced processed data for training"""
        if file_path is None:
            file_path = self.data_dir / "processed" / "enhanced_features.csv"
        
        logger.info(f"📂 Loading enhanced data from {file_path}")
        
        if not Path(file_path).exists():
            # Fallback to original features
            file_path = self.data_dir / "processed" / "features.csv"
            if not Path(file_path).exists():
                raise FileNotFoundError(f"No processed data found. Run feature engineering first.")
        
        df = pd.read_csv(file_path)
        logger.info(f"✅ Loaded {len(df)} samples with {len(df.columns)} columns")
        
        return df
    
    def prepare_enhanced_data(self, df=None, test_size=0.2, random_state=42):
        """Prepare enhanced data for training with feature selection"""
        logger.info("🔧 Preparing enhanced data for training...")
        
        if df is None:
            df = self.load_data()
        
        # Define target and feature columns
        target_col = 'failure_within_hour'
        exclude_cols = ['timestamp', 'system_state', target_col]
        
        # Also exclude other target horizons to prevent data leakage
        exclude_cols.extend([col for col in df.columns if 'failure_within_' in col and col != target_col])
        
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")
        
        # Prepare features and target
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        self.feature_columns = feature_cols
        
        X = df[feature_cols]
        y = df[target_col]
        
        # Handle any remaining missing values
        X = X.fillna(0)
        X = X.replace([np.inf, -np.inf], 0)
        
        # Remove low variance features
        X = self._remove_low_variance_features(X)
        
        # Feature selection for better performance
        X, selected_features = self._select_important_features(X, y)
        self.selected_features = selected_features
        
        # Split data with stratification
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"✅ Enhanced data prepared:")
        logger.info(f"   Selected features: {len(self.selected_features)}")
        logger.info(f"   Training samples: {len(self.X_train)}")
        logger.info(f"   Testing samples: {len(self.X_test)}")
        logger.info(f"   Positive class rate (train): {self.y_train.mean():.2%}")
        logger.info(f"   Positive class rate (test): {self.y_test.mean():.2%}")
    
    def _remove_low_variance_features(self, X, threshold=0.01):
        """Remove features with low variance"""
        from sklearn.feature_selection import VarianceThreshold
        
        selector = VarianceThreshold(threshold=threshold)
        X_filtered = selector.fit_transform(X)
        
        selected_features = X.columns[selector.get_support()].tolist()
        removed_count = len(X.columns) - len(selected_features)
        
        if removed_count > 0:
            logger.info(f"   Removed {removed_count} low variance features")
        
        return pd.DataFrame(X_filtered, columns=selected_features, index=X.index)
    
    def _select_important_features(self, X, y, max_features=200):
        """Select most important features using multiple methods"""
        logger.info(f"🎯 Selecting top {max_features} features...")
        
        # Use Random Forest for feature importance
        rf_selector = RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1
        )
        rf_selector.fit(X, y)
        
        # Get feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf_selector.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Select top features
        if len(X.columns) > max_features:
            selected_features = feature_importance.head(max_features)['feature'].tolist()
            X_selected = X[selected_features]
            logger.info(f"   Selected top {len(selected_features)} features by importance")
        else:
            selected_features = X.columns.tolist()
            X_selected = X
        
        return X_selected, selected_features
    
    def train_enhanced_random_forest(self, use_randomized_search=True):
        """Train enhanced Random Forest with better hyperparameters"""
        logger.info("🌲 Training Enhanced Random Forest...")
        
        if use_randomized_search:
            param_dist = {
                'n_estimators': [200, 300, 500],
                'max_depth': [15, 20, 25, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None],
                'class_weight': ['balanced', 'balanced_subsample'],
                'bootstrap': [True, False]
            }
            
            rf = RandomForestClassifier(random_state=42, n_jobs=-1)
            search = RandomizedSearchCV(
                rf, param_dist, n_iter=20, cv=5, scoring='roc_auc', 
                n_jobs=-1, random_state=42, verbose=1
            )
            search.fit(self.X_train, self.y_train)
            best_rf = search.best_estimator_
            logger.info(f"   Best parameters: {search.best_params_}")
        else:
            best_rf = RandomForestClassifier(
                n_estimators=300,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
            best_rf.fit(self.X_train, self.y_train)
        
        # Calibrate probabilities
        calibrated_rf = CalibratedClassifierCV(best_rf, cv=3)
        calibrated_rf.fit(self.X_train, self.y_train)
        
        # Evaluate
        auc_score = self._evaluate_model(calibrated_rf, 'enhanced_random_forest')
        self.models['enhanced_random_forest'] = calibrated_rf
        
        return calibrated_rf, auc_score
    
    def train_enhanced_xgboost(self, use_randomized_search=True):
        """Train enhanced XGBoost with better hyperparameters"""
        if not XGBOOST_AVAILABLE:
            logger.error("❌ XGBoost not available")
            return None, 0.0
        
        logger.info("🚀 Training Enhanced XGBoost...")
        
        if use_randomized_search:
            param_dist = {
                'n_estimators': [200, 300, 500],
                'max_depth': [4, 6, 8, 10],
                'learning_rate': [0.01, 0.05, 0.1, 0.15],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0],
                'reg_alpha': [0, 0.1, 0.5],
                'reg_lambda': [0.1, 1, 2],
                'scale_pos_weight': [1, 2, 3]
            }
            
            xgb_model = xgb.XGBClassifier(
                random_state=42, 
                n_jobs=-1,
                eval_metric='logloss',
                use_label_encoder=False
            )
            
            search = RandomizedSearchCV(
                xgb_model, param_dist, n_iter=25, cv=5, scoring='roc_auc',
                n_jobs=-1, random_state=42, verbose=1
            )
            search.fit(self.X_train, self.y_train)
            best_xgb = search.best_estimator_
            logger.info(f"   Best parameters: {search.best_params_}")
        else:
            # Calculate scale_pos_weight for class imbalance
            scale_pos_weight = (self.y_train == 0).sum() / (self.y_train == 1).sum()
            
            best_xgb = xgb.XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_alpha=0.1,
                reg_lambda=1,
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                n_jobs=-1,
                eval_metric='logloss',
                use_label_encoder=False
            )
            best_xgb.fit(self.X_train, self.y_train)
        
        # Evaluate
        auc_score = self._evaluate_model(best_xgb, 'enhanced_xgboost')
        self.models['enhanced_xgboost'] = best_xgb
        
        return best_xgb, auc_score
    
    def train_enhanced_lightgbm(self, use_randomized_search=True):
        """Train LightGBM model"""
        if not LIGHTGBM_AVAILABLE:
            logger.warning("⚠️ LightGBM not available")
            return None, 0.0
        
        logger.info("💡 Training Enhanced LightGBM...")
        
        if use_randomized_search:
            param_dist = {
                'n_estimators': [200, 300, 500],
                'max_depth': [4, 6, 8, -1],
                'learning_rate': [0.01, 0.05, 0.1],
                'num_leaves': [31, 50, 100],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0],
                'reg_alpha': [0, 0.1, 0.5],
                'reg_lambda': [0.1, 1, 2]
            }
            
            lgb_model = lgb.LGBMClassifier(
                random_state=42,
                n_jobs=-1,
                class_weight='balanced',
                verbose=-1
            )
            
            search = RandomizedSearchCV(
                lgb_model, param_dist, n_iter=20, cv=5, scoring='roc_auc',
                n_jobs=-1, random_state=42, verbose=1
            )
            search.fit(self.X_train, self.y_train)
            best_lgb = search.best_estimator_
            logger.info(f"   Best parameters: {search.best_params_}")
        else:
            best_lgb = lgb.LGBMClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                num_leaves=50,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_alpha=0.1,
                reg_lambda=1,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
            best_lgb.fit(self.X_train, self.y_train)
        
        # Evaluate
        auc_score = self._evaluate_model(best_lgb, 'enhanced_lightgbm')
        self.models['enhanced_lightgbm'] = best_lgb
        
        return best_lgb, auc_score
    
    def train_enhanced_gradient_boosting(self, use_randomized_search=True):
        """Train Gradient Boosting Classifier"""
        logger.info("📈 Training Enhanced Gradient Boosting...")
        
        if use_randomized_search:
            param_dist = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.15],
                'subsample': [0.8, 0.9, 1.0],
                'max_features': ['sqrt', 'log2', None]
            }
            
            gb = GradientBoostingClassifier(random_state=42)
            search = RandomizedSearchCV(
                gb, param_dist, n_iter=15, cv=5, scoring='roc_auc',
                n_jobs=-1, random_state=42, verbose=1
            )
            search.fit(self.X_train, self.y_train)
            best_gb = search.best_estimator_
            logger.info(f"   Best parameters: {search.best_params_}")
        else:
            best_gb = GradientBoostingClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.9,
                max_features='sqrt',
                random_state=42
            )
            best_gb.fit(self.X_train, self.y_train)
        
        # Evaluate
        auc_score = self._evaluate_model(best_gb, 'enhanced_gradient_boosting')
        self.models['enhanced_gradient_boosting'] = best_gb
        
        return best_gb, auc_score
    
    def train_enhanced_logistic_regression(self, use_randomized_search=True):
        """Train enhanced Logistic Regression with proper scaling"""
        logger.info("📊 Training Enhanced Logistic Regression...")
        
        # Use RobustScaler for better handling of outliers
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(self.X_train)
        X_test_scaled = scaler.transform(self.X_test)
        
        self.scalers['robust_scaler'] = scaler
        
        if use_randomized_search:
            param_dist = {
                'C': [0.01, 0.1, 1.0, 10.0, 100.0],
                'penalty': ['l1', 'l2', 'elasticnet'],
                'solver': ['liblinear', 'saga'],
                'class_weight': ['balanced', None],
                'max_iter': [1000, 2000]
            }
            
            lr = LogisticRegression(random_state=42)
            search = RandomizedSearchCV(
                lr, param_dist, n_iter=20, cv=5, scoring='roc_auc',
                n_jobs=-1, random_state=42, verbose=1
            )
            search.fit(X_train_scaled, self.y_train)
            best_lr = search.best_estimator_
            logger.info(f"   Best parameters: {search.best_params_}")
        else:
            best_lr = LogisticRegression(
                C=1.0,
                penalty='l2',
                solver='liblinear',
                class_weight='balanced',
                max_iter=2000,
                random_state=42
            )
            best_lr.fit(X_train_scaled, self.y_train)
        
        # Create a pipeline-like wrapper for prediction
        class ScaledLogisticRegression:
            def __init__(self, scaler, model):
                self.scaler = scaler
                self.model = model
            
            def predict(self, X):
                return self.model.predict(self.scaler.transform(X))
            
            def predict_proba(self, X):
                return self.model.predict_proba(self.scaler.transform(X))
        
        scaled_lr = ScaledLogisticRegression(scaler, best_lr)
        
        # Evaluate
        auc_score = self._evaluate_model(scaled_lr, 'enhanced_logistic_regression')
        self.models['enhanced_logistic_regression'] = scaled_lr
        
        return scaled_lr, auc_score
    
    def train_voting_ensemble(self):
        """Train a voting ensemble of the best models"""
        logger.info("🗳️ Training Voting Ensemble...")
        
        if len(self.models) < 2:
            logger.warning("Not enough models for ensemble")
            return None, 0.0
        
        from sklearn.ensemble import VotingClassifier
        
        # Select top 3 models by AUC
        model_scores = []
        for name, model in self.models.items():
            if name != 'voting_ensemble':  # Avoid recursion
                try:
                    y_pred_proba = model.predict_proba(self.X_test)[:, 1]
                    auc = roc_auc_score(self.y_test, y_pred_proba)
                    model_scores.append((name, model, auc))
                except:
                    continue
        
        # Sort by AUC and take top 3
        model_scores.sort(key=lambda x: x[2], reverse=True)
        top_models = model_scores[:3]
        
        if len(top_models) < 2:
            logger.warning("Not enough valid models for ensemble")
            return None, 0.0
        
        # Create voting classifier
        estimators = [(name, model) for name, model, _ in top_models]
        
        voting_clf = VotingClassifier(
            estimators=estimators,
            voting='soft'  # Use probabilities
        )
        
        voting_clf.fit(self.X_train, self.y_train)
        
        # Evaluate
        auc_score = self._evaluate_model(voting_clf, 'voting_ensemble')
        self.models['voting_ensemble'] = voting_clf
        
        logger.info(f"   Ensemble contains: {[name for name, _, _ in top_models]}")
        
        return voting_clf, auc_score
    
    def _evaluate_model(self, model, model_name):
        """Evaluate a single model comprehensively"""
        try:
            # Make predictions
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(self.y_test, y_pred),
                'precision': precision_score(self.y_test, y_pred, zero_division=0),
                'recall': recall_score(self.y_test, y_pred, zero_division=0),
                'f1_score': f1_score(self.y_test, y_pred, zero_division=0),
                'auc_score': roc_auc_score(self.y_test, y_pred_proba)
            }
            
            # Store results
            self.results[model_name] = metrics
            
            # Log to MLflow if available
            if MLFLOW_AVAILABLE:
                self._log_to_mlflow(model_name, model, metrics)
            
            logger.info(f"   AUC: {metrics['auc_score']:.4f}, F1: {metrics['f1_score']:.4f}")
            
            return metrics['auc_score']
            
        except Exception as e:
            logger.error(f"Failed to evaluate {model_name}: {str(e)}")
            return 0.0
    
    def _log_to_mlflow(self, model_name, model, metrics):
        """Log model and metrics to MLflow"""
        try:
            with mlflow.start_run(run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                # Log metrics
                mlflow.log_metrics(metrics)
                
                # Log parameters if available
                if hasattr(model, 'get_params'):
                    try:
                        mlflow.log_params(model.get_params())
                    except:
                        pass
                
                # Log model
                try:
                    mlflow.sklearn.log_model(model, model_name)
                except:
                    pass
                
                # Log feature importance if available
                if hasattr(model, 'feature_importances_'):
                    feature_importance = dict(zip(self.selected_features, model.feature_importances_))
                    mlflow.log_dict(feature_importance, f"{model_name}_feature_importance.json")
                
        except Exception as e:
            logger.warning(f"Failed to log to MLflow: {str(e)}")
    
    def train_all_enhanced_models(self, use_search=True):
        """Train all available enhanced models"""
        logger.info("🚀 Starting enhanced model training pipeline...")
        
        # Prepare data if not already done
        if self.X_train is None:
            self.prepare_enhanced_data()
        
        # Training methods
        training_methods = [
            ('enhanced_random_forest', self.train_enhanced_random_forest),
            ('enhanced_logistic_regression', self.train_enhanced_logistic_regression),
            ('enhanced_gradient_boosting', self.train_enhanced_gradient_boosting),
        ]
        
        # Add advanced models if available
        if XGBOOST_AVAILABLE:
            training_methods.append(('enhanced_xgboost', self.train_enhanced_xgboost))
        
        if LIGHTGBM_AVAILABLE:
            training_methods.append(('enhanced_lightgbm', self.train_enhanced_lightgbm))
        
        # Train each model
        for model_name, train_method in training_methods:
            try:
                logger.info(f"\n{'='*60}")
                logger.info(f"Training {model_name.upper()}")
                logger.info('='*60)
                
                model, score = train_method(use_randomized_search=use_search)
                
            except Exception as e:
                logger.error(f"❌ Failed to train {model_name}: {str(e)}")
        
        # Train ensemble if we have multiple models
        if len(self.models) >= 2:
            try:
                logger.info(f"\n{'='*60}")
                logger.info("TRAINING VOTING ENSEMBLE")
                logger.info('='*60)
                self.train_voting_ensemble()
            except Exception as e:
                logger.error(f"❌ Failed to train ensemble: {str(e)}")
        
        # Find best model
        if self.results:
            best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['auc_score'])
            self.best_model_name = best_model_name
            self.best_model = self.models[best_model_name]
            
            # Save everything
            self.save_enhanced_models()
            
            return self.results, best_model_name, self.best_model
        else:
            logger.error("❌ No models were trained successfully")
            return {}, None, None
    
    def save_enhanced_models(self):
        """Save enhanced models and artifacts"""
        logger.info("💾 Saving enhanced models and artifacts...")
        
        # Save individual models
        for model_name, model in self.models.items():
            model_path = self.model_dir / f"{model_name}_model.joblib"
            joblib.dump(model, model_path)
            logger.info(f"   Saved {model_name} to {model_path}")
        
        # Save scalers
        for scaler_name, scaler in self.scalers.items():
            scaler_path = self.model_dir / f"{scaler_name}.joblib"
            joblib.dump(scaler, scaler_path)
            logger.info(f"   Saved {scaler_name} to {scaler_path}")
        
        # Save selected feature columns
        features_path = self.model_dir / "selected_features.json"
        with open(features_path, 'w') as f:
            json.dump(self.selected_features, f, indent=2)
        logger.info(f"   Saved selected features to {features_path}")
        
        # Save enhanced training metadata
        metadata = {
            'training_date': datetime.now().isoformat(),
            'total_samples': len(self.X_train) + len(self.X_test),
            'training_samples': len(self.X_train),
            'testing_samples': len(self.X_test),
            'selected_feature_count': len(self.selected_features),
            'original_feature_count': len(self.feature_columns),
            'positive_class_rate': float(self.y_train.mean()),
            'models_trained': list(self.models.keys()),
            'best_model': self.best_model_name,
            'results': self.results,
            'enhancements': [
                'Advanced hyperparameter tuning',
                'Feature selection and engineering',
                'Probability calibration',
                'Ensemble methods',
                'Multiple gradient boosting algorithms',
                'Robust scaling for linear models'
            ]
        }
        
        metadata_path = self.model_dir / "enhanced_training_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info(f"   Saved enhanced metadata to {metadata_path}")
        
        logger.info("✅ All enhanced models and artifacts saved")

def main():
    """Main enhanced training function"""
    logger.info("🚀 Starting Enhanced MLOps Error Prediction Model Training")
    
    try:
        trainer = EnhancedModelTrainer()
        results, best_model_name, best_model = trainer.train_all_enhanced_models(use_search=True)
        
        if results:
            logger.info("\n" + "="*80)
            logger.info("🎉 ENHANCED TRAINING COMPLETED SUCCESSFULLY!")
            logger.info("="*80)
            logger.info(f"🏆 Best Model: {best_model_name}")
            logger.info(f"🏆 Best AUC Score: {results[best_model_name]['auc_score']:.4f}")
            
            logger.info("\n📊 All Enhanced Model Results:")
            for model_name, metrics in results.items():
                logger.info(f"\n{model_name.upper()}:")
                for metric_name, value in metrics.items():
                    logger.info(f"  {metric_name}: {value:.4f}")
            
            # Print improvement recommendations
            best_auc = results[best_model_name]['auc_score']
            if best_auc >= 0.90:
                logger.info("\n🌟 Excellent model performance!")
            elif best_auc >= 0.80:
                logger.info("\n✅ Good model performance!")
            else:
                logger.info("\n⚠️ Model performance could be improved. Consider:")
                logger.info("   - Generating more diverse failure scenarios")
                logger.info("   - Adding more time-based features")
                logger.info("   - Tuning hyperparameters further")
            
            logger.info("\n🎯 Next Steps:")
            logger.info("1. Start the enhanced API server:")
            logger.info("   python src/models/enhanced_prediction.py")
            logger.info("2. Test predictions with the improved model")
            logger.info("3. Monitor model performance in production")
            
        else:
            logger.error("❌ Enhanced training failed - no models were trained")
            return False
            
    except Exception as e:
        logger.error(f"❌ Enhanced training pipeline failed: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)