"""
Model evaluation module for comprehensive assessment of trained models.
Includes metrics calculation, visualizations, and performance reports.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
import joblib
from datetime import datetime
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)
from sklearn.model_selection import cross_val_score, learning_curve
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self, model_dir="models", data_dir="data"):
        self.model_dir = Path(model_dir)
        self.data_dir = Path(data_dir)
        self.models = {}
        self.scaler = None
        self.feature_columns = []
        self.evaluation_results = {}
        
    def load_models(self):
        """Load trained models and preprocessing artifacts"""
        logger.info("📥 Loading trained models...")
        
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
                self.models[name] = joblib.load(file_path)
                logger.info(f"   Loaded {name} model")
        
        # Load scaler
        scaler_file = self.model_dir / "scaler.joblib"
        if scaler_file.exists():
            self.scaler = joblib.load(scaler_file)
            logger.info("   Loaded scaler")
        
        # Load feature columns
        features_file = self.model_dir / "feature_columns.json"
        if features_file.exists():
            with open(features_file, 'r') as f:
                self.feature_columns = json.load(f)
            logger.info(f"   Loaded {len(self.feature_columns)} feature columns")
        
        if not self.models:
            raise ValueError("No models were loaded successfully")
        
        logger.info(f"✅ Loaded {len(self.models)} models")
    
    def load_test_data(self, file_path: str = "data/processed/features.csv", 
                       test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Load and split data for evaluation"""
        logger.info(f"📂 Loading test data from {file_path}")
        
        df = pd.read_csv(file_path)
        
        # Prepare features and target
        target_col = 'failure_within_hour'
        exclude_cols = ['timestamp', 'system_state', target_col]
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols]
        y = df[target_col]
        
        # Split data (same as training split)
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        logger.info(f"✅ Data loaded: {len(X_test)} test samples")
        return X_train, X_test, y_train, y_test
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         y_pred_proba: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='binary'),
            'recall': recall_score(y_true, y_pred, average='binary'),
            'f1_score': f1_score(y_true, y_pred, average='binary'),
            'roc_auc': roc_auc_score(y_true, y_pred_proba),
            'average_precision': average_precision_score(y_true, y_pred_proba)
        }
        
        # Calculate specificity
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        return metrics
    
    def evaluate_single_model(self, model, model_name: str, X_test: pd.DataFrame, 
                             y_test: pd.Series, X_train: pd.DataFrame = None, 
                             y_train: pd.Series = None) -> Dict[str, Any]:
        """Evaluate a single model"""
        logger.info(f"📊 Evaluating {model_name} model...")
        
        # Prepare data for model
        if model_name == 'logistic_regression' and self.scaler:
            X_test_model = self.scaler.transform(X_test)
            X_train_model = self.scaler.transform(X_train) if X_train is not None else None
        else:
            X_test_model = X_test
            X_train_model = X_train
        
        # Make predictions
        try:
            y_pred = model.predict(X_test_model)
            y_pred_proba = model.predict_proba(X_test_model)[:, 1]
        except Exception as e:
            logger.error(f"❌ Prediction failed for {model_name}: {str(e)}")
            return {}
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_test, y_pred, y_pred_proba)
        
        # Cross-validation scores
        cv_scores = {}
        if X_train_model is not None and y_train is not None:
            try:
                cv_auc = cross_val_score(model, X_train_model, y_train, cv=5, scoring='roc_auc')
                cv_f1 = cross_val_score(model, X_train_model, y_train, cv=5, scoring='f1')
                
                cv_scores = {
                    'cv_auc_mean': cv_auc.mean(),
                    'cv_auc_std': cv_auc.std(),
                    'cv_f1_mean': cv_f1.mean(),
                    'cv_f1_std': cv_f1.std()
                }
            except Exception as e:
                logger.warning(f"⚠️ Cross-validation failed for {model_name}: {str(e)}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Classification report
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        evaluation_result = {
            'metrics': metrics,
            'cv_scores': cv_scores,
            'confusion_matrix': cm.tolist(),
            'classification_report': class_report,
            'predictions': {
                'y_true': y_test.tolist(),
                'y_pred': y_pred.tolist(),
                'y_pred_proba': y_pred_proba.tolist()
            }
        }
        
        logger.info(f"   AUC: {metrics['roc_auc']:.4f}")
        logger.info(f"   F1-Score: {metrics['f1_score']:.4f}")
        logger.info(f"   Precision: {metrics['precision']:.4f}")
        logger.info(f"   Recall: {metrics['recall']:.4f}")
        
        return evaluation_result
    
    def evaluate_all_models(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                           y_train: pd.Series, y_test: pd.Series) -> Dict[str, Any]:
        """Evaluate all loaded models"""
        logger.info("🔍 Evaluating all models...")
        
        all_results = {}
        
        for model_name, model in self.models.items():
            result = self.evaluate_single_model(
                model, model_name, X_test, y_test, X_train, y_train
            )
            if result:
                all_results[model_name] = result
        
        # Find best model
        if all_results:
            best_model_name = max(all_results.keys(), 
                                 key=lambda x: all_results[x]['metrics']['roc_auc'])
            
            logger.info(f"🏆 Best model: {best_model_name} (AUC: {all_results[best_model_name]['metrics']['roc_auc']:.4f})")
        
        self.evaluation_results = all_results
        return all_results
    
    def plot_roc_curves(self, evaluation_results: Dict[str, Any], save_path: str = None):
        """Plot ROC curves for all models"""
        logger.info("📈 Plotting ROC curves...")
        
        plt.figure(figsize=(10, 8))
        
        for model_name, results in evaluation_results.items():
            y_true = np.array(results['predictions']['y_true'])
            y_pred_proba = np.array(results['predictions']['y_pred_proba'])
            
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            auc_score = results['metrics']['roc_auc']
            
            plt.plot(fpr, tpr, label=f'{model_name.replace("_", " ").title()} (AUC = {auc_score:.3f})', linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"💾 Saved ROC curves to {save_path}")
        
        plt.tight_layout()
        plt.show()
    
    def plot_precision_recall_curves(self, evaluation_results: Dict[str, Any], save_path: str = None):
        """Plot Precision-Recall curves for all models"""
        logger.info("📈 Plotting Precision-Recall curves...")
        
        plt.figure(figsize=(10, 8))
        
        for model_name, results in evaluation_results.items():
            y_true = np.array(results['predictions']['y_true'])
            y_pred_proba = np.array(results['predictions']['y_pred_proba'])
            
            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
            avg_precision = results['metrics']['average_precision']
            
            plt.plot(recall, precision, label=f'{model_name.replace("_", " ").title()} (AP = {avg_precision:.3f})', linewidth=2)
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves Comparison')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"💾 Saved PR curves to {save_path}")
        
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrices(self, evaluation_results: Dict[str, Any], save_path: str = None):
        """Plot confusion matrices for all models"""
        logger.info("📊 Plotting confusion matrices...")
        
        n_models = len(evaluation_results)
        fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (model_name, results) in enumerate(evaluation_results.items()):
            cm = np.array(results['confusion_matrix'])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
            axes[idx].set_title(f'{model_name.replace("_", " ").title()}')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"💾 Saved confusion matrices to {save_path}")
        
        plt.tight_layout()
        plt.show()
    
    def plot_learning_curves(self, model_name: str, X_train: pd.DataFrame, 
                           y_train: pd.Series, save_path: str = None):
        """Plot learning curves for a specific model"""
        logger.info(f"📈 Plotting learning curves for {model_name}...")
        
        if model_name not in self.models:
            logger.error(f"Model {model_name} not found")
            return
        
        model = self.models[model_name]
        
        # Prepare data
        if model_name == 'logistic_regression' and self.scaler:
            X_train_model = self.scaler.transform(X_train)
        else:
            X_train_model = X_train
        
        # Calculate learning curves
        train_sizes, train_scores, val_scores = learning_curve(
            model, X_train_model, y_train, cv=5, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10), scoring='roc_auc'
        )
        
        train_mean = train_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)
        val_mean = val_scores.mean(axis=1)
        val_std = val_scores.std(axis=1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training AUC')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
        
        plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation AUC')
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
        
        plt.xlabel('Training Set Size')
        plt.ylabel('AUC Score')
        plt.title(f'Learning Curves - {model_name.replace("_", " ").title()}')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"💾 Saved learning curves to {save_path}")
        
        plt.tight_layout()
        plt.show()
    
    def plot_feature_importance(self, model_name: str = 'random_forest', top_k: int = 20, 
                               save_path: str = None):
        """Plot feature importance for tree-based models"""
        logger.info(f"📊 Plotting feature importance for {model_name}...")
        
        if model_name not in self.models:
            logger.error(f"Model {model_name} not found")
            return
        
        model = self.models[model_name]
        
        if not hasattr(model, 'feature_importances_'):
            logger.error(f"Model {model_name} doesn't have feature_importances_ attribute")
            return
        
        # Get feature importance
        importance = model.feature_importances_
        feature_names = self.feature_columns
        
        # Create DataFrame and sort
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        # Plot top k features
        top_features = importance_df.head(top_k)
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_k} Feature Importance - {model_name.replace("_", " ").title()}')
        plt.gca().invert_yaxis()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"💾 Saved feature importance plot to {save_path}")
        
        plt.tight_layout()
        plt.show()
        
        return importance_df
    
    def generate_evaluation_report(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive evaluation report"""
        logger.info("📋 Generating evaluation report...")
        
        # Summary statistics
        summary = {
            'timestamp': datetime.now().isoformat(),
            'models_evaluated': list(evaluation_results.keys()),
            'best_model': max(evaluation_results.keys(), 
                            key=lambda x: evaluation_results[x]['metrics']['roc_auc']),
            'metrics_comparison': {}
        }
        
        # Compare metrics across models
        metric_names = ['roc_auc', 'f1_score', 'precision', 'recall', 'accuracy']
        for metric in metric_names:
            summary['metrics_comparison'][metric] = {
                model_name: results['metrics'][metric]
                for model_name, results in evaluation_results.items()
            }
        
        # Detailed results
        detailed_results = {}
        for model_name, results in evaluation_results.items():
            detailed_results[model_name] = {
                'performance_metrics': results['metrics'],
                'cross_validation': results['cv_scores'],
                'confusion_matrix': results['confusion_matrix'],
                'classification_report': results['classification_report']
            }
        
        report = {
            'summary': summary,
            'detailed_results': detailed_results,
            'recommendations': self.generate_recommendations(evaluation_results)
        }
        
        return report
    
    def generate_recommendations(self, evaluation_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on evaluation results"""
        recommendations = []
        
        # Find best model
        best_model = max(evaluation_results.keys(), 
                        key=lambda x: evaluation_results[x]['metrics']['roc_auc'])
        best_auc = evaluation_results[best_model]['metrics']['roc_auc']
        
        recommendations.append(f"✅ Use {best_model.replace('_', ' ').title()} as the primary model (AUC: {best_auc:.4f})")
        
        # Check for overfitting
        for model_name, results in evaluation_results.items():
            if 'cv_scores' in results and results['cv_scores']:
                cv_auc = results['cv_scores'].get('cv_auc_mean', 0)
                test_auc = results['metrics']['roc_auc']
                
                if test_auc - cv_auc > 0.05:
                    recommendations.append(f"⚠️ {model_name.replace('_', ' ').title()} may be overfitting")
        
        # Performance thresholds
        high_performance_threshold = 0.85
        acceptable_threshold = 0.75
        
        high_performers = [name for name, results in evaluation_results.items()
                          if results['metrics']['roc_auc'] >= high_performance_threshold]
        
        if high_performers:
            recommendations.append(f"🎯 High-performing models: {', '.join(high_performers)}")
        
        low_performers = [name for name, results in evaluation_results.items()
                         if results['metrics']['roc_auc'] < acceptable_threshold]
        
        if low_performers:
            recommendations.append(f"🔧 Consider tuning or replacing: {', '.join(low_performers)}")
        
        # Class imbalance check
        for model_name, results in evaluation_results.items():
            precision = results['metrics']['precision']
            recall = results['metrics']['recall']
            
            if precision > 0.9 and recall < 0.5:
                recommendations.append(f"⚖️ {model_name.replace('_', ' ').title()} has low recall - consider adjusting threshold")
            elif recall > 0.9 and precision < 0.5:
                recommendations.append(f"⚖️ {model_name.replace('_', ' ').title()} has low precision - consider adjusting threshold")
        
        return recommendations
    
    def save_evaluation_results(self, evaluation_results: Dict[str, Any], 
                               report: Dict[str, Any], output_dir: str = None):
        """Save evaluation results and report"""
        if output_dir is None:
            output_dir = self.data_dir / "processed"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True)
        
        # Save detailed results
        results_path = output_dir / "model_evaluation_results.json"
        with open(results_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2, default=str)
        logger.info(f"💾 Saved evaluation results to {results_path}")
        
        # Save report
        report_path = output_dir / "model_evaluation_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"💾 Saved evaluation report to {report_path}")
    
    def run_complete_evaluation(self, create_plots: bool = True) -> Dict[str, Any]:
        """Run complete model evaluation pipeline"""
        logger.info("🚀 Starting complete model evaluation...")
        
        # Load models and data
        self.load_models()
        X_train, X_test, y_train, y_test = self.load_test_data()
        
        # Evaluate all models
        evaluation_results = self.evaluate_all_models(X_train, X_test, y_train, y_test)
        
        if not evaluation_results:
            logger.error("❌ No evaluation results generated")
            return {}
        
        # Generate report
        report = self.generate_evaluation_report(evaluation_results)
        
        # Create visualizations
        if create_plots:
            plots_dir = self.data_dir / "processed" / "plots"
            plots_dir.mkdir(exist_ok=True)
            
            self.plot_roc_curves(evaluation_results, plots_dir / "roc_curves.png")
            self.plot_precision_recall_curves(evaluation_results, plots_dir / "pr_curves.png")
            self.plot_confusion_matrices(evaluation_results, plots_dir / "confusion_matrices.png")
            
            # Feature importance for tree-based models
            for model_name in ['random_forest', 'xgboost']:
                if model_name in self.models:
                    self.plot_feature_importance(model_name, save_path=plots_dir / f"feature_importance_{model_name}.png")
        
        # Save results
        self.save_evaluation_results(evaluation_results, report)
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("📊 EVALUATION SUMMARY")
        logger.info("="*60)
        logger.info(f"🏆 Best Model: {report['summary']['best_model']}")
        logger.info(f"🎯 Best AUC Score: {evaluation_results[report['summary']['best_model']]['metrics']['roc_auc']:.4f}")
        
        logger.info("\n📋 Recommendations:")
        for rec in report['recommendations']:
            logger.info(f"   {rec}")
        
        logger.info("\n🎉 Model evaluation completed successfully!")
        
        return report

def main():
    """Example usage of ModelEvaluator"""
    evaluator = ModelEvaluator()
    
    try:
        report = evaluator.run_complete_evaluation(create_plots=True)
        
        print("\n🏆 Evaluation Complete!")
        print(f"Best model: {report['summary']['best_model']}")
        print(f"Models evaluated: {len(report['summary']['models_evaluated'])}")
        
    except Exception as e:
        logger.error(f"❌ Model evaluation failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()