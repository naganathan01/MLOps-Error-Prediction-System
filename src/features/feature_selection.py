"""
Feature selection module for identifying the most important features
for the error prediction model using various selection techniques.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
from datetime import datetime
from typing import List, Dict, Tuple

from sklearn.feature_selection import (
    SelectKBest, f_classif, chi2, mutual_info_classif,
    RFE, RFECV, SelectFromModel
)
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureSelector:
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.selected_features = {}
        self.feature_scores = {}
        self.selection_results = {}
        
    def load_data(self, file_path: str = "data/processed/features.csv") -> Tuple[pd.DataFrame, pd.Series]:
        """Load processed data for feature selection"""
        logger.info(f"📂 Loading data from {file_path}")
        
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        df = pd.read_csv(file_path)
        
        # Separate features and target
        target_col = 'failure_within_hour'
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found")
        
        # Exclude non-feature columns
        exclude_cols = ['timestamp', 'system_state', target_col]
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols]
        y = df[target_col]
        
        logger.info(f"✅ Loaded data: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"Target distribution: {y.value_counts().to_dict()}")
        
        return X, y
    
    def univariate_selection(self, X: pd.DataFrame, y: pd.Series, 
                           k: int = 20, score_func=f_classif) -> List[str]:
        """Select features using univariate statistical tests"""
        logger.info(f"📊 Running univariate feature selection (k={k})...")
        
        selector = SelectKBest(score_func=score_func, k=k)
        X_selected = selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_features = X.columns[selector.get_support()].tolist()
        
        # Get feature scores
        scores = selector.scores_
        feature_scores = dict(zip(X.columns, scores))
        
        # Sort by score
        sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
        
        self.selected_features['univariate'] = selected_features
        self.feature_scores['univariate'] = feature_scores
        
        logger.info(f"✅ Selected {len(selected_features)} features using univariate selection")
        logger.info(f"Top 5 features: {[f[0] for f in sorted_features[:5]]}")
        
        return selected_features
    
    def correlation_selection(self, X: pd.DataFrame, y: pd.Series, 
                            threshold: float = 0.95) -> List[str]:
        """Remove highly correlated features"""
        logger.info(f"🔗 Running correlation-based feature selection (threshold={threshold})...")
        
        # Calculate correlation matrix
        corr_matrix = X.corr().abs()
        
        # Find pairs of highly correlated features
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > threshold:
                    col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                    high_corr_pairs.append((col1, col2, corr_matrix.iloc[i, j]))
        
        # Remove features with lower correlation to target
        features_to_remove = set()
        target_corr = X.corrwith(y).abs()
        
        for col1, col2, corr_val in high_corr_pairs:
            if target_corr[col1] < target_corr[col2]:
                features_to_remove.add(col1)
            else:
                features_to_remove.add(col2)
        
        selected_features = [col for col in X.columns if col not in features_to_remove]
        
        self.selected_features['correlation'] = selected_features
        self.selection_results['correlation'] = {
            'removed_features': list(features_to_remove),
            'high_corr_pairs': high_corr_pairs
        }
        
        logger.info(f"✅ Removed {len(features_to_remove)} highly correlated features")
        logger.info(f"Remaining features: {len(selected_features)}")
        
        return selected_features
    
    def recursive_feature_elimination(self, X: pd.DataFrame, y: pd.Series, 
                                    n_features: int = 20, cv: int = 5) -> List[str]:
        """Select features using Recursive Feature Elimination with Cross-Validation"""
        logger.info(f"🔄 Running Recursive Feature Elimination (n_features={n_features})...")
        
        # Use RandomForest as the base estimator
        estimator = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        
        # Use RFECV to find optimal number of features
        selector = RFECV(estimator=estimator, step=1, cv=cv, scoring='roc_auc', n_jobs=-1)
        X_selected = selector.fit_transform(X, y)
        
        # Get selected features
        selected_features = X.columns[selector.get_support()].tolist()
        
        # Get feature rankings
        feature_rankings = dict(zip(X.columns, selector.ranking_))
        
        self.selected_features['rfe'] = selected_features
        self.feature_scores['rfe'] = feature_rankings
        self.selection_results['rfe'] = {
            'optimal_features': selector.n_features_,
            'cv_scores': selector.grid_scores_
        }
        
        logger.info(f"✅ Selected {len(selected_features)} features using RFE")
        logger.info(f"Optimal number of features: {selector.n_features_}")
        
        return selected_features
    
    def tree_based_selection(self, X: pd.DataFrame, y: pd.Series,
                           threshold: str = 'median') -> List[str]:
        """Select features using tree-based feature importance"""
        logger.info(f"🌲 Running tree-based feature selection...")
        
        # Use Extra Trees for feature importance
        estimator = ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        estimator.fit(X, y)
        
        # Select features based on importance
        selector = SelectFromModel(estimator, threshold=threshold)
        X_selected = selector.fit_transform(X, y)
        
        # Get selected features
        selected_features = X.columns[selector.get_support()].tolist()
        
        # Get feature importance scores
        feature_importance = dict(zip(X.columns, estimator.feature_importances_))
        
        self.selected_features['tree_based'] = selected_features
        self.feature_scores['tree_based'] = feature_importance
        
        logger.info(f"✅ Selected {len(selected_features)} features using tree-based importance")
        
        return selected_features
    
    def lasso_selection(self, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> List[str]:
        """Select features using Lasso regularization"""
        logger.info(f"📐 Running Lasso-based feature selection...")
        
        # Use LassoCV to find optimal alpha
        lasso = LassoCV(cv=cv, random_state=42, max_iter=1000)
        lasso.fit(X, y)
        
        # Select features with non-zero coefficients
        selected_mask = lasso.coef_ != 0
        selected_features = X.columns[selected_mask].tolist()
        
        # Get feature coefficients
        feature_coefs = dict(zip(X.columns, lasso.coef_))
        
        self.selected_features['lasso'] = selected_features
        self.feature_scores['lasso'] = feature_coefs
        self.selection_results['lasso'] = {
            'alpha': lasso.alpha_,
            'n_iter': lasso.n_iter_
        }
        
        logger.info(f"✅ Selected {len(selected_features)} features using Lasso")
        logger.info(f"Optimal alpha: {lasso.alpha_:.6f}")
        
        return selected_features
    
    def mutual_information_selection(self, X: pd.DataFrame, y: pd.Series, 
                                   k: int = 20) -> List[str]:
        """Select features using mutual information"""
        logger.info(f"ℹ️ Running mutual information feature selection (k={k})...")
        
        selector = SelectKBest(score_func=mutual_info_classif, k=k)
        X_selected = selector.fit_transform(X, y)
        
        # Get selected features
        selected_features = X.columns[selector.get_support()].tolist()
        
        # Get mutual information scores
        mi_scores = selector.scores_
        feature_scores = dict(zip(X.columns, mi_scores))
        
        self.selected_features['mutual_info'] = selected_features
        self.feature_scores['mutual_info'] = feature_scores
        
        logger.info(f"✅ Selected {len(selected_features)} features using mutual information")
        
        return selected_features
    
    def ensemble_selection(self, X: pd.DataFrame, y: pd.Series, 
                         min_votes: int = 3) -> List[str]:
        """Combine multiple selection methods using ensemble voting"""
        logger.info(f"🗳️ Running ensemble feature selection (min_votes={min_votes})...")
        
        # Run all selection methods
        methods = {
            'univariate': lambda: self.univariate_selection(X, y, k=30),
            'correlation': lambda: self.correlation_selection(X, y),
            'rfe': lambda: self.recursive_feature_elimination(X, y, n_features=25),
            'tree_based': lambda: self.tree_based_selection(X, y),
            'lasso': lambda: self.lasso_selection(X, y),
            'mutual_info': lambda: self.mutual_information_selection(X, y, k=30)
        }
        
        # Count votes for each feature
        feature_votes = {}
        method_results = {}
        
        for method_name, method_func in methods.items():
            try:
                selected = method_func()
                method_results[method_name] = selected
                
                for feature in selected:
                    feature_votes[feature] = feature_votes.get(feature, 0) + 1
                    
            except Exception as e:
                logger.warning(f"⚠️ Method {method_name} failed: {str(e)}")
        
        # Select features with minimum votes
        ensemble_features = [
            feature for feature, votes in feature_votes.items() 
            if votes >= min_votes
        ]
        
        # Sort by number of votes
        ensemble_features.sort(key=lambda x: feature_votes[x], reverse=True)
        
        self.selected_features['ensemble'] = ensemble_features
        self.selection_results['ensemble'] = {
            'feature_votes': feature_votes,
            'method_results': method_results,
            'min_votes_threshold': min_votes
        }
        
        logger.info(f"✅ Ensemble selection completed: {len(ensemble_features)} features")
        logger.info(f"Top voted features: {ensemble_features[:10]}")
        
        return ensemble_features
    
    def evaluate_feature_sets(self, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> Dict:
        """Evaluate different feature sets using cross-validation"""
        logger.info("📊 Evaluating feature sets...")
        
        # Base model for evaluation
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        
        evaluation_results = {}
        
        # Evaluate each feature set
        for method_name, features in self.selected_features.items():
            if not features:
                continue
                
            try:
                X_subset = X[features]
                scores = cross_val_score(model, X_subset, y, cv=cv, scoring='roc_auc')
                
                evaluation_results[method_name] = {
                    'n_features': len(features),
                    'mean_auc': scores.mean(),
                    'std_auc': scores.std(),
                    'features': features
                }
                
                logger.info(f"   {method_name}: {len(features)} features, AUC: {scores.mean():.4f} ±{scores.std():.4f}")
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to evaluate {method_name}: {str(e)}")
        
        # Sort by AUC score
        sorted_results = sorted(evaluation_results.items(), 
                              key=lambda x: x[1]['mean_auc'], reverse=True)
        
        logger.info(f"🏆 Best feature set: {sorted_results[0][0]} (AUC: {sorted_results[0][1]['mean_auc']:.4f})")
        
        return evaluation_results
    
    def plot_feature_importance(self, method: str = 'tree_based', top_k: int = 20):
        """Plot feature importance scores"""
        if method not in self.feature_scores:
            logger.error(f"Feature scores for method '{method}' not available")
            return
        
        scores = self.feature_scores[method]
        
        # Get top k features
        if method == 'rfe':
            # For RFE, lower ranking is better
            sorted_features = sorted(scores.items(), key=lambda x: x[1])[:top_k]
        else:
            sorted_features = sorted(scores.items(), key=lambda x: abs(x[1]), reverse=True)[:top_k]
        
        features, values = zip(*sorted_features)
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(features)), values)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Feature Score')
        plt.title(f'Top {top_k} Features - {method.title()} Method')
        plt.tight_layout()
        
        # Save plot
        plot_path = self.data_dir / "processed" / f"feature_importance_{method}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"💾 Saved feature importance plot to {plot_path}")
        
        plt.show()
    
    def generate_selection_report(self, evaluation_results: Dict) -> Dict:
        """Generate comprehensive feature selection report"""
        logger.info("📋 Generating feature selection report...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_original_features': len(self.feature_scores.get('univariate', {})),
                'selection_methods_used': list(self.selected_features.keys()),
                'best_method': max(evaluation_results.items(), key=lambda x: x[1]['mean_auc'])[0] if evaluation_results else None
            },
            'method_results': {},
            'evaluation_results': evaluation_results,
            'feature_rankings': {}
        }
        
        # Add method-specific results
        for method, features in self.selected_features.items():
            report['method_results'][method] = {
                'n_features_selected': len(features),
                'selected_features': features,
                'additional_info': self.selection_results.get(method, {})
            }
        
        # Add feature rankings from different methods
        for method, scores in self.feature_scores.items():
            if method == 'rfe':
                # For RFE, convert rankings to importance (lower rank = higher importance)
                max_rank = max(scores.values())
                rankings = {f: max_rank - rank + 1 for f, rank in scores.items()}
            else:
                rankings = {f: abs(score) for f, score in scores.items()}
            
            sorted_rankings = sorted(rankings.items(), key=lambda x: x[1], reverse=True)
            report['feature_rankings'][method] = sorted_rankings[:20]  # Top 20
        
        return report
    
    def save_selected_features(self, output_dir: str = "models"):
        """Save selected features for later use"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Save selected features
        features_path = output_dir / "selected_features.json"
        with open(features_path, 'w') as f:
            json.dump(self.selected_features, f, indent=2)
        logger.info(f"💾 Saved selected features to {features_path}")
        
        # Save feature scores
        scores_path = output_dir / "feature_scores.json"
        with open(scores_path, 'w') as f:
            json.dump(self.feature_scores, f, indent=2, default=str)
        logger.info(f"💾 Saved feature scores to {scores_path}")
    
    def run_feature_selection_pipeline(self, min_votes: int = 3) -> Dict:
        """Run complete feature selection pipeline"""
        logger.info("🚀 Starting feature selection pipeline...")
        
        # Load data
        X, y = self.load_data()
        
        # Run ensemble selection
        selected_features = self.ensemble_selection(X, y, min_votes=min_votes)
        
        # Evaluate all feature sets
        evaluation_results = self.evaluate_feature_sets(X, y)
        
        # Generate report
        report = self.generate_selection_report(evaluation_results)
        
        # Save results
        self.save_selected_features()
        
        # Save report
        report_path = self.data_dir / "processed" / "feature_selection_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"💾 Saved feature selection report to {report_path}")
        
        # Create visualizations
        if 'tree_based' in self.feature_scores:
            self.plot_feature_importance('tree_based')
        
        logger.info("🎉 Feature selection pipeline completed!")
        logger.info(f"Best method: {report['summary']['best_method']}")
        logger.info(f"Selected {len(selected_features)} features using ensemble method")
        
        return report

def main():
    """Example usage of FeatureSelector"""
    selector = FeatureSelector()
    
    try:
        report = selector.run_feature_selection_pipeline(min_votes=3)
        
        print("\n🏆 Feature Selection Results:")
        print(f"Best method: {report['summary']['best_method']}")
        print(f"Total methods used: {len(report['summary']['selection_methods_used'])}")
        
        # Show top features from ensemble method
        if 'ensemble' in selector.selected_features:
            ensemble_features = selector.selected_features['ensemble'][:10]
            print(f"\nTop 10 Ensemble Features:")
            for i, feature in enumerate(ensemble_features, 1):
                votes = selector.selection_results['ensemble']['feature_votes'][feature]
                print(f"{i:2d}. {feature} ({votes} votes)")
        
    except Exception as e:
        logger.error(f"❌ Feature selection failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()