"""
Tests for model training and evaluation components.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import joblib
import json
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.training import ModelTrainer
from src.models.evaluation import ModelEvaluator
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

class TestModelTrainer:
    
    @pytest.fixture
    def sample_training_data(self):
        """Create sample data for model training"""
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        
        # Generate synthetic training data
        np.random.seed(42)
        n_samples = 1000
        
        # Create features that correlate with failure
        cpu_usage = np.random.uniform(0, 100, n_samples)
        memory_usage = np.random.uniform(0, 100, n_samples)
        error_count = np.random.poisson(2, n_samples)
        response_time = np.random.uniform(100, 1000, n_samples)
        
        # Create target based on features (higher values more likely to fail)
        failure_prob = (
            (cpu_usage / 100) * 0.3 +
            (memory_usage / 100) * 0.3 +
            (error_count / 10) * 0.2 +
            (response_time / 1000) * 0.2
        )
        
        failure_within_hour = (np.random.random(n_samples) < failure_prob).astype(int)
        
        # Create additional features
        data = {
            'cpu_usage': cpu_usage,
            'memory_usage': memory_usage,
            'disk_usage': np.random.uniform(0, 100, n_samples),
            'network_latency_ms': np.random.uniform(10, 500, n_samples),
            'error_count': error_count,
            'response_time_ms': response_time,
            'active_connections': np.random.poisson(50, n_samples),
            'hour': np.random.randint(0, 24, n_samples),
            'day_of_week': np.random.randint(0, 7, n_samples),
            'is_weekend': np.random.choice([0, 1], n_samples),
            'is_business_hours': np.random.choice([0, 1], n_samples),
            'cpu_memory_product': cpu_usage * memory_usage,
            'resource_pressure': (cpu_usage + memory_usage) / 2,
            'failure_within_hour': failure_within_hour,
            'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='5min')
        }
        
        df = pd.DataFrame(data)
        
        # Save to CSV
        processed_dir = Path(temp_dir) / 'processed'
        processed_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(processed_dir / 'features.csv', index=False)
        
        return temp_dir
    
    def test_data_loading(self, sample_training_data):
        """Test data loading functionality"""
        trainer = ModelTrainer(data_dir=sample_training_data)
        
        df = trainer.load_data()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert 'failure_within_hour' in df.columns
    
    def test_data_preparation(self, sample_training_data):
        """Test data preparation for training"""
        trainer = ModelTrainer(data_dir=sample_training_data)
        trainer.prepare_data()
        
        # Check that data is properly split
        assert trainer.X_train is not None
        assert trainer.X_test is not None
        assert trainer.y_train is not None
        assert trainer.y_test is not None
        
        # Check shapes
        assert len(trainer.X_train) > len(trainer.X_test)  # More training than test data
        assert len(trainer.X_train) == len(trainer.y_train)
        assert len(trainer.X_test) == len(trainer.y_test)
        
        # Check feature columns
        assert len(trainer.feature_columns) > 0
        assert 'failure_within_hour' not in trainer.feature_columns
        assert 'timestamp' not in trainer.feature_columns
    
    def test_random_forest_training(self, sample_training_data):
        """Test Random Forest model training"""
        trainer = ModelTrainer(data_dir=sample_training_data)
        trainer.prepare_data()
        
        model, score = trainer.train_random_forest(use_grid_search=False)
        
        # Check that model is trained
        assert model is not None
        assert isinstance(model, RandomForestClassifier)
        assert isinstance(score, float)
        assert 0 <= score <= 1
        
        # Check that model can make predictions
        predictions = model.predict(trainer.X_test)
        assert len(predictions) == len(trainer.y_test)
        assert set(predictions).issubset({0, 1})
    
    def test_logistic_regression_training(self, sample_training_data):
        """Test Logistic Regression model training"""
        trainer = ModelTrainer(data_dir=sample_training_data)
        trainer.prepare_data()
        
        model, score = trainer.train_logistic_regression(use_grid_search=False)
        
        # Check that model is trained
        assert model is not None
        assert isinstance(score, float)
        assert 0 <= score <= 1
        
        # Check that scaler is fitted
        assert trainer.scaler is not None
    
    def test_model_evaluation(self, sample_training_data):
        """Test model evaluation functionality"""
        trainer = ModelTrainer(data_dir=sample_training_data)
        trainer.prepare_data()
        
        # Train a simple model for testing
        rf_model, _ = trainer.train_random_forest(use_grid_search=False)
        
        # Evaluate models
        results, best_model_name, best_model = trainer.evaluate_models()
        
        assert isinstance(results, dict)
        assert len(results) > 0
        assert best_model_name in results
        assert best_model is not None
        
        # Check that results contain expected metrics
        for model_name, metrics in results.items():
            required_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_score']
            for metric in required_metrics:
                assert metric in metrics
                assert isinstance(metrics[metric], float)
                assert 0 <= metrics[metric] <= 1
    
    def test_model_saving(self, sample_training_data):
        """Test model saving functionality"""
        with tempfile.TemporaryDirectory() as temp_model_dir:
            trainer = ModelTrainer(data_dir=sample_training_data, model_dir=temp_model_dir)
            trainer.prepare_data()
            
            # Train and save models
            trainer.train_random_forest(use_grid_search=False)
            trainer.save_models()
            
            # Check that files are saved
            model_dir = Path(temp_model_dir)
            assert (model_dir / "random_forest_model.joblib").exists()
            assert (model_dir / "scaler.joblib").exists()
            assert (model_dir / "feature_columns.json").exists()
            assert (model_dir / "training_metadata.json").exists()
            
            # Check that saved model can be loaded
            saved_model = joblib.load(model_dir / "random_forest_model.joblib")
            assert saved_model is not None
            
            # Check metadata
            with open(model_dir / "training_metadata.json", 'r') as f:
                metadata = json.load(f)
                assert 'training_date' in metadata
                assert 'total_samples' in metadata
                assert 'feature_count' in metadata
    
    def test_complete_training_pipeline(self, sample_training_data):
        """Test complete training pipeline"""
        with tempfile.TemporaryDirectory() as temp_model_dir:
            trainer = ModelTrainer(data_dir=sample_training_data, model_dir=temp_model_dir)
            
            # Run complete training pipeline
            results, best_model_name, best_model = trainer.train_all_models(use_grid_search=False)
            
            # Check results
            assert isinstance(results, dict)
            assert len(results) > 0
            assert best_model_name is not None
            assert best_model is not None
            
            # Check that multiple models were trained
            expected_models = ['random_forest', 'logistic_regression']
            for model_name in expected_models:
                assert model_name in results
            
            # Check that files are saved
            model_dir = Path(temp_model_dir)
            for model_name in expected_models:
                assert (model_dir / f"{model_name}_model.joblib").exists()

class TestModelEvaluator:
    
    @pytest.fixture
    def trained_models_setup(self):
        """Setup trained models for evaluation testing"""
        # Create temporary directories
        temp_data_dir = tempfile.mkdtemp()
        temp_model_dir = tempfile.mkdtemp()
        
        # Generate and save test data (same as training test)
        np.random.seed(42)
        n_samples = 500
        
        cpu_usage = np.random.uniform(0, 100, n_samples)
        memory_usage = np.random.uniform(0, 100, n_samples)
        error_count = np.random.poisson(2, n_samples)
        
        failure_prob = (
            (cpu_usage / 100) * 0.3 +
            (memory_usage / 100) * 0.3 +
            (error_count / 10) * 0.4
        )
        failure_within_hour = (np.random.random(n_samples) < failure_prob).astype(int)
        
        data = {
            'cpu_usage': cpu_usage,
            'memory_usage': memory_usage,
            'disk_usage': np.random.uniform(0, 100, n_samples),
            'network_latency_ms': np.random.uniform(10, 500, n_samples),
            'error_count': error_count,
            'response_time_ms': np.random.uniform(100, 1000, n_samples),
            'active_connections': np.random.poisson(50, n_samples),
            'failure_within_hour': failure_within_hour,
            'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='5min')
        }
        
        df = pd.DataFrame(data)
        processed_dir = Path(temp_data_dir) / 'processed'
        processed_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(processed_dir / 'features.csv', index=False)
        
        # Train and save a simple model
        X = df.drop(['failure_within_hour', 'timestamp'], axis=1)
        y = df['failure_within_hour']
        
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X, y)
        
        # Save model and artifacts
        model_dir = Path(temp_model_dir)
        joblib.dump(model, model_dir / "random_forest_model.joblib")
        
        with open(model_dir / "feature_columns.json", 'w') as f:
            json.dump(list(X.columns), f)
        
        return temp_data_dir, temp_model_dir
    
    def test_model_loading(self, trained_models_setup):
        """Test model loading functionality"""
        temp_data_dir, temp_model_dir = trained_models_setup
        
        evaluator = ModelEvaluator(model_dir=temp_model_dir, data_dir=temp_data_dir)
        evaluator.load_models()
        
        # Check that models are loaded
        assert len(evaluator.models) > 0
        assert 'random_forest' in evaluator.models
        assert len(evaluator.feature_columns) > 0
    
    def test_metrics_calculation(self):
        """Test metrics calculation"""
        evaluator = ModelEvaluator()
        
        # Create sample predictions
        y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0])
        y_pred = np.array([0, 1, 1, 1, 0, 0, 1, 0])
        y_pred_proba = np.array([0.1, 0.6, 0.8, 0.9, 0.2, 0.4, 0.7, 0.3])
        
        metrics = evaluator.calculate_metrics(y_true, y_pred, y_pred_proba)
        
        # Check that all expected metrics are calculated
        expected_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'average_precision']
        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], float)
            assert 0 <= metrics[metric] <= 1
    
    def test_single_model_evaluation(self, trained_models_setup):
        """Test evaluation of a single model"""
        temp_data_dir, temp_model_dir = trained_models_setup
        
        evaluator = ModelEvaluator(model_dir=temp_model_dir, data_dir=temp_data_dir)
        evaluator.load_models()
        
        # Load test data
        X_train, X_test, y_train, y_test = evaluator.load_test_data()
        
        # Evaluate the model
        model = evaluator.models['random_forest']
        result = evaluator.evaluate_single_model(model, 'random_forest', X_test, y_test)
        
        # Check evaluation result structure
        assert 'metrics' in result
        assert 'confusion_matrix' in result
        assert 'predictions' in result
        
        # Check metrics
        metrics = result['metrics']
        assert 'accuracy' in metrics
        assert 'roc_auc' in metrics
        assert isinstance(metrics['accuracy'], float)
    
    def test_evaluation_report_generation(self, trained_models_setup):
        """Test evaluation report generation"""
        temp_data_dir, temp_model_dir = trained_models_setup
        
        evaluator = ModelEvaluator(model_dir=temp_model_dir, data_dir=temp_data_dir)
        evaluator.load_models()
        
        # Create mock evaluation results
        mock_results = {
            'random_forest': {
                'metrics': {
                    'accuracy': 0.85,
                    'precision': 0.80,
                    'recall': 0.75,
                    'f1_score': 0.77,
                    'roc_auc': 0.82
                },
                'cv_scores': {},
                'confusion_matrix': [[100, 20], [15, 65]],
                'classification_report': {},
                'predictions': {
                    'y_true': [0, 1, 0, 1],
                    'y_pred': [0, 1, 1, 1],
                    'y_pred_proba': [0.2, 0.8, 0.6, 0.9]
                }
            }
        }
        
        report = evaluator.generate_evaluation_report(mock_results)
        
        # Check report structure
        assert 'summary' in report
        assert 'detailed_results' in report
        assert 'recommendations' in report
        
        # Check summary
        summary = report['summary']
        assert 'models_evaluated' in summary
        assert 'best_model' in summary
        assert 'metrics_comparison' in summary

class TestModelIntegration:
    """Integration tests for the complete model pipeline"""
    
    def test_end_to_end_model_pipeline(self):
        """Test complete model training and evaluation pipeline"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup directories
            data_dir = Path(temp_dir) / 'data'
            model_dir = Path(temp_dir) / 'models'
            
            # Create synthetic data
            np.random.seed(42)
            n_samples = 800
            
            # Generate features with realistic relationships
            cpu_usage = np.random.beta(2, 3) * 100  # Skewed towards lower values
            memory_usage = cpu_usage * 0.7 + np.random.normal(0, 10, n_samples)
            memory_usage = np.clip(memory_usage, 0, 100)
            
            disk_usage = np.random.uniform(10, 90, n_samples)
            network_latency = np.random.exponential(50, n_samples)
            error_count = np.random.poisson(1, n_samples)
            response_time = cpu_usage * 5 + memory_usage * 3 + np.random.normal(200, 50, n_samples)
            active_connections = np.random.poisson(40, n_samples)
            
            # Create realistic failure pattern
            system_stress = (cpu_usage + memory_usage) / 200
            error_factor = np.minimum(error_count / 5, 1)
            latency_factor = np.minimum(network_latency / 200, 1)
            
            failure_prob = (system_stress * 0.4 + error_factor * 0.3 + latency_factor * 0.3)
            failure_prob = np.clip(failure_prob, 0.01, 0.99)  # Avoid extreme probabilities
            
            failure_within_hour = (np.random.random(n_samples) < failure_prob).astype(int)
            
            # Add time-based features
            timestamps = pd.date_range('2024-01-01', periods=n_samples, freq='5min')
            hour = timestamps.hour
            day_of_week = timestamps.dayofweek
            is_weekend = (day_of_week >= 5).astype(int)
            is_business_hours = ((hour >= 9) & (hour <= 17) & (is_weekend == 0)).astype(int)
            
            # Create interaction features
            cpu_memory_product = cpu_usage * memory_usage
            resource_pressure = (cpu_usage + memory_usage + disk_usage) / 3
            performance_score = response_time / (active_connections + 1)
            
            # Create rolling features (simplified)
            cpu_rolling_mean_5 = pd.Series(cpu_usage).rolling(5, min_periods=1).mean().values
            memory_rolling_std_5 = pd.Series(memory_usage).rolling(5, min_periods=1).std().fillna(0).values
            
            # Create feature dataframe
            feature_data = {
                'timestamp': timestamps,
                'cpu_usage': cpu_usage,
                'memory_usage': memory_usage,
                'disk_usage': disk_usage,
                'network_latency_ms': network_latency,
                'error_count': error_count,
                'response_time_ms': response_time,
                'active_connections': active_connections,
                'hour': hour,
                'day_of_week': day_of_week,
                'is_weekend': is_weekend,
                'is_business_hours': is_business_hours,
                'cpu_memory_product': cpu_memory_product,
                'resource_pressure': resource_pressure,
                'performance_score': performance_score,
                'cpu_rolling_mean_5': cpu_rolling_mean_5,
                'memory_rolling_std_5': memory_rolling_std_5,
                'failure_within_hour': failure_within_hour
            }
            
            df = pd.DataFrame(feature_data)
            
            # Save data
            processed_dir = data_dir / 'processed'
            processed_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(processed_dir / 'features.csv', index=False)
            
            # Train models
            trainer = ModelTrainer(data_dir=str(data_dir), model_dir=str(model_dir))
            results, best_model_name, best_model = trainer.train_all_models(use_grid_search=False)
            
            # Verify training results
            assert isinstance(results, dict)
            assert len(results) > 0
            assert best_model_name is not None
            assert best_model is not None
            
            # Check that model performance is reasonable
            best_auc = results[best_model_name]['auc_score']
            assert best_auc > 0.6, f"Model AUC too low: {best_auc}"
            
            # Test model evaluation
            evaluator = ModelEvaluator(model_dir=str(model_dir), data_dir=str(data_dir))
            evaluator.load_models()
            
            X_train, X_test, y_train, y_test = evaluator.load_test_data()
            eval_results = evaluator.evaluate_all_models(X_train, X_test, y_train, y_test)
            
            # Verify evaluation results
            assert isinstance(eval_results, dict)
            assert len(eval_results) > 0
            
            for model_name, result in eval_results.items():
                assert 'metrics' in result
                assert 'predictions' in result
                assert result['metrics']['roc_auc'] > 0.5  # Better than random
    
    def test_model_consistency(self):
        """Test that models produce consistent results"""
        # Create deterministic test data
        np.random.seed(123)
        n_samples = 200
        
        test_data = pd.DataFrame({
            'cpu_usage': np.random.uniform(20, 80, n_samples),
            'memory_usage': np.random.uniform(30, 70, n_samples),
            'disk_usage': np.random.uniform(10, 60, n_samples),
            'network_latency_ms': np.random.uniform(20, 150, n_samples),
            'error_count': np.random.poisson(1, n_samples),
            'response_time_ms': np.random.uniform(150, 600, n_samples),
            'active_connections': np.random.poisson(35, n_samples),
            'hour': np.random.randint(0, 24, n_samples),
            'day_of_week': np.random.randint(0, 7, n_samples),
            'is_weekend': np.random.choice([0, 1], n_samples),
            'is_business_hours': np.random.choice([0, 1], n_samples),
            'cpu_memory_product': np.random.uniform(600, 5600, n_samples),
            'resource_pressure': np.random.uniform(20, 70, n_samples),
            'performance_score': np.random.uniform(4, 20, n_samples),
            'failure_within_hour': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
        })
        
        X = test_data.drop('failure_within_hour', axis=1)
        y = test_data['failure_within_hour']
        
        # Train model multiple times with same seed
        model1 = RandomForestClassifier(n_estimators=10, random_state=42)
        model2 = RandomForestClassifier(n_estimators=10, random_state=42)
        
        model1.fit(X, y)
        model2.fit(X, y)
        
        # Predictions should be identical with same random seed
        pred1 = model1.predict_proba(X)
        pred2 = model2.predict_proba(X)
        
        np.testing.assert_array_almost_equal(pred1, pred2, decimal=10)
    
    def test_model_robustness(self):
        """Test model robustness with edge cases"""
        # Create edge case data
        edge_cases = pd.DataFrame({
            'cpu_usage': [0, 100, 50, 50, 50],
            'memory_usage': [0, 100, 50, 50, 50],
            'disk_usage': [0, 100, 50, 50, 50],
            'network_latency_ms': [1, 1000, 100, 100, 100],
            'error_count': [0, 50, 5, 5, 5],
            'response_time_ms': [100, 5000, 300, 300, 300],
            'active_connections': [1, 1000, 50, 50, 50],
            'hour': [0, 23, 12, 12, 12],
            'day_of_week': [0, 6, 3, 3, 3],
            'is_weekend': [0, 1, 0, 0, 0],
            'is_business_hours': [0, 0, 1, 1, 1],
            'cpu_memory_product': [0, 10000, 2500, 2500, 2500],
            'resource_pressure': [0, 100, 50, 50, 50],
            'performance_score': [100, 5, 6, 6, 6],
            'failure_within_hour': [0, 1, 0, 1, 0]
        })
        
        X = edge_cases.drop('failure_within_hour', axis=1)
        y = edge_cases['failure_within_hour']
        
        # Train a simple model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        # Model should handle edge cases without crashing
        predictions = model.predict_proba(X)
        
        # Check that predictions are valid probabilities
        assert predictions.shape == (len(X), 2)
        assert np.all(predictions >= 0)
        assert np.all(predictions <= 1)
        assert np.allclose(predictions.sum(axis=1), 1.0)

def test_model_performance_benchmarks():
    """Test that models meet minimum performance benchmarks"""
    # Create a dataset with clear signal
    np.random.seed(42)
    n_samples = 1000
    
    # Create features with strong predictive signal
    cpu_usage = np.random.uniform(0, 100, n_samples)
    memory_usage = np.random.uniform(0, 100, n_samples)
    error_count = np.random.poisson(2, n_samples)
    
    # Create strong relationship with target
    system_load = (cpu_usage + memory_usage) / 200
    error_impact = np.minimum(error_count / 10, 0.5)
    
    # High system load + errors = high failure probability
    failure_prob = system_load * 0.6 + error_impact * 0.4
    failure_prob = np.clip(failure_prob + np.random.normal(0, 0.1, n_samples), 0.05, 0.95)
    
    failure_within_hour = (np.random.random(n_samples) < failure_prob).astype(int)
    
    # Create training data
    X = pd.DataFrame({
        'cpu_usage': cpu_usage,
        'memory_usage': memory_usage,
        'error_count': error_count,
        'system_load': system_load,
        'error_impact': error_impact
    })
    y = pd.Series(failure_within_hour)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate performance
    from sklearn.metrics import roc_auc_score, accuracy_score
    
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    auc_score = roc_auc_score(y_test, y_pred_proba)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Performance benchmarks
    assert auc_score > 0.75, f"AUC score too low: {auc_score}"
    assert accuracy > 0.70, f"Accuracy too low: {accuracy}"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])