"""
Tests for the data pipeline components.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.data_generator import SystemDataGenerator
from src.features.feature_engineering import FeatureEngineer
from src.data.data_ingestion import DataIngestion
from src.data.preprocessing import DataPreprocessor

class TestDataGenerator:
    
    def test_system_data_generation(self):
        """Test system metrics data generation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            generator = SystemDataGenerator(output_dir=temp_dir)
            
            # Generate small dataset for testing
            metrics_df = generator.generate_system_metrics(n_days=2, samples_per_hour=4)
            
            # Check basic properties
            assert isinstance(metrics_df, pd.DataFrame)
            assert len(metrics_df) > 0
            assert 'timestamp' in metrics_df.columns
            assert 'cpu_usage' in metrics_df.columns
            assert 'failure_within_hour' in metrics_df.columns
            
            # Check data ranges
            assert metrics_df['cpu_usage'].min() >= 0
            assert metrics_df['cpu_usage'].max() <= 100
            assert metrics_df['memory_usage'].min() >= 0
            assert metrics_df['memory_usage'].max() <= 100
            
            # Check target variable
            assert set(metrics_df['failure_within_hour'].unique()).issubset({0, 1})
    
    def test_log_data_generation(self):
        """Test application logs data generation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            generator = SystemDataGenerator(output_dir=temp_dir)
            
            # Generate small dataset for testing
            logs_df = generator.generate_application_logs(n_days=1, logs_per_hour=10)
            
            # Check basic properties
            assert isinstance(logs_df, pd.DataFrame)
            assert len(logs_df) > 0
            assert 'timestamp' in logs_df.columns
            assert 'level' in logs_df.columns
            assert 'message' in logs_df.columns
            
            # Check log levels
            valid_levels = {'INFO', 'WARN', 'ERROR', 'DEBUG'}
            assert set(logs_df['level'].unique()).issubset(valid_levels)
    
    def test_complete_data_generation(self):
        """Test complete data generation pipeline"""
        with tempfile.TemporaryDirectory() as temp_dir:
            generator = SystemDataGenerator(output_dir=temp_dir)
            
            metrics_df, logs_df = generator.generate_all_data(n_days=1)
            
            # Check that both datasets are generated
            assert isinstance(metrics_df, pd.DataFrame)
            assert isinstance(logs_df, pd.DataFrame)
            assert len(metrics_df) > 0
            assert len(logs_df) > 0
            
            # Check that files are saved
            metrics_file = Path(temp_dir) / "system_metrics.csv"
            logs_file = Path(temp_dir) / "application_logs.csv"
            summary_file = Path(temp_dir) / "data_summary.json"
            
            assert metrics_file.exists()
            assert logs_file.exists()
            assert summary_file.exists()

class TestFeatureEngineering:
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        # Create temporary directory with sample data
        temp_dir = tempfile.mkdtemp()
        
        # Generate sample metrics data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='5min')
        
        np.random.seed(42)
        metrics_data = {
            'timestamp': dates,
            'cpu_usage': np.random.uniform(20, 90, 100),
            'memory_usage': np.random.uniform(30, 85, 100),
            'disk_usage': np.random.uniform(10, 60, 100),
            'network_latency_ms': np.random.uniform(10, 200, 100),
            'error_count': np.random.poisson(2, 100),
            'response_time_ms': np.random.uniform(100, 800, 100),
            'active_connections': np.random.poisson(50, 100),
            'system_state': np.random.choice(['normal', 'warning', 'critical'], 100),
            'failure_within_hour': np.random.choice([0, 1], 100, p=[0.8, 0.2]),
            'hour': dates.hour,
            'day_of_week': dates.dayofweek,
            'is_weekend': (dates.dayofweek >= 5).astype(int)
        }
        
        metrics_df = pd.DataFrame(metrics_data)
        
        # Generate sample logs data
        log_data = {
            'timestamp': dates[:50],
            'level': np.random.choice(['INFO', 'WARN', 'ERROR', 'DEBUG'], 50, p=[0.7, 0.15, 0.1, 0.05]),
            'message': ['Test message'] * 50,
            'source': np.random.choice(['API', 'Database', 'Auth'], 50),
            'user_id': [f'user_{i}' for i in range(50)],
            'session_id': [f'session_{i}' for i in range(50)]
        }
        
        logs_df = pd.DataFrame(log_data)
        
        # Save to temporary files
        raw_dir = Path(temp_dir) / 'raw'
        raw_dir.mkdir(parents=True, exist_ok=True)
        
        metrics_df.to_csv(raw_dir / 'system_metrics.csv', index=False)
        logs_df.to_csv(raw_dir / 'application_logs.csv', index=False)
        
        return temp_dir
    
    def test_feature_creation(self, sample_data):
        """Test basic feature creation"""
        engineer = FeatureEngineer(data_dir=sample_data)
        
        # Load data
        engineer.load_data()
        
        assert hasattr(engineer, 'metrics_df')
        assert hasattr(engineer, 'logs_df')
        assert len(engineer.metrics_df) > 0
        assert len(engineer.logs_df) > 0
    
    def test_time_features(self, sample_data):
        """Test time-based feature creation"""
        engineer = FeatureEngineer(data_dir=sample_data)
        engineer.load_data()
        
        df = engineer.metrics_df.copy()
        df_with_time = engineer.create_time_features(df)
        
        # Check that time features are created
        expected_time_features = ['hour', 'day_of_week', 'is_weekend', 'is_business_hours', 'is_night']
        for feature in expected_time_features:
            assert feature in df_with_time.columns
        
        # Check value ranges
        assert df_with_time['hour'].min() >= 0
        assert df_with_time['hour'].max() <= 23
        assert df_with_time['day_of_week'].min() >= 0
        assert df_with_time['day_of_week'].max() <= 6
        assert set(df_with_time['is_weekend'].unique()).issubset({0, 1})
    
    def test_rolling_features(self, sample_data):
        """Test rolling window feature creation"""
        engineer = FeatureEngineer(data_dir=sample_data)
        engineer.load_data()
        
        df = engineer.metrics_df.copy()
        df_with_rolling = engineer.create_rolling_features(df, windows=[5, 10])
        
        # Check that rolling features are created
        rolling_features = [col for col in df_with_rolling.columns if 'rolling' in col]
        assert len(rolling_features) > 0
        
        # Check for specific rolling features
        assert 'cpu_usage_rolling_mean_5' in df_with_rolling.columns
        assert 'memory_usage_rolling_std_10' in df_with_rolling.columns
    
    def test_lag_features(self, sample_data):
        """Test lag feature creation"""
        engineer = FeatureEngineer(data_dir=sample_data)
        engineer.load_data()
        
        df = engineer.metrics_df.copy()
        df_with_lags = engineer.create_lag_features(df, lags=[1, 2])
        
        # Check that lag features are created
        lag_features = [col for col in df_with_lags.columns if 'lag' in col]
        assert len(lag_features) > 0
        
        # Check for specific lag features
        assert 'cpu_usage_lag_1' in df_with_lags.columns
        assert 'memory_usage_lag_2' in df_with_lags.columns
    
    def test_anomaly_features(self, sample_data):
        """Test anomaly detection features"""
        engineer = FeatureEngineer(data_dir=sample_data)
        engineer.load_data()
        
        df = engineer.metrics_df.copy()
        df_with_anomalies = engineer.create_anomaly_features(df)
        
        # Check that anomaly features are created
        anomaly_features = [col for col in df_with_anomalies.columns if 'anomaly' in col or 'zscore' in col]
        assert len(anomaly_features) > 0
        
        # Check for specific anomaly features
        assert 'cpu_usage_zscore' in df_with_anomalies.columns
        assert 'cpu_usage_is_anomaly' in df_with_anomalies.columns
    
    def test_interaction_features(self, sample_data):
        """Test interaction feature creation"""
        engineer = FeatureEngineer(data_dir=sample_data)
        engineer.load_data()
        
        df = engineer.metrics_df.copy()
        df_with_interactions = engineer.create_interaction_features(df)
        
        # Check that interaction features are created
        expected_interactions = ['cpu_memory_product', 'resource_pressure', 'performance_score']
        for feature in expected_interactions:
            assert feature in df_with_interactions.columns
    
    def test_log_aggregation(self, sample_data):
        """Test log aggregation features"""
        engineer = FeatureEngineer(data_dir=sample_data)
        engineer.load_data()
        
        log_features = engineer.create_aggregated_log_features()
        
        # Check that log features are created
        expected_log_features = ['error_log_count', 'total_log_count', 'unique_sources', 'error_rate']
        for feature in expected_log_features:
            assert feature in log_features.columns
        
        assert len(log_features) > 0
    
    def test_complete_feature_pipeline(self, sample_data):
        """Test complete feature engineering pipeline"""
        engineer = FeatureEngineer(data_dir=sample_data)
        
        # Process features
        processed_df = engineer.process_features()
        
        # Check that processed data is returned
        assert isinstance(processed_df, pd.DataFrame)
        assert len(processed_df) > 0
        
        # Check that various feature types are present
        feature_types = {
            'time': [col for col in processed_df.columns if any(t in col for t in ['hour', 'day', 'weekend'])],
            'rolling': [col for col in processed_df.columns if 'rolling' in col],
            'lag': [col for col in processed_df.columns if 'lag' in col],
            'anomaly': [col for col in processed_df.columns if 'anomaly' in col or 'zscore' in col],
            'interaction': [col for col in processed_df.columns if any(i in col for i in ['product', 'pressure', 'score'])]
        }
        
        for feature_type, features in feature_types.items():
            assert len(features) > 0, f"No {feature_type} features found"

class TestDataIngestion:
    
    def test_csv_ingestion(self):
        """Test CSV data ingestion"""
        # Create sample CSV data
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_data = pd.DataFrame({
                'timestamp': pd.date_range('2024-01-01', periods=10, freq='H'),
                'cpu_usage': np.random.uniform(20, 80, 10),
                'memory_usage': np.random.uniform(30, 70, 10),
                'error_count': np.random.randint(0, 5, 10)
            })
            sample_data.to_csv(f.name, index=False)
            csv_file = f.name
        
        try:
            ingestion = DataIngestion()
            df = ingestion.ingest_csv_data(csv_file)
            
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 10
            assert 'timestamp' in df.columns
            assert 'cpu_usage' in df.columns
            
        finally:
            os.unlink(csv_file)
    
    def test_data_validation(self):
        """Test data validation functionality"""
        ingestion = DataIngestion()
        
        # Create sample data with issues
        sample_data = pd.DataFrame({
            'cpu_usage': [50, 150, -10, 75, 80],  # Out of range values
            'memory_usage': [60, 70, None, 65, 85],  # Missing value
            'error_count': [1, 2, 3, 4, 5]
        })
        
        validated_df = ingestion.validate_data(sample_data, data_type="metrics")
        
        # Check that validation was applied
        assert validated_df['cpu_usage'].min() >= 0
        assert validated_df['cpu_usage'].max() <= 100
        assert validated_df['memory_usage'].isnull().sum() == 0  # Missing values filled
    
    def test_data_quality_detection(self):
        """Test data quality issue detection"""
        ingestion = DataIngestion()
        
        # Create data with quality issues
        sample_data = pd.DataFrame({
            'feature1': [1, 2, None, None, None],  # High missing rate
            'feature2': [10, 10, 10, 10, 10],  # No variance
            'feature3': [1, 2, 3, 100, 5],  # Outlier
            'timestamp': pd.date_range('2024-01-01', periods=5, freq='H')
        })
        
        issues = ingestion.detect_data_quality_issues(sample_data)
        
        # Check that issues are detected
        assert len(issues) > 0
        issue_types = [issue['type'] for issue in issues]
        assert 'high_missing_rate' in issue_types

class TestDataPreprocessor:
    
    def test_missing_value_handling(self):
        """Test missing value handling"""
        preprocessor = DataPreprocessor()
        
        # Create data with missing values
        sample_data = pd.DataFrame({
            'numeric_col': [1, 2, None, 4, 5],
            'categorical_col': ['A', 'B', None, 'A', 'B'],
            'target': [0, 1, 0, 1, 0]
        })
        
        cleaned_data = preprocessor.handle_missing_values(sample_data, strategy='adaptive')
        
        # Check that missing values are handled
        assert cleaned_data.isnull().sum().sum() == 0
    
    def test_outlier_detection(self):
        """Test outlier detection and handling"""
        preprocessor = DataPreprocessor()
        
        # Create data with outliers
        np.random.seed(42)
        normal_data = np.random.normal(50, 10, 95)
        outliers = [150, 200, -50, -100, 180]  # Obvious outliers
        sample_data = pd.DataFrame({
            'feature': np.concatenate([normal_data, outliers])
        })
        
        cleaned_data, outlier_summary = preprocessor.detect_and_handle_outliers(
            sample_data, method='iqr', threshold=1.5
        )
        
        # Check that outliers were detected
        assert 'feature' in outlier_summary
        assert outlier_summary['feature']['count'] > 0
    
    def test_normalization(self):
        """Test data normalization"""
        preprocessor = DataPreprocessor()
        
        sample_data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [100, 200, 300, 400, 500],
            'target': [0, 1, 0, 1, 0]
        })
        
        normalized_data = preprocessor.normalize_data(
            sample_data, method='standard', exclude_columns=['target']
        )
        
        # Check that normalization was applied
        # Standard scaling should result in mean ≈ 0 and std ≈ 1
        assert abs(normalized_data['feature1'].mean()) < 0.01
        assert abs(normalized_data['feature1'].std() - 1.0) < 0.01
        
        # Target should remain unchanged
        assert normalized_data['target'].equals(sample_data['target'])

def test_integration_data_pipeline():
    """Test integration of data pipeline components"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Generate data
        generator = SystemDataGenerator(output_dir=temp_dir)
        metrics_df, logs_df = generator.generate_all_data(n_days=2)
        
        # Process features
        engineer = FeatureEngineer(data_dir=temp_dir)
        processed_df = engineer.process_features()
        
        # Check that the pipeline produces valid results
        assert isinstance(processed_df, pd.DataFrame)
        assert len(processed_df) > 0
        assert 'failure_within_hour' in processed_df.columns
        
        # Check that various feature types are present
        has_time_features = any('hour' in col or 'day' in col for col in processed_df.columns)
        has_rolling_features = any('rolling' in col for col in processed_df.columns)
        has_interaction_features = any('product' in col or 'pressure' in col for col in processed_df.columns)
        
        assert has_time_features, "Time features missing"
        assert has_rolling_features, "Rolling features missing"
        assert has_interaction_features, "Interaction features missing"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])