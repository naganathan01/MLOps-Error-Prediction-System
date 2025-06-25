"""
Feature engineering for the error prediction model.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def load_data(self):
        """Load raw data files"""
        print("📂 Loading raw data...")
        
        # Load system metrics
        metrics_file = self.data_dir / "raw" / "system_metrics.csv"
        self.metrics_df = pd.read_csv(metrics_file)
        self.metrics_df['timestamp'] = pd.to_datetime(self.metrics_df['timestamp'])
        
        # Load application logs
        logs_file = self.data_dir / "raw" / "application_logs.csv"
        self.logs_df = pd.read_csv(logs_file)
        self.logs_df['timestamp'] = pd.to_datetime(self.logs_df['timestamp'])
        
        print(f"✅ Loaded {len(self.metrics_df)} metrics and {len(self.logs_df)} logs")
        
    def create_time_features(self, df):
        """Create time-based features"""
        print("⏰ Creating time-based features...")
        
        df = df.copy()
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17) & (df['is_weekend'] == 0)).astype(int)
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
        
        return df
    
    def create_rolling_features(self, df, windows=[5, 15, 30]):
        """Create rolling window features"""
        print("📊 Creating rolling window features...")
        
        df = df.copy()
        df = df.sort_values('timestamp')
        
        # Numeric columns for rolling features
        numeric_cols = ['cpu_usage', 'memory_usage', 'disk_usage', 'network_latency_ms', 
                       'error_count', 'response_time_ms', 'active_connections']
        
        for window in windows:
            for col in numeric_cols:
                # Rolling mean
                df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window=window, min_periods=1).mean()
                
                # Rolling std
                df[f'{col}_rolling_std_{window}'] = df[col].rolling(window=window, min_periods=1).std().fillna(0)
                
                # Rolling max
                df[f'{col}_rolling_max_{window}'] = df[col].rolling(window=window, min_periods=1).max()
                
                # Difference from rolling mean
                df[f'{col}_diff_from_mean_{window}'] = df[col] - df[f'{col}_rolling_mean_{window}']
        
        return df
    
    def create_lag_features(self, df, lags=[1, 2, 3, 5]):
        """Create lag features"""
        print("⏮️ Creating lag features...")
        
        df = df.copy()
        df = df.sort_values('timestamp')
        
        # Key metrics for lag features
        lag_cols = ['cpu_usage', 'memory_usage', 'error_count', 'response_time_ms']
        
        for lag in lags:
            for col in lag_cols:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        return df
    
    def create_aggregated_log_features(self):
        """Create features from application logs"""
        print("📝 Creating log-based features...")
        
        # Aggregate logs by hour
        self.logs_df['hour_bucket'] = self.logs_df['timestamp'].dt.floor('H')
        
        log_features = self.logs_df.groupby('hour_bucket').agg({
            'level': lambda x: (x == 'ERROR').sum(),  # Error count
            'message': 'count',  # Total log count
            'source': 'nunique'  # Unique sources
        }).reset_index()
        
        log_features.columns = ['timestamp', 'error_log_count', 'total_log_count', 'unique_sources']
        
        # Add error rate
        log_features['error_rate'] = log_features['error_log_count'] / log_features['total_log_count']
        log_features['error_rate'] = log_features['error_rate'].fillna(0)
        
        return log_features
    
    def create_anomaly_features(self, df):
        """Create anomaly detection features"""
        print("🔍 Creating anomaly detection features...")
        
        df = df.copy()
        
        # Key metrics for anomaly detection
        anomaly_cols = ['cpu_usage', 'memory_usage', 'response_time_ms']
        
        for col in anomaly_cols:
            # Z-score based anomaly detection
            mean_val = df[col].mean()
            std_val = df[col].std()
            df[f'{col}_zscore'] = (df[col] - mean_val) / std_val
            df[f'{col}_is_anomaly'] = (np.abs(df[f'{col}_zscore']) > 2).astype(int)
            
            # IQR based anomaly detection
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[f'{col}_iqr_anomaly'] = ((df[col] < lower_bound) | (df[col] > upper_bound)).astype(int)
        
        return df
    
    def create_interaction_features(self, df):
        """Create interaction features"""
        print("🔗 Creating interaction features...")
        
        df = df.copy()
        
        # Resource utilization combinations
        df['cpu_memory_product'] = df['cpu_usage'] * df['memory_usage']
        df['resource_pressure'] = (df['cpu_usage'] + df['memory_usage'] + df['disk_usage']) / 3
        
        # Performance indicators
        df['performance_score'] = df['response_time_ms'] / (df['active_connections'] + 1)
        df['error_per_connection'] = df['error_count'] / (df['active_connections'] + 1)
        
        # System stress indicators
        df['system_stress'] = (df['cpu_usage'] > 80).astype(int) + \
                             (df['memory_usage'] > 80).astype(int) + \
                             (df['error_count'] > 5).astype(int)
        
        return df
    
    def process_features(self):
        """Main feature processing pipeline"""
        print("🔧 Starting feature engineering pipeline...")
        
        # Load data
        self.load_data()
        
        # Start with metrics data
        df = self.metrics_df.copy()
        
        # Create time features
        df = self.create_time_features(df)
        
        # Create rolling features
        df = self.create_rolling_features(df)
        
        # Create lag features
        df = self.create_lag_features(df)
        
        # Create anomaly features
        df = self.create_anomaly_features(df)
        
        # Create interaction features
        df = self.create_interaction_features(df)
        
        # Merge with log features
        log_features = self.create_aggregated_log_features()
        df['hour_bucket'] = df['timestamp'].dt.floor('H')
        df = df.merge(log_features, left_on='hour_bucket', right_on='timestamp', 
                     how='left', suffixes=('', '_log'))
        
        # Fill missing log features
        log_cols = ['error_log_count', 'total_log_count', 'unique_sources', 'error_rate']
        for col in log_cols:
            df[col] = df[col].fillna(0)
        
        # Remove temporary columns
        df = df.drop(['hour_bucket', 'timestamp_log'], axis=1, errors='ignore')
        
        # Fill any remaining NaN values
        df = df.fillna(0)
        
        print(f"✅ Feature engineering complete! Created {len(df.columns)} features")
        
        # Save processed data
        processed_dir = self.data_dir / "processed"
        processed_dir.mkdir(exist_ok=True)
        
        output_file = processed_dir / "features.csv"
        df.to_csv(output_file, index=False)
        print(f"💾 Saved processed features to {output_file}")
        
        # Create feature importance summary
        self.create_feature_summary(df)
        
        return df
    
    def create_feature_summary(self, df):
        """Create a summary of features"""
        print("📋 Creating feature summary...")
        
        # Feature categories
        feature_categories = {
            'time_features': [col for col in df.columns if any(time_word in col.lower() 
                             for time_word in ['hour', 'day', 'weekend', 'business', 'night'])],
            'rolling_features': [col for col in df.columns if 'rolling' in col],
            'lag_features': [col for col in df.columns if 'lag' in col],
            'anomaly_features': [col for col in df.columns if any(anom_word in col.lower() 
                                for anom_word in ['zscore', 'anomaly', 'iqr'])],
            'interaction_features': [col for col in df.columns if any(int_word in col.lower() 
                                    for int_word in ['product', 'pressure', 'score', 'stress'])],
            'log_features': [col for col in df.columns if 'log' in col or 'error_rate' in col],
            'original_features': ['cpu_usage', 'memory_usage', 'disk_usage', 'network_latency_ms',
                                'error_count', 'response_time_ms', 'active_connections']
        }
        
        summary = {
            'total_features': len(df.columns),
            'total_samples': len(df),
            'feature_categories': {cat: len(features) for cat, features in feature_categories.items()},
            'target_variable': 'failure_within_hour',
            'missing_values': df.isnull().sum().sum(),
            'feature_list': list(df.columns)
        }
        
        # Save summary
        import json
        summary_file = self.data_dir / "processed" / "feature_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"📊 Feature Summary:")
        print(f"   Total features: {summary['total_features']}")
        print(f"   Total samples: {summary['total_samples']}")
        print(f"   Missing values: {summary['missing_values']}")
        
        for category, count in summary['feature_categories'].items():
            print(f"   {category}: {count} features")

if __name__ == "__main__":
    engineer = FeatureEngineer()
    df = engineer.process_features()
    print("\n🎉 Feature engineering completed successfully!")