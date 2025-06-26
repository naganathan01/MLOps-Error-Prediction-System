"""
Enhanced feature engineering with better predictive features and target engineering.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class EnhancedFeatureEngineer:
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load enhanced raw data files"""
        print("📂 Loading enhanced raw data...")
        
        # Load system metrics
        metrics_file = self.data_dir / "raw" / "system_metrics.csv"
        self.metrics_df = pd.read_csv(metrics_file)
        self.metrics_df['timestamp'] = pd.to_datetime(self.metrics_df['timestamp'])
        
        # Load application logs
        logs_file = self.data_dir / "raw" / "application_logs.csv"
        self.logs_df = pd.read_csv(logs_file)
        self.logs_df['timestamp'] = pd.to_datetime(self.logs_df['timestamp'])
        
        print(f"✅ Loaded {len(self.metrics_df)} metrics and {len(self.logs_df)} logs")
        
    def create_enhanced_time_features(self, df):
        """Create enhanced time-based features"""
        print("⏰ Creating enhanced time-based features...")
        
        df = df.copy()
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['day_of_month'] = df['timestamp'].dt.day
        
        # Enhanced time features
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17) & (df['is_weekend'] == 0)).astype(int)
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
        df['is_peak_hours'] = ((df['hour'] >= 10) & (df['hour'] <= 12) | (df['hour'] >= 14) & (df['hour'] <= 16)).astype(int)
        df['is_off_hours'] = ((df['hour'] >= 18) | (df['hour'] <= 8)).astype(int)
        
        # Cyclical encoding for time features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        return df
    
    def create_advanced_rolling_features(self, df, windows=[3, 5, 10, 15, 30]):
        """Create advanced rolling window features with trend detection"""
        print("📊 Creating advanced rolling window features...")
        
        df = df.copy()
        df = df.sort_values('timestamp')
        
        # Core metrics for rolling features
        numeric_cols = ['cpu_usage', 'memory_usage', 'disk_usage', 'network_latency_ms', 
                       'error_count', 'response_time_ms', 'active_connections']
        
        for window in windows:
            for col in numeric_cols:
                # Basic rolling statistics
                df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window=window, min_periods=1).mean()
                df[f'{col}_rolling_std_{window}'] = df[col].rolling(window=window, min_periods=1).std().fillna(0)
                df[f'{col}_rolling_max_{window}'] = df[col].rolling(window=window, min_periods=1).max()
                df[f'{col}_rolling_min_{window}'] = df[col].rolling(window=window, min_periods=1).min()
                df[f'{col}_rolling_median_{window}'] = df[col].rolling(window=window, min_periods=1).median()
                
                # Trend and change features
                df[f'{col}_rolling_slope_{window}'] = df[col].rolling(window=window, min_periods=2).apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
                )
                
                # Rate of change features
                df[f'{col}_pct_change_{window}'] = df[col].pct_change(window).fillna(0)
                df[f'{col}_diff_from_mean_{window}'] = df[col] - df[f'{col}_rolling_mean_{window}']
                df[f'{col}_zscore_{window}'] = (df[col] - df[f'{col}_rolling_mean_{window}']) / (df[f'{col}_rolling_std_{window}'] + 1e-8)
                
                # Volatility and stability measures
                df[f'{col}_coefficient_variation_{window}'] = df[f'{col}_rolling_std_{window}'] / (df[f'{col}_rolling_mean_{window}'] + 1e-8)
                df[f'{col}_range_ratio_{window}'] = (df[f'{col}_rolling_max_{window}'] - df[f'{col}_rolling_min_{window}']) / (df[f'{col}_rolling_mean_{window}'] + 1e-8)
        
        return df
    
    def create_lag_and_lead_features(self, df, lags=[1, 2, 3, 5, 10], leads=[1, 2, 3]):
        """Create lag and lead features for time series patterns"""
        print("⏮️ Creating lag and lead features...")
        
        df = df.copy()
        df = df.sort_values('timestamp')
        
        # Key metrics for lag/lead features
        lag_cols = ['cpu_usage', 'memory_usage', 'error_count', 'response_time_ms', 'active_connections']
        
        for col in lag_cols:
            # Lag features (past values)
            for lag in lags:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
                
                # Lag differences
                df[f'{col}_lag_diff_{lag}'] = df[col] - df[f'{col}_lag_{lag}']
                df[f'{col}_lag_ratio_{lag}'] = df[col] / (df[f'{col}_lag_{lag}'] + 1e-8)
            
            # Lead features (future values) - only for non-target features
            if col != 'failure_within_hour':
                for lead in leads:
                    df[f'{col}_lead_{lead}'] = df[col].shift(-lead)
                    df[f'{col}_lead_diff_{lead}'] = df[f'{col}_lead_{lead}'] - df[col]
        
        return df
    
    def create_enhanced_log_features(self):
        """Create enhanced features from application logs"""
        print("📝 Creating enhanced log-based features...")
        
        # Aggregate logs by 5-minute windows to match metrics frequency
        self.logs_df['time_bucket'] = self.logs_df['timestamp'].dt.floor('5T')
        
        # Basic log aggregations
        log_features = self.logs_df.groupby('time_bucket').agg({
            'level': [
                lambda x: (x == 'ERROR').sum(),
                lambda x: (x == 'WARN').sum(),
                lambda x: (x == 'INFO').sum(),
                lambda x: (x == 'DEBUG').sum(),
                'count'
            ],
            'source': 'nunique',
            'user_id': 'nunique',
            'session_id': 'nunique'
        }).reset_index()
        
        # Flatten column names
        log_features.columns = ['timestamp', 'error_log_count', 'warn_log_count', 'info_log_count', 
                               'debug_log_count', 'total_log_count', 'unique_sources', 
                               'unique_users', 'unique_sessions']
        
        # Enhanced log features
        log_features['error_rate'] = log_features['error_log_count'] / (log_features['total_log_count'] + 1)
        log_features['warn_rate'] = log_features['warn_log_count'] / (log_features['total_log_count'] + 1)
        log_features['critical_rate'] = (log_features['error_log_count'] + log_features['warn_log_count']) / (log_features['total_log_count'] + 1)
        
        # Log diversity features
        log_features['source_diversity'] = log_features['unique_sources'] / (log_features['total_log_count'] + 1)
        log_features['user_activity'] = log_features['unique_users'] / (log_features['unique_sessions'] + 1)
        
        # Log message content analysis (simple keyword-based)
        keyword_features = self._analyze_log_keywords()
        log_features = log_features.merge(keyword_features, on='timestamp', how='left')
        
        return log_features.fillna(0)
    
    def _analyze_log_keywords(self):
        """Analyze log messages for failure-related keywords"""
        # Define failure-related keywords
        failure_keywords = {
            'memory_keywords': ['memory', 'heap', 'oom', 'outofmemory', 'allocation'],
            'performance_keywords': ['timeout', 'slow', 'performance', 'latency', 'bottleneck'],
            'error_keywords': ['exception', 'failed', 'error', 'critical', 'fatal'],
            'network_keywords': ['connection', 'network', 'socket', 'unreachable', 'timeout'],
            'disk_keywords': ['disk', 'space', 'full', 'quota', 'storage'],
            'database_keywords': ['database', 'query', 'deadlock', 'connection', 'pool']
        }
        
        # Aggregate by time bucket
        self.logs_df['time_bucket'] = self.logs_df['timestamp'].dt.floor('5T')
        keyword_features = []
        
        for time_bucket, group in self.logs_df.groupby('time_bucket'):
            messages = ' '.join(group['message'].astype(str).str.lower())
            
            feature_row = {'timestamp': time_bucket}
            
            for category, keywords in failure_keywords.items():
                keyword_count = sum(messages.count(keyword) for keyword in keywords)
                feature_row[f'{category}_count'] = keyword_count
                feature_row[f'{category}_ratio'] = keyword_count / len(group) if len(group) > 0 else 0
            
            keyword_features.append(feature_row)
        
        return pd.DataFrame(keyword_features)
    
    def create_advanced_interaction_features(self, df):
        """Create advanced interaction and composite features"""
        print("🔗 Creating advanced interaction features...")
        
        df = df.copy()
        
        # Resource utilization combinations
        df['cpu_memory_product'] = df['cpu_usage'] * df['memory_usage']
        df['cpu_memory_max'] = np.maximum(df['cpu_usage'], df['memory_usage'])
        df['resource_pressure'] = (df['cpu_usage'] + df['memory_usage'] + df['disk_usage']) / 3
        df['resource_imbalance'] = np.std([df['cpu_usage'], df['memory_usage'], df['disk_usage']], axis=0)
        
        # Performance indicators
        df['performance_score'] = df['response_time_ms'] / (df['active_connections'] + 1)
        df['throughput_estimate'] = df['active_connections'] / (df['response_time_ms'] / 1000 + 1)
        df['error_per_connection'] = df['error_count'] / (df['active_connections'] + 1)
        df['latency_response_ratio'] = df['network_latency_ms'] / (df['response_time_ms'] + 1)
        
        # System stress indicators (multi-level)
        df['cpu_stress_level'] = pd.cut(df['cpu_usage'], bins=[0, 50, 70, 85, 100], labels=[0, 1, 2, 3]).astype(int)
        df['memory_stress_level'] = pd.cut(df['memory_usage'], bins=[0, 60, 80, 90, 100], labels=[0, 1, 2, 3]).astype(int)
        df['disk_stress_level'] = pd.cut(df['disk_usage'], bins=[0, 70, 85, 95, 100], labels=[0, 1, 2, 3]).astype(int)
        
        df['total_stress_score'] = df['cpu_stress_level'] + df['memory_stress_level'] + df['disk_stress_level']
        df['max_stress_level'] = np.maximum.reduce([df['cpu_stress_level'], df['memory_stress_level'], df['disk_stress_level']])
        
        # Critical thresholds (binary indicators)
        df['cpu_critical'] = (df['cpu_usage'] > 90).astype(int)
        df['memory_critical'] = (df['memory_usage'] > 85).astype(int)
        df['disk_critical'] = (df['disk_usage'] > 95).astype(int)
        df['response_critical'] = (df['response_time_ms'] > 2000).astype(int)
        df['error_critical'] = (df['error_count'] > 10).astype(int)
        df['latency_critical'] = (df['network_latency_ms'] > 500).astype(int)
        
        df['critical_indicators_count'] = (df['cpu_critical'] + df['memory_critical'] + 
                                         df['disk_critical'] + df['response_critical'] + 
                                         df['error_critical'] + df['latency_critical'])
        
        # Workload characterization
        df['high_cpu_low_memory'] = ((df['cpu_usage'] > 80) & (df['memory_usage'] < 60)).astype(int)
        df['high_memory_low_cpu'] = ((df['memory_usage'] > 80) & (df['cpu_usage'] < 60)).astype(int)
        df['balanced_high_load'] = ((df['cpu_usage'] > 70) & (df['memory_usage'] > 70)).astype(int)
        df['io_bound_workload'] = ((df['disk_usage'] > 80) | (df['network_latency_ms'] > 200)).astype(int)
        
        return df
    
    def create_enhanced_anomaly_features(self, df):
        """Create enhanced anomaly detection features"""
        print("🔍 Creating enhanced anomaly detection features...")
        
        df = df.copy()
        
        # Core metrics for anomaly detection
        anomaly_cols = ['cpu_usage', 'memory_usage', 'response_time_ms', 'error_count', 'network_latency_ms']
        
        for col in anomaly_cols:
            # Statistical anomaly detection
            mean_val = df[col].mean()
            std_val = df[col].std()
            median_val = df[col].median()
            mad_val = df[col].mad()  # Median Absolute Deviation
            
            # Z-score based anomaly detection
            df[f'{col}_zscore'] = (df[col] - mean_val) / (std_val + 1e-8)
            df[f'{col}_zscore_abs'] = np.abs(df[f'{col}_zscore'])
            df[f'{col}_is_anomaly_zscore'] = (df[f'{col}_zscore_abs'] > 2.5).astype(int)
            df[f'{col}_is_extreme_anomaly'] = (df[f'{col}_zscore_abs'] > 3.5).astype(int)
            
            # Modified Z-score (more robust)
            df[f'{col}_modified_zscore'] = 0.6745 * (df[col] - median_val) / (mad_val + 1e-8)
            df[f'{col}_is_anomaly_modified'] = (np.abs(df[f'{col}_modified_zscore']) > 3.5).astype(int)
            
            # IQR based anomaly detection
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            df[f'{col}_iqr_anomaly'] = ((df[col] < lower_bound) | (df[col] > upper_bound)).astype(int)
            df[f'{col}_iqr_distance'] = np.minimum(df[col] - lower_bound, upper_bound - df[col])
            
            # Percentile-based features
            df[f'{col}_percentile'] = df[col].rank(pct=True)
            df[f'{col}_is_top_5pct'] = (df[f'{col}_percentile'] > 0.95).astype(int)
            df[f'{col}_is_bottom_5pct'] = (df[f'{col}_percentile'] < 0.05).astype(int)
        
        # Composite anomaly score
        anomaly_columns = [col for col in df.columns if '_is_anomaly_' in col or '_iqr_anomaly' in col]
        df['total_anomaly_score'] = df[anomaly_columns].sum(axis=1)
        df['is_multi_anomaly'] = (df['total_anomaly_score'] >= 3).astype(int)
        
        return df
    
    def create_target_engineering_features(self, df):
        """Create features that help predict failures better"""
        print("🎯 Creating target engineering features...")
        
        df = df.copy()
        df = df.sort_values('timestamp')
        
        # Create multiple prediction horizons
        for horizon in [1, 2, 3, 6, 12]:  # 5min, 10min, 15min, 30min, 1hour
            df[f'failure_within_{horizon*5}min'] = df['failure_within_hour'].shift(-horizon).fillna(0)
        
        # Failure probability indicators
        df['failure_risk_score'] = (
            df['cpu_critical'] * 0.3 +
            df['memory_critical'] * 0.3 +
            df['critical_indicators_count'] * 0.1 +
            df['total_stress_score'] * 0.05 +
            df['is_multi_anomaly'] * 0.2 +
            (df['error_count'] > 5).astype(int) * 0.05
        )
        
        # Early warning indicators
        df['early_warning_cpu'] = (df['cpu_usage'] > 75).astype(int)
        df['early_warning_memory'] = (df['memory_usage'] > 75).astype(int)
        df['early_warning_response'] = (df['response_time_ms'] > 1000).astype(int)
        df['early_warning_errors'] = (df['error_count'] > 3).astype(int)
        
        df['early_warning_count'] = (df['early_warning_cpu'] + df['early_warning_memory'] + 
                                    df['early_warning_response'] + df['early_warning_errors'])
        
        # Failure scenario indicators (based on patterns)
        df['memory_leak_pattern'] = ((df['memory_usage'] > 80) & 
                                    (df['memory_usage_rolling_slope_10'] > 0.5)).astype(int)
        df['cpu_spike_pattern'] = ((df['cpu_usage'] > 85) & 
                                  (df['cpu_usage_pct_change_3'] > 0.2)).astype(int)
        df['cascade_failure_pattern'] = ((df['total_stress_score'] >= 6) & 
                                        (df['critical_indicators_count'] >= 3)).astype(int)
        
        return df
    
    def process_enhanced_features(self):
        """Main enhanced feature processing pipeline"""
        print("🔧 Starting enhanced feature engineering pipeline...")
        
        # Load data
        self.load_data()
        
        # Start with metrics data
        df = self.metrics_df.copy()
        
        # Create enhanced time features
        df = self.create_enhanced_time_features(df)
        
        # Create advanced rolling features
        df = self.create_advanced_rolling_features(df)
        
        # Create lag and lead features
        df = self.create_lag_and_lead_features(df)
        
        # Create enhanced interaction features
        df = self.create_advanced_interaction_features(df)
        
        # Create enhanced anomaly features
        df = self.create_enhanced_anomaly_features(df)
        
        # Create target engineering features
        df = self.create_target_engineering_features(df)
        
        # Merge with enhanced log features
        log_features = self.create_enhanced_log_features()
        df['time_bucket'] = df['timestamp'].dt.floor('5T')
        df = df.merge(log_features, left_on='time_bucket', right_on='timestamp', 
                     how='left', suffixes=('', '_log'))
        
        # Fill missing log features
        log_cols = [col for col in df.columns if 'log' in col or 'keyword' in col]
        for col in log_cols:
            df[col] = df[col].fillna(0)
        
        # Remove temporary columns
        df = df.drop(['time_bucket', 'timestamp_log'], axis=1, errors='ignore')
        
        # Fill any remaining NaN values with appropriate defaults
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        
        # Remove infinite values
        df = df.replace([np.inf, -np.inf], 0)
        
        print(f"✅ Enhanced feature engineering complete! Created {len(df.columns)} features")
        print(f"📊 Dataset shape: {df.shape}")
        print(f"🎯 Failure rate: {df['failure_within_hour'].mean():.2%}")
        
        # Save processed data
        processed_dir = self.data_dir / "processed"
        processed_dir.mkdir(exist_ok=True)
        
        output_file = processed_dir / "enhanced_features.csv"
        df.to_csv(output_file, index=False)
        print(f"💾 Saved enhanced features to {output_file}")
        
        # Create enhanced feature summary
        self.create_enhanced_feature_summary(df)
        
        return df
    
    def create_enhanced_feature_summary(self, df):
        """Create a comprehensive summary of enhanced features"""
        print("📋 Creating enhanced feature summary...")
        
        # Enhanced feature categories
        feature_categories = {
            'basic_metrics': ['cpu_usage', 'memory_usage', 'disk_usage', 'network_latency_ms',
                            'error_count', 'response_time_ms', 'active_connections'],
            'time_features': [col for col in df.columns if any(time_word in col.lower() 
                             for time_word in ['hour', 'day', 'weekend', 'business', 'night', 'peak', 'sin', 'cos'])],
            'rolling_features': [col for col in df.columns if 'rolling' in col],
            'lag_lead_features': [col for col in df.columns if 'lag' in col or 'lead' in col],
            'anomaly_features': [col for col in df.columns if any(anom_word in col.lower() 
                                for anom_word in ['zscore', 'anomaly', 'iqr', 'percentile'])],
            'interaction_features': [col for col in df.columns if any(int_word in col.lower() 
                                    for int_word in ['product', 'pressure', 'score', 'stress', 'ratio', 'critical'])],
            'log_features': [col for col in df.columns if 'log' in col or 'keyword' in col],
            'target_features': [col for col in df.columns if 'failure_within' in col or 'warning' in col or 'pattern' in col]
        }
        
        # Calculate feature importance proxies
        target_correlations = {}
        if 'failure_within_hour' in df.columns:
            for col in df.select_dtypes(include=[np.number]).columns:
                if col != 'failure_within_hour':
                    corr = df[col].corr(df['failure_within_hour'])
                    if not pd.isna(corr):
                        target_correlations[col] = abs(corr)
        
        # Top correlated features
        top_features = sorted(target_correlations.items(), key=lambda x: x[1], reverse=True)[:20]
        
        summary = {
            'total_features': len(df.columns),
            'total_samples': len(df),
            'feature_categories': {cat: len(features) for cat, features in feature_categories.items()},
            'target_variable': 'failure_within_hour',
            'failure_rate': float(df['failure_within_hour'].mean()),
            'missing_values': int(df.isnull().sum().sum()),
            'top_correlated_features': top_features[:10],
            'feature_quality_metrics': {
                'variance_threshold_passed': len([col for col in df.select_dtypes(include=[np.number]).columns 
                                                if df[col].var() > 0.01]),
                'correlation_pairs_high': len([(i, j) for i in df.select_dtypes(include=[np.number]).columns 
                                             for j in df.select_dtypes(include=[np.number]).columns 
                                             if i < j and abs(df[i].corr(df[j])) > 0.9]),
            },
            'enhancement_features': [
                'Multiple prediction horizons',
                'Advanced rolling statistics with trends',
                'Lag and lead features',
                'Enhanced anomaly detection',
                'Composite stress indicators',
                'Log keyword analysis',
                'Failure pattern detection'
            ]
        }
        
        # Save enhanced summary
        import json
        summary_file = self.data_dir / "processed" / "enhanced_feature_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"📊 Enhanced Feature Summary:")
        print(f"   Total features: {summary['total_features']}")
        print(f"   Total samples: {summary['total_samples']}")
        print(f"   Failure rate: {summary['failure_rate']:.2%}")
        print(f"   Missing values: {summary['missing_values']}")
        
        for category, count in summary['feature_categories'].items():
            print(f"   {category}: {count} features")
        
        print(f"\n🏆 Top 5 Correlated Features:")
        for feature, corr in summary['top_correlated_features'][:5]:
            print(f"   {feature}: {corr:.4f}")

if __name__ == "__main__":
    engineer = EnhancedFeatureEngineer()
    df = engineer.process_enhanced_features()
    print("\n🎉 Enhanced feature engineering completed successfully!")