"""
Data preprocessing module for cleaning and preparing data for machine learning.
Handles missing values, outliers, normalization, and data transformations.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
import logging
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.scalers = {}
        self.imputers = {}
        self.preprocessing_config = {}
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load data from file"""
        logger.info(f"📂 Loading data from {file_path}")
        
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        if file_path.suffix == '.csv':
            df = pd.read_csv(file_path)
        elif file_path.suffix == '.json':
            df = pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        # Convert timestamp if present
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        logger.info(f"✅ Loaded {len(df)} records with {len(df.columns)} columns")
        return df
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'adaptive') -> pd.DataFrame:
        """Handle missing values with various strategies"""
        logger.info(f"🔧 Handling missing values with {strategy} strategy...")
        
        df_clean = df.copy()
        
        # Get missing value statistics
        missing_stats = df_clean.isnull().sum()
        missing_columns = missing_stats[missing_stats > 0]
        
        if len(missing_columns) == 0:
            logger.info("✅ No missing values found")
            return df_clean
        
        logger.info(f"Found missing values in {len(missing_columns)} columns")
        
        if strategy == 'adaptive':
            # Use different strategies based on missing percentage and data type
            for column, missing_count in missing_columns.items():
                missing_rate = missing_count / len(df_clean)
                
                if missing_rate > 0.5:
                    # Drop columns with >50% missing
                    logger.info(f"   Dropping {column} (>{missing_rate:.1%} missing)")
                    df_clean = df_clean.drop(column, axis=1)
                    
                elif df_clean[column].dtype in ['object', 'category']:
                    # Mode for categorical
                    mode_value = df_clean[column].mode().iloc[0] if not df_clean[column].mode().empty else 'unknown'
                    df_clean[column] = df_clean[column].fillna(mode_value)
                    logger.info(f"   Filled {column} with mode: {mode_value}")
                    
                elif missing_rate < 0.1:
                    # Mean/median for low missing numeric
                    if df_clean[column].skew() > 1:  # Highly skewed
                        fill_value = df_clean[column].median()
                        df_clean[column] = df_clean[column].fillna(fill_value)
                        logger.info(f"   Filled {column} with median: {fill_value:.2f}")
                    else:
                        fill_value = df_clean[column].mean()
                        df_clean[column] = df_clean[column].fillna(fill_value)
                        logger.info(f"   Filled {column} with mean: {fill_value:.2f}")
                else:
                    # KNN imputation for moderate missing numeric
                    imputer = KNNImputer(n_neighbors=5)
                    df_clean[column] = imputer.fit_transform(df_clean[[column]]).ravel()
                    self.imputers[column] = imputer
                    logger.info(f"   Filled {column} with KNN imputation")
        
        elif strategy == 'mean':
            numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
            imputer = SimpleImputer(strategy='mean')
            df_clean[numeric_columns] = imputer.fit_transform(df_clean[numeric_columns])
            self.imputers['numeric'] = imputer
            
        elif strategy == 'median':
            numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
            imputer = SimpleImputer(strategy='median')
            df_clean[numeric_columns] = imputer.fit_transform(df_clean[numeric_columns])
            self.imputers['numeric'] = imputer
            
        elif strategy == 'knn':
            numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
            imputer = KNNImputer(n_neighbors=5)
            df_clean[numeric_columns] = imputer.fit_transform(df_clean[numeric_columns])
            self.imputers['numeric'] = imputer
        
        # Handle categorical missing values
        categorical_columns = df_clean.select_dtypes(include=['object']).columns
        categorical_columns = [col for col in categorical_columns if col != 'timestamp']
        
        for col in categorical_columns:
            if df_clean[col].isnull().any():
                mode_value = df_clean[col].mode().iloc[0] if not df_clean[col].mode().empty else 'unknown'
                df_clean[col] = df_clean[col].fillna(mode_value)
        
        final_missing = df_clean.isnull().sum().sum()
        logger.info(f"✅ Missing value handling complete. Remaining missing values: {final_missing}")
        
        return df_clean
    
    def detect_and_handle_outliers(self, df: pd.DataFrame, method: str = 'iqr', 
                                  threshold: float = 1.5) -> pd.DataFrame:
        """Detect and handle outliers"""
        logger.info(f"🔍 Detecting outliers using {method} method...")
        
        df_clean = df.copy()
        numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
        
        outlier_summary = {}
        
        for column in numeric_columns:
            if method == 'iqr':
                Q1 = df_clean[column].quantile(0.25)
                Q3 = df_clean[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outliers = ((df_clean[column] < lower_bound) | (df_clean[column] > upper_bound))
                
            elif method == 'zscore':
                z_scores = np.abs((df_clean[column] - df_clean[column].mean()) / df_clean[column].std())
                outliers = z_scores > threshold
                
            elif method == 'modified_zscore':
                median = df_clean[column].median()
                mad = np.median(np.abs(df_clean[column] - median))
                modified_z_scores = 0.6745 * (df_clean[column] - median) / mad
                outliers = np.abs(modified_z_scores) > threshold
            
            outlier_count = outliers.sum()
            outlier_percentage = outlier_count / len(df_clean) * 100
            
            outlier_summary[column] = {
                'count': int(outlier_count),
                'percentage': round(outlier_percentage, 2)
            }
            
            if outlier_count > 0:
                if outlier_percentage < 5:  # Cap outliers if <5%
                    if method == 'iqr':
                        df_clean[column] = df_clean[column].clip(lower_bound, upper_bound)
                        logger.info(f"   Capped {outlier_count} outliers in {column}")
                    else:
                        # For z-score methods, cap at percentiles
                        lower_cap = df_clean[column].quantile(0.01)
                        upper_cap = df_clean[column].quantile(0.99)
                        df_clean[column] = df_clean[column].clip(lower_cap, upper_cap)
                        logger.info(f"   Capped {outlier_count} outliers in {column}")
                else:
                    logger.warning(f"   High outlier rate in {column}: {outlier_percentage:.1f}% - keeping as is")
        
        total_outliers = sum(summary['count'] for summary in outlier_summary.values())
        logger.info(f"✅ Outlier detection complete. Total outliers processed: {total_outliers}")
        
        return df_clean, outlier_summary
    
    def normalize_data(self, df: pd.DataFrame, method: str = 'standard', 
                      exclude_columns: list = None) -> pd.DataFrame:
        """Normalize numerical data"""
        logger.info(f"📊 Normalizing data using {method} method...")
        
        df_normalized = df.copy()
        
        # Columns to exclude from normalization
        if exclude_columns is None:
            exclude_columns = ['timestamp', 'failure_within_hour', 'system_state']
        
        numeric_columns = df_normalized.select_dtypes(include=[np.number]).columns
        columns_to_normalize = [col for col in numeric_columns if col not in exclude_columns]
        
        if len(columns_to_normalize) == 0:
            logger.info("No columns to normalize")
            return df_normalized
        
        # Select scaler
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        # Fit and transform
        df_normalized[columns_to_normalize] = scaler.fit_transform(df_normalized[columns_to_normalize])
        
        # Store scaler for later use
        self.scalers[method] = scaler
        
        logger.info(f"✅ Normalized {len(columns_to_normalize)} columns using {method} scaling")
        
        return df_normalized
    
    def create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create derived features from existing data"""
        logger.info("🛠️ Creating derived features...")
        
        df_enhanced = df.copy()
        
        # Time-based features
        if 'timestamp' in df_enhanced.columns:
            df_enhanced['hour'] = df_enhanced['timestamp'].dt.hour
            df_enhanced['day_of_week'] = df_enhanced['timestamp'].dt.dayofweek
            df_enhanced['is_weekend'] = (df_enhanced['day_of_week'] >= 5).astype(int)
            df_enhanced['is_business_hours'] = (
                (df_enhanced['hour'] >= 9) & 
                (df_enhanced['hour'] <= 17) & 
                (df_enhanced['is_weekend'] == 0)
            ).astype(int)
        
        # System resource ratios
        if all(col in df_enhanced.columns for col in ['cpu_usage', 'memory_usage']):
            df_enhanced['cpu_memory_ratio'] = df_enhanced['cpu_usage'] / (df_enhanced['memory_usage'] + 1)
            df_enhanced['resource_utilization'] = (df_enhanced['cpu_usage'] + df_enhanced['memory_usage']) / 2
        
        # Performance indicators
        if all(col in df_enhanced.columns for col in ['response_time_ms', 'active_connections']):
            df_enhanced['response_per_connection'] = df_enhanced['response_time_ms'] / (df_enhanced['active_connections'] + 1)
        
        if all(col in df_enhanced.columns for col in ['error_count', 'active_connections']):
            df_enhanced['error_rate'] = df_enhanced['error_count'] / (df_enhanced['active_connections'] + 1)
        
        # System stress indicators
        stress_columns = ['cpu_usage', 'memory_usage', 'disk_usage']
        if all(col in df_enhanced.columns for col in stress_columns):
            for threshold in [70, 80, 90]:
                df_enhanced[f'stress_level_{threshold}'] = sum(
                    (df_enhanced[col] > threshold).astype(int) for col in stress_columns
                )
        
        new_features = len(df_enhanced.columns) - len(df.columns)
        logger.info(f"✅ Created {new_features} derived features")
        
        return df_enhanced
    
    def remove_low_variance_features(self, df: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
        """Remove features with low variance"""
        logger.info(f"🔽 Removing low variance features (threshold: {threshold})...")
        
        df_filtered = df.copy()
        
        # Exclude non-numeric and important columns
        exclude_columns = ['timestamp', 'failure_within_hour', 'system_state']
        numeric_columns = df_filtered.select_dtypes(include=[np.number]).columns
        columns_to_check = [col for col in numeric_columns if col not in exclude_columns]
        
        low_variance_columns = []
        
        for column in columns_to_check:
            variance = df_filtered[column].var()
            if variance < threshold:
                low_variance_columns.append(column)
        
        if low_variance_columns:
            df_filtered = df_filtered.drop(columns=low_variance_columns)
            logger.info(f"   Removed {len(low_variance_columns)} low variance features: {low_variance_columns}")
        else:
            logger.info("   No low variance features found")
        
        return df_filtered
    
    def handle_class_imbalance(self, df: pd.DataFrame, target_column: str = 'failure_within_hour',
                              method: str = 'smote') -> pd.DataFrame:
        """Handle class imbalance in target variable"""
        logger.info(f"⚖️ Handling class imbalance using {method} method...")
        
        if target_column not in df.columns:
            logger.warning(f"Target column {target_column} not found")
            return df
        
        # Check current class distribution
        class_counts = df[target_column].value_counts()
        logger.info(f"Current class distribution: {class_counts.to_dict()}")
        
        # Calculate imbalance ratio
        minority_class = class_counts.min()
        majority_class = class_counts.max()
        imbalance_ratio = majority_class / minority_class
        
        if imbalance_ratio < 2:
            logger.info("Classes are relatively balanced, no action needed")
            return df
        
        logger.info(f"Imbalance ratio: {imbalance_ratio:.2f}")
        
        if method == 'undersample':
            # Random undersampling of majority class
            majority_class_label = class_counts.idxmax()
            minority_class_label = class_counts.idxmin()
            
            majority_samples = df[df[target_column] == majority_class_label]
            minority_samples = df[df[target_column] == minority_class_label]
            
            # Sample majority class to match minority class size
            majority_downsampled = majority_samples.sample(n=len(minority_samples), random_state=42)
            
            df_balanced = pd.concat([majority_downsampled, minority_samples], ignore_index=True)
            
        elif method == 'oversample':
            # Random oversampling of minority class
            majority_class_label = class_counts.idxmax()
            minority_class_label = class_counts.idxmin()
            
            majority_samples = df[df[target_column] == majority_class_label]
            minority_samples = df[df[target_column] == minority_class_label]
            
            # Oversample minority class to match majority class size
            minority_oversampled = minority_samples.sample(n=len(majority_samples), 
                                                         replace=True, random_state=42)
            
            df_balanced = pd.concat([majority_samples, minority_oversampled], ignore_index=True)
        
        else:
            logger.warning(f"Unknown balancing method: {method}")
            return df
        
        # Shuffle the balanced dataset
        df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
        
        new_class_counts = df_balanced[target_column].value_counts()
        logger.info(f"New class distribution: {new_class_counts.to_dict()}")
        
        return df_balanced
    
    def preprocess_pipeline(self, df: pd.DataFrame, config: dict = None) -> pd.DataFrame:
        """Run complete preprocessing pipeline"""
        logger.info("🚀 Starting preprocessing pipeline...")
        
        # Default configuration
        default_config = {
            'handle_missing': True,
            'missing_strategy': 'adaptive',
            'handle_outliers': True,
            'outlier_method': 'iqr',
            'outlier_threshold': 1.5,
            'normalize': True,
            'normalization_method': 'standard',
            'create_derived': True,
            'remove_low_variance': True,
            'variance_threshold': 0.01,
            'handle_imbalance': False,
            'balance_method': 'undersample'
        }
        
        if config:
            default_config.update(config)
        
        self.preprocessing_config = default_config
        processed_df = df.copy()
        
        # Step 1: Handle missing values
        if default_config['handle_missing']:
            processed_df = self.handle_missing_values(
                processed_df, 
                strategy=default_config['missing_strategy']
            )
        
        # Step 2: Create derived features
        if default_config['create_derived']:
            processed_df = self.create_derived_features(processed_df)
        
        # Step 3: Handle outliers
        if default_config['handle_outliers']:
            processed_df, outlier_summary = self.detect_and_handle_outliers(
                processed_df,
                method=default_config['outlier_method'],
                threshold=default_config['outlier_threshold']
            )
        
        # Step 4: Remove low variance features
        if default_config['remove_low_variance']:
            processed_df = self.remove_low_variance_features(
                processed_df,
                threshold=default_config['variance_threshold']
            )
        
        # Step 5: Handle class imbalance
        if default_config['handle_imbalance']:
            processed_df = self.handle_class_imbalance(
                processed_df,
                method=default_config['balance_method']
            )
        
        # Step 6: Normalize data (do this last to avoid affecting other steps)
        if default_config['normalize']:
            processed_df = self.normalize_data(
                processed_df,
                method=default_config['normalization_method']
            )
        
        logger.info("✅ Preprocessing pipeline completed successfully!")
        logger.info(f"Final dataset shape: {processed_df.shape}")
        
        return processed_df
    
    def save_preprocessing_artifacts(self, output_dir: str = "models"):
        """Save preprocessing artifacts for later use"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Save scalers
        if self.scalers:
            for scaler_name, scaler in self.scalers.items():
                scaler_path = output_dir / f"{scaler_name}_scaler.joblib"
                import joblib
                joblib.dump(scaler, scaler_path)
                logger.info(f"💾 Saved {scaler_name} scaler to {scaler_path}")
        
        # Save imputers
        if self.imputers:
            for imputer_name, imputer in self.imputers.items():
                imputer_path = output_dir / f"{imputer_name}_imputer.joblib"
                import joblib
                joblib.dump(imputer, imputer_path)
                logger.info(f"💾 Saved {imputer_name} imputer to {imputer_path}")
        
        # Save preprocessing configuration
        config_path = output_dir / "preprocessing_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.preprocessing_config, f, indent=2, default=str)
        logger.info(f"💾 Saved preprocessing config to {config_path}")
    
    def generate_preprocessing_report(self, original_df: pd.DataFrame, 
                                    processed_df: pd.DataFrame) -> dict:
        """Generate a preprocessing report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'original_shape': original_df.shape,
            'processed_shape': processed_df.shape,
            'preprocessing_config': self.preprocessing_config,
            'changes': {
                'rows_added': processed_df.shape[0] - original_df.shape[0],
                'columns_added': processed_df.shape[1] - original_df.shape[1],
                'missing_values_before': original_df.isnull().sum().sum(),
                'missing_values_after': processed_df.isnull().sum().sum()
            },
            'column_changes': {
                'original_columns': list(original_df.columns),
                'processed_columns': list(processed_df.columns),
                'added_columns': list(set(processed_df.columns) - set(original_df.columns)),
                'removed_columns': list(set(original_df.columns) - set(processed_df.columns))
            }
        }
        
        return report
    
    def apply_saved_preprocessing(self, df: pd.DataFrame, artifacts_dir: str = "models") -> pd.DataFrame:
        """Apply saved preprocessing artifacts to new data"""
        logger.info("🔄 Applying saved preprocessing artifacts...")
        
        artifacts_dir = Path(artifacts_dir)
        processed_df = df.copy()
        
        # Load and apply scalers
        for scaler_file in artifacts_dir.glob("*_scaler.joblib"):
            scaler_name = scaler_file.stem.replace('_scaler', '')
            
            try:
                import joblib
                scaler = joblib.load(scaler_file)
                
                # Apply scaler to numeric columns (excluding target and timestamp)
                exclude_columns = ['timestamp', 'failure_within_hour', 'system_state']
                numeric_columns = processed_df.select_dtypes(include=[np.number]).columns
                columns_to_scale = [col for col in numeric_columns if col not in exclude_columns]
                
                if len(columns_to_scale) > 0:
                    processed_df[columns_to_scale] = scaler.transform(processed_df[columns_to_scale])
                    logger.info(f"   Applied {scaler_name} scaler")
                    
            except Exception as e:
                logger.warning(f"   Failed to apply {scaler_name} scaler: {str(e)}")
        
        # Load and apply imputers
        for imputer_file in artifacts_dir.glob("*_imputer.joblib"):
            imputer_name = imputer_file.stem.replace('_imputer', '')
            
            try:
                import joblib
                imputer = joblib.load(imputer_file)
                
                # Apply imputer based on name
                if imputer_name == 'numeric':
                    numeric_columns = processed_df.select_dtypes(include=[np.number]).columns
                    processed_df[numeric_columns] = imputer.transform(processed_df[numeric_columns])
                else:
                    if imputer_name in processed_df.columns:
                        processed_df[imputer_name] = imputer.transform(processed_df[[imputer_name]]).ravel()
                
                logger.info(f"   Applied {imputer_name} imputer")
                
            except Exception as e:
                logger.warning(f"   Failed to apply {imputer_name} imputer: {str(e)}")
        
        logger.info("✅ Preprocessing artifacts applied")
        return processed_df

def main():
    """Example usage of DataPreprocessor"""
    preprocessor = DataPreprocessor()
    
    # Example with synthetic data
    import numpy as np
    
    # Create sample data with issues
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=1000, freq='H'),
        'cpu_usage': np.random.uniform(0, 100, 1000),
        'memory_usage': np.concatenate([np.random.uniform(20, 80, 950), [None] * 50]),  # Missing values
        'disk_usage': np.concatenate([np.random.uniform(10, 90, 990), [150, 200] * 5]),  # Outliers
        'error_count': np.random.poisson(2, 1000),
        'response_time_ms': np.random.uniform(100, 1000, 1000),
        'failure_within_hour': np.random.choice([0, 1], 1000, p=[0.8, 0.2])  # Imbalanced
    })
    
    print("🔍 Original data summary:")
    print(f"Shape: {sample_data.shape}")
    print(f"Missing values: {sample_data.isnull().sum().sum()}")
    print(f"Target distribution: {sample_data['failure_within_hour'].value_counts().to_dict()}")
    
    # Apply preprocessing pipeline
    config = {
        'handle_missing': True,
        'handle_outliers': True,
        'create_derived': True,
        'normalize': True,
        'handle_imbalance': True,
        'balance_method': 'oversample'
    }
    
    processed_data = preprocessor.preprocess_pipeline(sample_data, config)
    
    print("\n✅ Processed data summary:")
    print(f"Shape: {processed_data.shape}")
    print(f"Missing values: {processed_data.isnull().sum().sum()}")
    if 'failure_within_hour' in processed_data.columns:
        print(f"Target distribution: {processed_data['failure_within_hour'].value_counts().to_dict()}")
    
    # Generate report
    report = preprocessor.generate_preprocessing_report(sample_data, processed_data)
    print(f"\n📊 Processing Report:")
    print(f"Rows changed: {report['changes']['rows_added']}")
    print(f"Columns added: {report['changes']['columns_added']}")
    print(f"Missing values removed: {report['changes']['missing_values_before'] - report['changes']['missing_values_after']}")
    
    # Save artifacts
    preprocessor.save_preprocessing_artifacts("models")
    
    print("\n🎉 Preprocessing completed successfully!")

if __name__ == "__main__":
    main()