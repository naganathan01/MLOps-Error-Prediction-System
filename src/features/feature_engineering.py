"""
Feature engineering for MLOps Error Prediction System.
File: src/features/feature_engineering.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        
    def create_features(self):
        """Create comprehensive features for ML"""
        logger.info("🔧 Creating features...")
        
        # Load data
        input_file = self.data_dir / "raw" / "system_metrics.csv"
        if not input_file.exists():
            raise FileNotFoundError(f"Data file not found: {input_file}")
            
        df = pd.read_csv(input_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Basic features already exist
        logger.info(f"   Starting with {len(df.columns)} basic features")
        
        # Create interaction features
        df['cpu_memory_product'] = df['cpu_usage'] * df['memory_usage']
        df['resource_pressure'] = (df['cpu_usage'] + df['memory_usage'] + df['disk_usage']) / 3
        df['performance_ratio'] = df['response_time_ms'] / (df['active_connections'] + 1)
        df['error_rate'] = df['error_count'] / (df['active_connections'] + 1)
        
        # Create stress indicators
        df['cpu_high'] = (df['cpu_usage'] > 80).astype(int)
        df['memory_high'] = (df['memory_usage'] > 85).astype(int)
        df['response_slow'] = (df['response_time_ms'] > 1000).astype(int)
        df['errors_high'] = (df['error_count'] > 5).astype(int)
        
        df['total_stress'] = df['cpu_high'] + df['memory_high'] + df['response_slow'] + df['errors_high']
        
        # Create rolling features (simplified)
        for window in [3, 5]:
            for col in ['cpu_usage', 'memory_usage', 'error_count', 'response_time_ms']:
                df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window=window, min_periods=1).mean()
                df[f'{col}_rolling_max_{window}'] = df[col].rolling(window=window, min_periods=1).max()
        
        # Create lag features
        for lag in [1, 2]:
            for col in ['cpu_usage', 'memory_usage', 'error_count']:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag).fillna(df[col].mean())
        
        # Time features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Fill any NaN values
        df = df.fillna(method='forward').fillna(0)
        
        # Save processed features
        processed_dir = self.data_dir / "processed"
        processed_dir.mkdir(exist_ok=True)
        
        output_file = processed_dir / "features.csv"
        df.to_csv(output_file, index=False)
        
        logger.info(f"✅ Created {len(df.columns)} total features")
        logger.info(f"💾 Saved to {output_file}")
        
        return df

def main():
    """Main function for standalone execution"""
    logger.info("🚀 Starting feature engineering...")
    
    engineer = FeatureEngineer()
    df = engineer.create_features()
    
    logger.info("🎉 Feature engineering completed successfully!")
    return df

if __name__ == "__main__":
    main()
