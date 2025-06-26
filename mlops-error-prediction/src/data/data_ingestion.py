"""
Data ingestion module for processing various data sources.
Handles real log files, CSV imports, and data validation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
import re
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataIngestion:
    def __init__(self, raw_data_dir="data/raw", processed_data_dir="data/processed"):
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        
        # Create directories if they don't exist
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Data validation rules
        self.validation_rules = {
            'cpu_usage': {'min': 0, 'max': 100, 'type': float},
            'memory_usage': {'min': 0, 'max': 100, 'type': float},
            'disk_usage': {'min': 0, 'max': 100, 'type': float},
            'network_latency_ms': {'min': 0, 'max': 10000, 'type': float},
            'error_count': {'min': 0, 'max': 1000, 'type': int},
            'response_time_ms': {'min': 0, 'max': 60000, 'type': float},
            'active_connections': {'min': 0, 'max': 10000, 'type': int}
        }
    
    def ingest_csv_data(self, file_path: Union[str, Path], data_type: str = "metrics") -> pd.DataFrame:
        """Ingest data from CSV files"""
        logger.info(f"📥 Ingesting CSV data from {file_path}")
        
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        try:
            # Read CSV with flexible parsing
            df = pd.read_csv(file_path, parse_dates=['timestamp'] if 'timestamp' in pd.read_csv(file_path, nrows=1).columns else None)
            
            logger.info(f"✅ Loaded {len(df)} records from {file_path}")
            
            # Validate data
            df_validated = self.validate_data(df, data_type)
            
            return df_validated
            
        except Exception as e:
            logger.error(f"❌ Failed to ingest CSV data: {str(e)}")
            raise
    
    def ingest_log_files(self, log_directory: Union[str, Path], pattern: str = "*.log") -> pd.DataFrame:
        """Ingest and parse log files"""
        logger.info(f"📥 Ingesting log files from {log_directory}")
        
        log_dir = Path(log_directory)
        if not log_dir.exists():
            raise FileNotFoundError(f"Log directory not found: {log_dir}")
        
        log_files = list(log_dir.glob(pattern))
        if not log_files:
            raise FileNotFoundError(f"No log files found with pattern {pattern} in {log_dir}")
        
        all_logs = []
        
        for log_file in log_files:
            logger.info(f"📄 Processing {log_file}")
            logs = self.parse_log_file(log_file)
            all_logs.extend(logs)
        
        # Convert to DataFrame
        df = pd.DataFrame(all_logs)
        
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df = df.dropna(subset=['timestamp'])
            df = df.sort_values('timestamp')
        
        logger.info(f"✅ Ingested {len(df)} log entries from {len(log_files)} files")
        
        return df
    
    def parse_log_file(self, log_file: Path) -> List[Dict]:
        """Parse individual log file"""
        logs = []
        
        # Common log patterns
        patterns = {
            'apache': r'(?P<ip>\S+) - - \[(?P<timestamp>[^\]]+)\] "(?P<method>\S+) (?P<url>\S+) (?P<protocol>\S+)" (?P<status>\d+) (?P<size>\d+)',
            'nginx': r'(?P<ip>\S+) - - \[(?P<timestamp>[^\]]+)\] "(?P<method>\S+) (?P<url>\S+) (?P<protocol>\S+)" (?P<status>\d+) (?P<size>\d+) "(?P<referer>[^"]*)" "(?P<user_agent>[^"]*)"',
            'application': r'(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) (?P<level>\w+) (?P<message>.*)',
            'system': r'(?P<timestamp>\w+ \d+ \d{2}:\d{2}:\d{2}) (?P<hostname>\S+) (?P<service>\S+): (?P<message>.*)'
        }
        
        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    parsed = None
                    
                    # Try each pattern
                    for pattern_name, pattern in patterns.items():
                        match = re.match(pattern, line)
                        if match:
                            parsed = match.groupdict()
                            parsed['pattern_type'] = pattern_name
                            parsed['source_file'] = log_file.name
                            parsed['line_number'] = line_num
                            break
                    
                    # If no pattern matched, create a generic entry
                    if not parsed:
                        parsed = {
                            'timestamp': datetime.now().isoformat(),
                            'level': 'UNKNOWN',
                            'message': line,
                            'pattern_type': 'generic',
                            'source_file': log_file.name,
                            'line_number': line_num
                        }
                    
                    logs.append(parsed)
                    
        except Exception as e:
            logger.warning(f"⚠️ Error parsing {log_file}: {str(e)}")
        
        return logs
    
    def ingest_json_data(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """Ingest data from JSON files"""
        logger.info(f"📥 Ingesting JSON data from {file_path}")
        
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"JSON file not found: {file_path}")
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Handle different JSON structures
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                if 'data' in data:
                    df = pd.DataFrame(data['data'])
                else:
                    df = pd.DataFrame([data])
            else:
                raise ValueError("Unsupported JSON structure")
            
            # Convert timestamp columns
            timestamp_cols = [col for col in df.columns if 'timestamp' in col.lower() or 'time' in col.lower()]
            for col in timestamp_cols:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            
            logger.info(f"✅ Loaded {len(df)} records from JSON file")
            return df
            
        except Exception as e:
            logger.error(f"❌ Failed to ingest JSON data: {str(e)}")
            raise
    
    def validate_data(self, df: pd.DataFrame, data_type: str = "metrics") -> pd.DataFrame:
        """Validate and clean data"""
        logger.info(f"🔍 Validating {data_type} data...")
        
        original_len = len(df)
        df_clean = df.copy()
        
        if data_type == "metrics":
            # Validate system metrics
            for column, rules in self.validation_rules.items():
                if column in df_clean.columns:
                    # Type conversion
                    try:
                        df_clean[column] = df_clean[column].astype(rules['type'])
                    except:
                        logger.warning(f"⚠️ Could not convert {column} to {rules['type']}")
                    
                    # Range validation
                    mask = (df_clean[column] >= rules['min']) & (df_clean[column] <= rules['max'])
                    invalid_count = (~mask).sum()
                    
                    if invalid_count > 0:
                        logger.warning(f"⚠️ Found {invalid_count} invalid values in {column}")
                        # Cap values to valid range
                        df_clean[column] = df_clean[column].clip(rules['min'], rules['max'])
        
        # Remove duplicates
        if 'timestamp' in df_clean.columns:
            df_clean = df_clean.drop_duplicates(subset=['timestamp'])
        else:
            df_clean = df_clean.drop_duplicates()
        
        # Remove rows with too many missing values
        missing_threshold = len(df_clean.columns) * 0.5  # 50% missing
        df_clean = df_clean.dropna(thresh=missing_threshold)
        
        # Fill remaining missing values
        numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[numeric_columns] = df_clean[numeric_columns].fillna(df_clean[numeric_columns].median())
        
        categorical_columns = df_clean.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if col != 'timestamp':
                df_clean[col] = df_clean[col].fillna('unknown')
        
        cleaned_len = len(df_clean)
        removed_count = original_len - cleaned_len
        
        if removed_count > 0:
            logger.info(f"🧹 Removed {removed_count} invalid/duplicate records")
        
        logger.info(f"✅ Data validation completed: {cleaned_len} valid records")
        
        return df_clean
    
    def merge_data_sources(self, metrics_df: pd.DataFrame, logs_df: pd.DataFrame, 
                          time_window: str = '1H') -> pd.DataFrame:
        """Merge metrics and logs data"""
        logger.info("🔗 Merging data sources...")
        
        if 'timestamp' not in metrics_df.columns or 'timestamp' not in logs_df.columns:
            raise ValueError("Both DataFrames must have 'timestamp' column")
        
        # Ensure timestamps are datetime
        metrics_df['timestamp'] = pd.to_datetime(metrics_df['timestamp'])
        logs_df['timestamp'] = pd.to_datetime(logs_df['timestamp'])
        
        # Aggregate logs by time window
        logs_df['time_bucket'] = logs_df['timestamp'].dt.floor(time_window)
        
        log_agg = logs_df.groupby('time_bucket').agg({
            'level': lambda x: (x == 'ERROR').sum() if 'level' in logs_df.columns else 0,
            'message': 'count',
            'source_file': 'nunique'
        }).reset_index()
        
        log_agg.columns = ['timestamp', 'error_count_logs', 'total_log_count', 'unique_sources']
        
        # Create time buckets for metrics
        metrics_df['time_bucket'] = metrics_df['timestamp'].dt.floor(time_window)
        
        # Merge data
        merged_df = metrics_df.merge(log_agg, left_on='time_bucket', right_on='timestamp', 
                                   how='left', suffixes=('', '_log'))
        
        # Fill missing log data
        log_columns = ['error_count_logs', 'total_log_count', 'unique_sources']
        for col in log_columns:
            merged_df[col] = merged_df[col].fillna(0)
        
        # Clean up
        merged_df = merged_df.drop(['time_bucket', 'timestamp_log'], axis=1, errors='ignore')
        
        logger.info(f"✅ Merged data: {len(merged_df)} records")
        
        return merged_df
    
    def save_processed_data(self, df: pd.DataFrame, filename: str):
        """Save processed data"""
        output_path = self.processed_data_dir / filename
        df.to_csv(output_path, index=False)
        logger.info(f"💾 Saved processed data to {output_path}")
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict:
        """Generate data summary statistics"""
        summary = {
            'total_records': len(df),
            'columns': list(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.astype(str).to_dict(),
            'date_range': {}
        }
        
        # Date range information
        if 'timestamp' in df.columns:
            summary['date_range'] = {
                'start': df['timestamp'].min().isoformat() if pd.notna(df['timestamp'].min()) else None,
                'end': df['timestamp'].max().isoformat() if pd.notna(df['timestamp'].max()) else None,
                'duration_days': (df['timestamp'].max() - df['timestamp'].min()).days if pd.notna(df['timestamp'].min()) else None
            }
        
        # Numeric summary
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            summary['numeric_summary'] = df[numeric_cols].describe().to_dict()
        
        return summary
    
    def detect_data_quality_issues(self, df: pd.DataFrame) -> List[Dict]:
        """Detect data quality issues"""
        issues = []
        
        # Check for high missing value rates
        missing_rates = df.isnull().sum() / len(df)
        high_missing = missing_rates[missing_rates > 0.1]  # >10% missing
        
        for col, rate in high_missing.items():
            issues.append({
                'type': 'high_missing_rate',
                'column': col,
                'missing_rate': rate,
                'severity': 'high' if rate > 0.3 else 'medium'
            })
        
        # Check for duplicate timestamps
        if 'timestamp' in df.columns:
            duplicate_timestamps = df['timestamp'].duplicated().sum()
            if duplicate_timestamps > 0:
                issues.append({
                    'type': 'duplicate_timestamps',
                    'count': duplicate_timestamps,
                    'severity': 'medium'
                })
        
        # Check for outliers in numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            if outliers > len(df) * 0.05:  # >5% outliers
                issues.append({
                    'type': 'high_outlier_rate',
                    'column': col,
                    'outlier_count': outliers,
                    'outlier_rate': outliers / len(df),
                    'severity': 'low'
                })
        
        return issues
    
    def run_ingestion_pipeline(self, config: Dict) -> pd.DataFrame:
        """Run complete data ingestion pipeline"""
        logger.info("🚀 Starting data ingestion pipeline...")
        
        dataframes = []
        
        # Ingest CSV files
        if 'csv_files' in config:
            for csv_config in config['csv_files']:
                df = self.ingest_csv_data(csv_config['path'], csv_config.get('type', 'metrics'))
                dataframes.append(df)
        
        # Ingest log files
        if 'log_directories' in config:
            for log_config in config['log_directories']:
                df = self.ingest_log_files(log_config['path'], log_config.get('pattern', '*.log'))
                dataframes.append(df)
        
        # Ingest JSON files
        if 'json_files' in config:
            for json_file in config['json_files']:
                df = self.ingest_json_data(json_file)
                dataframes.append(df)
        
        if not dataframes:
            raise ValueError("No data sources configured")
        
        # Combine all dataframes
        if len(dataframes) == 1:
            combined_df = dataframes[0]
        else:
            # If multiple dataframes, attempt to merge them
            combined_df = dataframes[0]
            for df in dataframes[1:]:
                if 'timestamp' in combined_df.columns and 'timestamp' in df.columns:
                    combined_df = self.merge_data_sources(combined_df, df)
                else:
                    # Simple concatenation if no timestamp
                    combined_df = pd.concat([combined_df, df], ignore_index=True)
        
        # Generate summary and quality report
        summary = self.get_data_summary(combined_df)
        issues = self.detect_data_quality_issues(combined_df)
        
        # Save summary
        summary_path = self.processed_data_dir / "ingestion_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Save quality report
        quality_report = {
            'timestamp': datetime.now().isoformat(),
            'total_issues': len(issues),
            'issues': issues
        }
        
        quality_path = self.processed_data_dir / "data_quality_report.json"
        with open(quality_path, 'w') as f:
            json.dump(quality_report, f, indent=2, default=str)
        
        # Log summary
        logger.info("📊 Ingestion Summary:")
        logger.info(f"   Total records: {summary['total_records']}")
        logger.info(f"   Columns: {len(summary['columns'])}")
        logger.info(f"   Data quality issues: {len(issues)}")
        
        if issues:
            high_severity = [i for i in issues if i.get('severity') == 'high']
            if high_severity:
                logger.warning(f"⚠️ Found {len(high_severity)} high-severity data quality issues")
        
        logger.info("✅ Data ingestion pipeline completed")
        
        return combined_df

def main():
    """Example usage of DataIngestion class"""
    ingestion = DataIngestion()
    
    # Example configuration
    config = {
        'csv_files': [
            {'path': 'data/raw/system_metrics.csv', 'type': 'metrics'},
            {'path': 'data/raw/application_logs.csv', 'type': 'logs'}
        ]
    }
    
    try:
        df = ingestion.run_ingestion_pipeline(config)
        ingestion.save_processed_data(df, 'ingested_data.csv')
        print(f"✅ Successfully ingested {len(df)} records")
        
    except Exception as e:
        print(f"❌ Ingestion failed: {str(e)}")

if __name__ == "__main__":
    main()