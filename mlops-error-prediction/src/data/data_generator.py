"""
Generate synthetic system logs and metrics for training the error prediction model.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
from pathlib import Path

class SystemDataGenerator:
    def __init__(self, output_dir="data/raw"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # System states
        self.system_states = ['normal', 'warning', 'critical']
        self.error_types = ['OutOfMemory', 'DiskFull', 'NetworkTimeout', 'DatabaseError', 'AppCrash']
        
    def generate_system_metrics(self, n_days=30, samples_per_hour=12):
        """Generate system metrics data"""
        print(f"🔄 Generating {n_days} days of system metrics...")
        
        # Calculate total samples
        total_samples = n_days * 24 * samples_per_hour
        
        # Generate timestamps
        start_time = datetime.now() - timedelta(days=n_days)
        timestamps = [start_time + timedelta(minutes=i*5) for i in range(total_samples)]
        
        data = []
        
        for i, timestamp in enumerate(timestamps):
            # Create patterns for realistic data
            hour = timestamp.hour
            day_of_week = timestamp.weekday()
            
            # Business hours have higher load
            business_hour_multiplier = 1.5 if 9 <= hour <= 17 and day_of_week < 5 else 1.0
            
            # Weekend has lower load
            weekend_multiplier = 0.7 if day_of_week >= 5 else 1.0
            
            # Random spikes
            spike_multiplier = np.random.choice([1.0, 2.0, 3.0], p=[0.85, 0.10, 0.05])
            
            base_load = business_hour_multiplier * weekend_multiplier * spike_multiplier
            
            # Generate correlated metrics
            cpu_base = np.random.normal(30, 15) * base_load
            cpu_usage = max(0, min(100, cpu_base))
            
            # Memory usage correlated with CPU
            memory_usage = max(0, min(100, cpu_usage * 0.8 + np.random.normal(0, 10)))
            
            # Disk usage grows slowly over time
            disk_usage = min(95, 20 + (i / total_samples) * 50 + np.random.normal(0, 5))
            
            # Network and error metrics
            network_latency = max(1, np.random.exponential(50) * (base_load ** 0.5))
            error_count = np.random.poisson(max(0, (cpu_usage - 70) / 10)) if cpu_usage > 70 else 0
            
            # Response time increases with load
            response_time = max(100, 200 + cpu_usage * 10 + np.random.normal(0, 50))
            
            # Determine system state and failure
            if cpu_usage > 90 or memory_usage > 95 or error_count > 10:
                system_state = 'critical'
                failure_within_hour = np.random.choice([0, 1], p=[0.3, 0.7])
            elif cpu_usage > 75 or memory_usage > 80 or error_count > 5:
                system_state = 'warning'
                failure_within_hour = np.random.choice([0, 1], p=[0.8, 0.2])
            else:
                system_state = 'normal'
                failure_within_hour = np.random.choice([0, 1], p=[0.95, 0.05])
            
            record = {
                'timestamp': timestamp,
                'cpu_usage': round(cpu_usage, 2),
                'memory_usage': round(memory_usage, 2),
                'disk_usage': round(disk_usage, 2),
                'network_latency_ms': round(network_latency, 2),
                'error_count': int(error_count),
                'response_time_ms': round(response_time, 2),
                'active_connections': int(max(1, np.random.poisson(50 * base_load))),
                'system_state': system_state,
                'failure_within_hour': failure_within_hour,
                'hour': hour,
                'day_of_week': day_of_week,
                'is_weekend': int(day_of_week >= 5)
            }
            
            data.append(record)
        
        df = pd.DataFrame(data)
        
        # Save to CSV
        output_file = self.output_dir / "system_metrics.csv"
        df.to_csv(output_file, index=False)
        print(f"✅ Saved {len(df)} system metrics records to {output_file}")
        
        return df
    
    def generate_application_logs(self, n_days=30, logs_per_hour=100):
        """Generate application log entries"""
        print(f"🔄 Generating {n_days} days of application logs...")
        
        total_logs = n_days * 24 * logs_per_hour
        start_time = datetime.now() - timedelta(days=n_days)
        
        logs = []
        
        for i in range(total_logs):
            timestamp = start_time + timedelta(seconds=i * (86400 * n_days / total_logs))
            
            # Log levels with different probabilities
            log_level = np.random.choice(
                ['INFO', 'WARN', 'ERROR', 'DEBUG'], 
                p=[0.70, 0.15, 0.10, 0.05]
            )
            
            # Generate realistic log messages
            if log_level == 'ERROR':
                error_type = np.random.choice(self.error_types)
                message = f"{error_type}: {self._generate_error_message(error_type)}"
            elif log_level == 'WARN':
                message = self._generate_warning_message()
            else:
                message = self._generate_info_message()
            
            log_entry = {
                'timestamp': timestamp,
                'level': log_level,
                'message': message,
                'source': np.random.choice(['API', 'Database', 'Auth', 'Cache', 'Queue']),
                'user_id': f"user_{np.random.randint(1, 1000)}",
                'session_id': f"session_{np.random.randint(1, 10000)}"
            }
            
            logs.append(log_entry)
        
        df_logs = pd.DataFrame(logs)
        
        # Save to CSV
        output_file = self.output_dir / "application_logs.csv"
        df_logs.to_csv(output_file, index=False)
        print(f"✅ Saved {len(df_logs)} application log entries to {output_file}")
        
        return df_logs
    
    def _generate_error_message(self, error_type):
        """Generate realistic error messages"""
        messages = {
            'OutOfMemory': [
                "Java heap space exceeded",
                "Cannot allocate memory for operation",
                "GC overhead limit exceeded"
            ],
            'DiskFull': [
                "No space left on device",
                "Disk quota exceeded",
                "Cannot write to log file"
            ],
            'NetworkTimeout': [
                "Connection timeout after 30s",
                "Read timeout on socket",
                "Gateway timeout"
            ],
            'DatabaseError': [
                "Connection pool exhausted",
                "Query timeout exceeded",
                "Deadlock detected"
            ],
            'AppCrash': [
                "Unexpected application termination",
                "Segmentation fault",
                "Unhandled exception"
            ]
        }
        return np.random.choice(messages[error_type])
    
    def _generate_warning_message(self):
        """Generate warning messages"""
        warnings = [
            "High CPU usage detected",
            "Memory usage above threshold",
            "Slow query detected",
            "Connection pool nearly full",
            "Cache hit ratio low"
        ]
        return np.random.choice(warnings)
    
    def _generate_info_message(self):
        """Generate info messages"""
        info_messages = [
            "User authenticated successfully",
            "Request processed",
            "Cache updated",
            "Scheduled task completed",
            "Health check passed"
        ]
        return np.random.choice(info_messages)
    
    def generate_all_data(self, n_days=30):
        """Generate all types of data"""
        print("🚀 Starting data generation...")
        
        # Generate system metrics
        metrics_df = self.generate_system_metrics(n_days)
        
        # Generate application logs
        logs_df = self.generate_application_logs(n_days)
        
        # Create summary
        summary = {
            'generation_date': datetime.now().isoformat(),
            'days_generated': n_days,
            'total_metrics': len(metrics_df),
            'total_logs': len(logs_df),
            'failure_rate': metrics_df['failure_within_hour'].mean(),
            'error_rate': len(logs_df[logs_df['level'] == 'ERROR']) / len(logs_df)
        }
        
        # Save summary
        with open(self.output_dir / "data_summary.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"✅ Data generation complete!")
        print(f"📊 Generated {summary['total_metrics']} metrics and {summary['total_logs']} logs")
        print(f"📈 Failure rate: {summary['failure_rate']:.2%}")
        print(f"🚨 Error rate: {summary['error_rate']:.2%}")
        
        return metrics_df, logs_df

if __name__ == "__main__":
    generator = SystemDataGenerator()
    generator.generate_all_data(n_days=30)