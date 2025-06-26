"""
Improved data generator with more realistic failure scenarios and better class balance.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
from pathlib import Path

class ImprovedSystemDataGenerator:
    def __init__(self, output_dir="data/raw"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # System states with more realistic probabilities
        self.system_states = ['normal', 'warning', 'critical']
        self.error_types = ['OutOfMemory', 'DiskFull', 'NetworkTimeout', 'DatabaseError', 'AppCrash']
        
        # Realistic failure scenarios
        self.failure_scenarios = {
            'memory_leak': {'memory_usage': (85, 98), 'cpu_usage': (70, 90), 'failure_prob': 0.8},
            'cpu_overload': {'cpu_usage': (90, 100), 'memory_usage': (60, 85), 'failure_prob': 0.75},
            'disk_full': {'disk_usage': (95, 100), 'error_count': (5, 15), 'failure_prob': 0.9},
            'network_issues': {'network_latency_ms': (500, 2000), 'response_time_ms': (2000, 5000), 'failure_prob': 0.7},
            'cascade_failure': {'cpu_usage': (85, 100), 'memory_usage': (85, 100), 'error_count': (10, 25), 'failure_prob': 0.95},
            'database_overload': {'response_time_ms': (1500, 4000), 'active_connections': (200, 500), 'failure_prob': 0.65}
        }
        
    def generate_realistic_system_metrics(self, n_days=30, samples_per_hour=12, failure_rate=0.3):
        """Generate more realistic system metrics with higher failure rate"""
        print(f"🔄 Generating {n_days} days of realistic system metrics (target failure rate: {failure_rate:.1%})...")
        
        total_samples = n_days * 24 * samples_per_hour
        start_time = datetime.now() - timedelta(days=n_days)
        timestamps = [start_time + timedelta(minutes=i*5) for i in range(total_samples)]
        
        data = []
        failure_count = 0
        target_failures = int(total_samples * failure_rate)
        
        for i, timestamp in enumerate(timestamps):
            hour = timestamp.hour
            day_of_week = timestamp.weekday()
            
            # Decide if this should be a failure scenario
            should_be_failure = (failure_count < target_failures and 
                               np.random.random() < (failure_rate * 2))  # Increase probability to meet target
            
            if should_be_failure:
                # Choose a failure scenario
                scenario_name = np.random.choice(list(self.failure_scenarios.keys()))
                scenario = self.failure_scenarios[scenario_name]
                failure_within_hour = 1
                failure_count += 1
                
                # Generate metrics based on failure scenario
                record = self._generate_failure_scenario_metrics(
                    timestamp, scenario, scenario_name
                )
                
            else:
                # Generate normal operation metrics
                record = self._generate_normal_metrics(timestamp)
                record['failure_within_hour'] = 0
            
            data.append(record)
        
        # Ensure we have enough failures by converting some normal samples
        while failure_count < target_failures:
            # Find a random normal sample and convert it to failure
            normal_indices = [i for i, d in enumerate(data) if d['failure_within_hour'] == 0]
            if not normal_indices:
                break
                
            idx = np.random.choice(normal_indices)
            scenario_name = np.random.choice(list(self.failure_scenarios.keys()))
            scenario = self.failure_scenarios[scenario_name]
            
            # Update the record to be a failure scenario
            failure_record = self._generate_failure_scenario_metrics(
                data[idx]['timestamp'], scenario, scenario_name
            )
            data[idx] = failure_record
            failure_count += 1
        
        df = pd.DataFrame(data)
        actual_failure_rate = df['failure_within_hour'].mean()
        
        print(f"✅ Generated {len(df)} records with {failure_count} failures ({actual_failure_rate:.1%} failure rate)")
        
        # Save to CSV
        output_file = self.output_dir / "system_metrics.csv"
        df.to_csv(output_file, index=False)
        print(f"💾 Saved system metrics to {output_file}")
        
        return df
    
    def _generate_failure_scenario_metrics(self, timestamp, scenario, scenario_name):
        """Generate metrics for a specific failure scenario"""
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        
        # Base load factors
        business_hour_multiplier = 1.5 if 9 <= hour <= 17 and day_of_week < 5 else 1.0
        weekend_multiplier = 0.7 if day_of_week >= 5 else 1.0
        base_load = business_hour_multiplier * weekend_multiplier
        
        # Start with elevated normal metrics
        cpu_usage = max(50, np.random.normal(60, 15) * base_load)
        memory_usage = max(40, np.random.normal(55, 15) * base_load)
        disk_usage = max(20, np.random.normal(40, 10))
        network_latency = max(10, np.random.exponential(80) * base_load)
        error_count = max(0, np.random.poisson(3) * base_load)
        response_time = max(150, np.random.normal(300, 100) * base_load)
        active_connections = max(10, np.random.poisson(50) * base_load)
        
        # Apply scenario-specific overrides
        for metric, (min_val, max_val) in scenario.items():
            if metric == 'failure_prob':
                continue
            
            if metric == 'cpu_usage':
                cpu_usage = np.random.uniform(min_val, max_val)
            elif metric == 'memory_usage':
                memory_usage = np.random.uniform(min_val, max_val)
            elif metric == 'disk_usage':
                disk_usage = np.random.uniform(min_val, max_val)
            elif metric == 'network_latency_ms':
                network_latency = np.random.uniform(min_val, max_val)
            elif metric == 'error_count':
                error_count = np.random.randint(min_val, max_val + 1)
            elif metric == 'response_time_ms':
                response_time = np.random.uniform(min_val, max_val)
            elif metric == 'active_connections':
                active_connections = np.random.randint(min_val, max_val + 1)
        
        # Ensure values are within realistic bounds
        cpu_usage = min(100, max(0, cpu_usage))
        memory_usage = min(100, max(0, memory_usage))
        disk_usage = min(100, max(0, disk_usage))
        network_latency = max(1, network_latency)
        error_count = max(0, int(error_count))
        response_time = max(50, response_time)
        active_connections = max(1, int(active_connections))
        
        # Determine system state
        if cpu_usage > 90 or memory_usage > 95 or error_count > 15:
            system_state = 'critical'
        elif cpu_usage > 75 or memory_usage > 80 or error_count > 5:
            system_state = 'warning'
        else:
            system_state = 'normal'
        
        return {
            'timestamp': timestamp,
            'cpu_usage': round(cpu_usage, 2),
            'memory_usage': round(memory_usage, 2),
            'disk_usage': round(disk_usage, 2),
            'network_latency_ms': round(network_latency, 2),
            'error_count': int(error_count),
            'response_time_ms': round(response_time, 2),
            'active_connections': int(active_connections),
            'system_state': system_state,
            'failure_within_hour': 1,
            'failure_scenario': scenario_name,
            'hour': hour,
            'day_of_week': day_of_week,
            'is_weekend': int(day_of_week >= 5)
        }
    
    def _generate_normal_metrics(self, timestamp):
        """Generate normal operation metrics"""
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        
        # Base load factors
        business_hour_multiplier = 1.3 if 9 <= hour <= 17 and day_of_week < 5 else 1.0
        weekend_multiplier = 0.8 if day_of_week >= 5 else 1.0
        base_load = business_hour_multiplier * weekend_multiplier
        
        # Generate normal metrics (lower than failure scenarios)
        cpu_usage = max(5, min(75, np.random.normal(35, 15) * base_load))
        memory_usage = max(10, min(70, np.random.normal(45, 15) * base_load))
        disk_usage = max(10, min(80, 20 + np.random.normal(25, 15)))
        network_latency = max(5, min(150, np.random.exponential(30) * base_load))
        error_count = max(0, int(np.random.poisson(1) * base_load))
        response_time = max(80, min(800, np.random.normal(200, 50) * base_load))
        active_connections = max(5, int(np.random.poisson(30) * base_load))
        
        # System state for normal operations
        if cpu_usage > 65 or memory_usage > 65 or error_count > 3:
            system_state = 'warning'
        else:
            system_state = 'normal'
        
        return {
            'timestamp': timestamp,
            'cpu_usage': round(cpu_usage, 2),
            'memory_usage': round(memory_usage, 2),
            'disk_usage': round(disk_usage, 2),
            'network_latency_ms': round(network_latency, 2),
            'error_count': int(error_count),
            'response_time_ms': round(response_time, 2),
            'active_connections': int(active_connections),
            'system_state': system_state,
            'failure_within_hour': 0,
            'failure_scenario': 'none',
            'hour': hour,
            'day_of_week': day_of_week,
            'is_weekend': int(day_of_week >= 5)
        }
    
    def generate_all_data(self, n_days=30, failure_rate=0.3):
        """Generate all types of data with realistic failure rate"""
        print("🚀 Starting improved data generation...")
        
        # Generate system metrics with higher failure rate
        metrics_df = self.generate_realistic_system_metrics(n_days, failure_rate=failure_rate)
        
        # Generate corresponding application logs
        logs_df = self.generate_corresponding_logs(metrics_df)
        
        # Create summary
        summary = {
            'generation_date': datetime.now().isoformat(),
            'days_generated': n_days,
            'total_metrics': len(metrics_df),
            'total_logs': len(logs_df),
            'failure_rate': metrics_df['failure_within_hour'].mean(),
            'failure_scenarios': metrics_df['failure_scenario'].value_counts().to_dict(),
            'system_state_distribution': metrics_df['system_state'].value_counts().to_dict(),
            'error_rate': len(logs_df[logs_df['level'] == 'ERROR']) / len(logs_df) if len(logs_df) > 0 else 0
        }
        
        # Save summary
        with open(self.output_dir / "data_summary.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"✅ Improved data generation complete!")
        print(f"📊 Generated {summary['total_metrics']} metrics and {summary['total_logs']} logs")
        print(f"📈 Failure rate: {summary['failure_rate']:.2%}")
        print(f"🎯 Failure scenarios: {summary['failure_scenarios']}")
        print(f"📊 System states: {summary['system_state_distribution']}")
        
        return metrics_df, logs_df
    
    def generate_corresponding_logs(self, metrics_df):
        """Generate logs that correspond to the metrics"""
        logs = []
        
        for _, row in metrics_df.iterrows():
            timestamp = row['timestamp']
            system_state = row['system_state']
            failure_scenario = row['failure_scenario']
            
            # Generate 3-8 log entries per metric sample
            n_logs = np.random.randint(3, 9)
            
            for i in range(n_logs):
                log_time = timestamp + timedelta(minutes=np.random.randint(0, 5))
                
                # Determine log level based on system state
                if system_state == 'critical':
                    log_level = np.random.choice(['ERROR', 'WARN', 'INFO'], p=[0.6, 0.3, 0.1])
                elif system_state == 'warning':
                    log_level = np.random.choice(['ERROR', 'WARN', 'INFO'], p=[0.2, 0.5, 0.3])
                else:
                    log_level = np.random.choice(['ERROR', 'WARN', 'INFO', 'DEBUG'], p=[0.05, 0.15, 0.6, 0.2])
                
                # Generate message based on failure scenario
                if failure_scenario != 'none' and log_level in ['ERROR', 'WARN']:
                    message = self._generate_scenario_message(failure_scenario, log_level)
                else:
                    message = self._generate_normal_message(log_level)
                
                log_entry = {
                    'timestamp': log_time,
                    'level': log_level,
                    'message': message,
                    'source': np.random.choice(['API', 'Database', 'Auth', 'Cache', 'Queue', 'Monitor']),
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
    
    def _generate_scenario_message(self, scenario, level):
        """Generate log messages specific to failure scenarios"""
        messages = {
            'memory_leak': {
                'ERROR': ['OutOfMemoryError: Java heap space', 'Memory allocation failed', 'GC overhead limit exceeded'],
                'WARN': ['Memory usage above 90%', 'Potential memory leak detected', 'High memory pressure']
            },
            'cpu_overload': {
                'ERROR': ['CPU usage critical: 95%+', 'Process scheduling timeout', 'System overload detected'],
                'WARN': ['High CPU usage detected', 'CPU throttling enabled', 'Performance degradation warning']
            },
            'disk_full': {
                'ERROR': ['No space left on device', 'Disk quota exceeded', 'Write operation failed'],
                'WARN': ['Disk usage above 90%', 'Low disk space warning', 'Cleanup required']
            },
            'network_issues': {
                'ERROR': ['Connection timeout', 'Network unreachable', 'Socket error'],
                'WARN': ['High network latency detected', 'Packet loss observed', 'Connection retry']
            },
            'cascade_failure': {
                'ERROR': ['Multiple system failures detected', 'Service cascade failure', 'Critical system failure'],
                'WARN': ['System instability detected', 'Multiple warnings active', 'Service degradation']
            },
            'database_overload': {
                'ERROR': ['Database connection timeout', 'Query execution failed', 'Connection pool exhausted'],
                'WARN': ['Slow query detected', 'Database performance warning', 'Connection pool nearly full']
            }
        }
        
        if scenario in messages and level in messages[scenario]:
            return np.random.choice(messages[scenario][level])
        else:
            return self._generate_normal_message(level)
    
    def _generate_normal_message(self, level):
        """Generate normal log messages"""
        messages = {
            'ERROR': ['Authentication failed', 'Invalid request format', 'Service temporarily unavailable'],
            'WARN': ['Rate limit approaching', 'Cache miss rate high', 'Deprecated API usage'],
            'INFO': ['User authenticated successfully', 'Request processed', 'Cache updated', 'Health check passed'],
            'DEBUG': ['Debug: Processing request', 'Debug: Cache lookup', 'Debug: Validation passed']
        }
        
        return np.random.choice(messages.get(level, messages['INFO']))

if __name__ == "__main__":
    generator = ImprovedSystemDataGenerator()
    # Generate data with 30% failure rate (much more realistic for training)
    generator.generate_all_data(n_days=30, failure_rate=0.3)