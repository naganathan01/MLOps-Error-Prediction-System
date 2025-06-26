"""
Improved data generator with realistic failure patterns and better class distribution.
This creates more diverse failure scenarios for better model training.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
from pathlib import Path

class EnhancedSystemDataGenerator:
    def __init__(self, output_dir="data/raw"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Enhanced failure scenarios with realistic patterns
        self.failure_scenarios = {
            'memory_leak_gradual': {
                'pattern': 'gradual_increase',
                'primary_metric': 'memory_usage',
                'base_range': (60, 75),
                'failure_range': (85, 98),
                'duration_hours': 4,
                'failure_probability': 0.9,
                'triggers': ['high_memory', 'slow_response']
            },
            'cpu_spike_sudden': {
                'pattern': 'sudden_spike',
                'primary_metric': 'cpu_usage',
                'base_range': (40, 60),
                'failure_range': (85, 100),
                'duration_hours': 1,
                'failure_probability': 0.8,
                'triggers': ['high_cpu', 'system_overload']
            },
            'disk_full_critical': {
                'pattern': 'threshold_breach',
                'primary_metric': 'disk_usage',
                'base_range': (70, 85),
                'failure_range': (95, 100),
                'duration_hours': 2,
                'failure_probability': 0.95,
                'triggers': ['disk_full', 'write_errors']
            },
            'network_degradation': {
                'pattern': 'gradual_increase',
                'primary_metric': 'network_latency_ms',
                'base_range': (80, 150),
                'failure_range': (400, 1500),
                'duration_hours': 3,
                'failure_probability': 0.7,
                'triggers': ['high_latency', 'connection_issues']
            },
            'cascade_failure': {
                'pattern': 'multi_metric_failure',
                'metrics': ['cpu_usage', 'memory_usage', 'error_count'],
                'base_ranges': [(50, 70), (60, 75), (2, 5)],
                'failure_ranges': [(80, 95), (85, 95), (8, 20)],
                'duration_hours': 2,
                'failure_probability': 0.85,
                'triggers': ['system_cascade', 'multiple_alerts']
            },
            'database_overload': {
                'pattern': 'performance_degradation',
                'primary_metric': 'response_time_ms',
                'base_range': (200, 400),
                'failure_range': (1200, 3000),
                'duration_hours': 2,
                'failure_probability': 0.75,
                'triggers': ['slow_queries', 'connection_pool_exhausted']
            },
            'high_traffic_overload': {
                'pattern': 'load_based',
                'primary_metric': 'active_connections',
                'base_range': (50, 120),
                'failure_range': (300, 600),
                'duration_hours': 1,
                'failure_probability': 0.6,
                'triggers': ['high_load', 'resource_contention']
            }
        }
        
    def generate_enhanced_failure_scenarios(self, n_days=30, samples_per_hour=12, failure_rate=0.35):
        """Generate enhanced system metrics with realistic failure patterns"""
        print(f"🔄 Generating {n_days} days of enhanced system metrics (target failure rate: {failure_rate:.1%})...")
        
        total_samples = n_days * 24 * samples_per_hour
        start_time = datetime.now() - timedelta(days=n_days)
        
        data = []
        failure_count = 0
        target_failures = int(total_samples * failure_rate)
        
        # Track ongoing scenarios
        active_scenarios = {}
        
        for i in range(total_samples):
            timestamp = start_time + timedelta(minutes=i*5)
            hour = timestamp.hour
            day_of_week = timestamp.weekday()
            
            # Check if we should start a new failure scenario
            should_start_failure = (
                failure_count < target_failures and 
                np.random.random() < (failure_rate * 1.5) and
                len(active_scenarios) < 2  # Limit concurrent scenarios
            )
            
            # Start new failure scenario
            if should_start_failure:
                scenario_name = np.random.choice(list(self.failure_scenarios.keys()))
                scenario = self.failure_scenarios[scenario_name]
                
                # Calculate scenario duration in samples
                duration_samples = scenario['duration_hours'] * samples_per_hour
                
                active_scenarios[scenario_name] = {
                    'scenario': scenario,
                    'start_sample': i,
                    'end_sample': i + duration_samples,
                    'progress': 0
                }
            
            # Generate metrics based on active scenarios
            if active_scenarios:
                record = self._generate_scenario_metrics(timestamp, active_scenarios, i)
                if record['failure_within_hour'] == 1:
                    failure_count += 1
            else:
                record = self._generate_normal_metrics(timestamp, hour, day_of_week)
            
            # Clean up completed scenarios
            completed_scenarios = [
                name for name, info in active_scenarios.items()
                if i >= info['end_sample']
            ]
            for name in completed_scenarios:
                del active_scenarios[name]
            
            data.append(record)
        
        df = pd.DataFrame(data)
        actual_failure_rate = df['failure_within_hour'].mean()
        
        print(f"✅ Generated {len(df)} records with {failure_count} failures ({actual_failure_rate:.1%} failure rate)")
        
        # Add realistic noise and correlations
        df = self._add_realistic_correlations(df)
        
        # Save to CSV
        output_file = self.output_dir / "system_metrics.csv"
        df.to_csv(output_file, index=False)
        print(f"💾 Saved enhanced system metrics to {output_file}")
        
        return df
    
    def _generate_scenario_metrics(self, timestamp, active_scenarios, current_sample):
        """Generate metrics during active failure scenarios"""
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        
        # Base load factors
        business_multiplier = 1.4 if 9 <= hour <= 17 and day_of_week < 5 else 1.0
        weekend_multiplier = 0.8 if day_of_week >= 5 else 1.0
        night_multiplier = 0.6 if 22 <= hour or hour <= 6 else 1.0
        base_load = business_multiplier * weekend_multiplier * night_multiplier
        
        # Start with normal metrics
        metrics = self._generate_base_metrics(base_load)
        
        # Apply scenario effects
        failure_probability = 0.0
        scenario_types = []
        
        for scenario_name, scenario_info in active_scenarios.items():
            scenario = scenario_info['scenario']
            progress = (current_sample - scenario_info['start_sample']) / (scenario_info['end_sample'] - scenario_info['start_sample'])
            progress = min(1.0, max(0.0, progress))
            
            # Apply scenario-specific modifications
            metrics = self._apply_scenario_effects(metrics, scenario, progress)
            failure_probability = max(failure_probability, scenario['failure_probability'] * progress)
            scenario_types.append(scenario_name)
        
        # Determine if this is a failure sample
        failure_within_hour = 1 if np.random.random() < failure_probability else 0
        
        # If it's a failure, make metrics more extreme
        if failure_within_hour:
            metrics = self._amplify_failure_metrics(metrics, scenario_types)
        
        return {
            'timestamp': timestamp,
            'cpu_usage': round(min(100, max(0, metrics['cpu_usage'])), 2),
            'memory_usage': round(min(100, max(0, metrics['memory_usage'])), 2),
            'disk_usage': round(min(100, max(0, metrics['disk_usage'])), 2),
            'network_latency_ms': round(max(1, metrics['network_latency_ms']), 2),
            'error_count': max(0, int(metrics['error_count'])),
            'response_time_ms': round(max(50, metrics['response_time_ms']), 2),
            'active_connections': max(1, int(metrics['active_connections'])),
            'system_state': self._determine_system_state(metrics),
            'failure_within_hour': failure_within_hour,
            'failure_scenario': ','.join(scenario_types) if scenario_types else 'none',
            'hour': hour,
            'day_of_week': day_of_week,
            'is_weekend': int(day_of_week >= 5)
        }
    
    def _generate_base_metrics(self, base_load):
        """Generate base metrics with realistic distributions"""
        return {
            'cpu_usage': np.random.normal(35, 15) * base_load,
            'memory_usage': np.random.normal(50, 20) * base_load,
            'disk_usage': 20 + np.random.exponential(15),
            'network_latency_ms': max(10, np.random.exponential(60) * base_load),
            'error_count': np.random.poisson(2) * base_load,
            'response_time_ms': max(100, np.random.normal(250, 80) * base_load),
            'active_connections': max(5, np.random.poisson(40) * base_load)
        }
    
    def _apply_scenario_effects(self, metrics, scenario, progress):
        """Apply scenario-specific effects to metrics"""
        pattern = scenario['pattern']
        
        if pattern == 'gradual_increase':
            primary_metric = scenario['primary_metric']
            base_min, base_max = scenario['base_range']
            fail_min, fail_max = scenario['failure_range']
            
            # Gradually increase the primary metric
            current_min = base_min + (fail_min - base_min) * progress
            current_max = base_max + (fail_max - base_max) * progress
            metrics[primary_metric] = np.random.uniform(current_min, current_max)
            
        elif pattern == 'sudden_spike':
            if progress > 0.7:  # Spike happens in last 30% of scenario
                primary_metric = scenario['primary_metric']
                fail_min, fail_max = scenario['failure_range']
                metrics[primary_metric] = np.random.uniform(fail_min, fail_max)
                
        elif pattern == 'threshold_breach':
            if progress > 0.5:  # Breach happens halfway through
                primary_metric = scenario['primary_metric']
                fail_min, fail_max = scenario['failure_range']
                metrics[primary_metric] = np.random.uniform(fail_min, fail_max)
                
        elif pattern == 'multi_metric_failure':
            for i, metric in enumerate(scenario['metrics']):
                base_min, base_max = scenario['base_ranges'][i]
                fail_min, fail_max = scenario['failure_ranges'][i]
                
                current_min = base_min + (fail_min - base_min) * progress
                current_max = base_max + (fail_max - base_max) * progress
                metrics[metric] = np.random.uniform(current_min, current_max)
                
        elif pattern == 'performance_degradation':
            # Affects response time and related metrics
            primary_metric = scenario['primary_metric']
            fail_min, fail_max = scenario['failure_range']
            
            degradation_factor = 1 + progress * 3  # Up to 4x degradation
            metrics[primary_metric] *= degradation_factor
            metrics['error_count'] *= (1 + progress)
            
        elif pattern == 'load_based':
            # High load affects multiple metrics
            primary_metric = scenario['primary_metric']
            fail_min, fail_max = scenario['failure_range']
            
            current_load = scenario['base_range'][0] + (fail_max - scenario['base_range'][0]) * progress
            metrics[primary_metric] = current_load
            
            # High load affects other metrics
            metrics['cpu_usage'] *= (1 + progress * 0.5)
            metrics['memory_usage'] *= (1 + progress * 0.3)
            metrics['response_time_ms'] *= (1 + progress * 0.8)
        
        return metrics
    
    def _amplify_failure_metrics(self, metrics, scenario_types):
        """Amplify metrics for failure samples to make them more distinct"""
        amplification_factor = 1.2 + np.random.uniform(0, 0.3)
        
        # Amplify based on scenario types
        if 'memory_leak_gradual' in scenario_types:
            metrics['memory_usage'] = min(98, metrics['memory_usage'] * amplification_factor)
            metrics['response_time_ms'] *= 1.5
            
        if 'cpu_spike_sudden' in scenario_types:
            metrics['cpu_usage'] = min(100, metrics['cpu_usage'] * amplification_factor)
            metrics['error_count'] *= 2
            
        if 'cascade_failure' in scenario_types:
            metrics['cpu_usage'] = min(95, metrics['cpu_usage'] * amplification_factor)
            metrics['memory_usage'] = min(95, metrics['memory_usage'] * amplification_factor)
            metrics['error_count'] = max(10, metrics['error_count'] * 3)
            
        return metrics
    
    def _generate_normal_metrics(self, timestamp, hour, day_of_week):
        """Generate normal operation metrics"""
        business_multiplier = 1.2 if 9 <= hour <= 17 and day_of_week < 5 else 1.0
        weekend_multiplier = 0.8 if day_of_week >= 5 else 1.0
        base_load = business_multiplier * weekend_multiplier
        
        metrics = self._generate_base_metrics(base_load)
        
        # Keep normal metrics within safe ranges
        metrics['cpu_usage'] = min(70, max(5, metrics['cpu_usage']))
        metrics['memory_usage'] = min(75, max(10, metrics['memory_usage']))
        metrics['disk_usage'] = min(80, max(10, metrics['disk_usage']))
        metrics['error_count'] = min(5, metrics['error_count'])
        
        return {
            'timestamp': timestamp,
            'cpu_usage': round(metrics['cpu_usage'], 2),
            'memory_usage': round(metrics['memory_usage'], 2),
            'disk_usage': round(metrics['disk_usage'], 2),
            'network_latency_ms': round(metrics['network_latency_ms'], 2),
            'error_count': int(metrics['error_count']),
            'response_time_ms': round(metrics['response_time_ms'], 2),
            'active_connections': int(metrics['active_connections']),
            'system_state': 'normal',
            'failure_within_hour': 0,
            'failure_scenario': 'none',
            'hour': hour,
            'day_of_week': day_of_week,
            'is_weekend': int(day_of_week >= 5)
        }
    
    def _add_realistic_correlations(self, df):
        """Add realistic correlations between metrics"""
        # Add correlation between CPU and response time
        high_cpu_mask = df['cpu_usage'] > 80
        df.loc[high_cpu_mask, 'response_time_ms'] *= np.random.uniform(1.5, 3.0, high_cpu_mask.sum())
        
        # Add correlation between memory and error count
        high_memory_mask = df['memory_usage'] > 85
        df.loc[high_memory_mask, 'error_count'] += np.random.poisson(3, high_memory_mask.sum())
        
        # Add correlation between network latency and response time
        high_latency_mask = df['network_latency_ms'] > 200
        df.loc[high_latency_mask, 'response_time_ms'] += df.loc[high_latency_mask, 'network_latency_ms'] * 0.5
        
        # Add correlation between active connections and resource usage
        high_conn_mask = df['active_connections'] > 200
        df.loc[high_conn_mask, 'cpu_usage'] += np.random.uniform(10, 25, high_conn_mask.sum())
        df.loc[high_conn_mask, 'memory_usage'] += np.random.uniform(5, 15, high_conn_mask.sum())
        
        # Ensure values stay within bounds
        df['cpu_usage'] = df['cpu_usage'].clip(0, 100)
        df['memory_usage'] = df['memory_usage'].clip(0, 100)
        df['disk_usage'] = df['disk_usage'].clip(0, 100)
        df['error_count'] = df['error_count'].clip(0, 50)
        
        return df
    
    def _determine_system_state(self, metrics):
        """Determine system state based on metrics"""
        if (metrics['cpu_usage'] > 90 or metrics['memory_usage'] > 90 or 
            metrics['error_count'] > 15 or metrics['disk_usage'] > 95):
            return 'critical'
        elif (metrics['cpu_usage'] > 75 or metrics['memory_usage'] > 80 or 
              metrics['error_count'] > 5 or metrics['response_time_ms'] > 1000):
            return 'warning'
        else:
            return 'normal'
    
    def generate_corresponding_logs(self, metrics_df):
        """Generate logs that correspond to the metrics with failure scenarios"""
        logs = []
        
        for _, row in metrics_df.iterrows():
            timestamp = row['timestamp']
            system_state = row['system_state']
            failure_scenario = row['failure_scenario']
            failure_within_hour = row['failure_within_hour']
            
            # Generate more logs for failure scenarios
            if failure_within_hour:
                n_logs = np.random.randint(8, 15)  # More logs during failures
            else:
                n_logs = np.random.randint(3, 8)
            
            for i in range(n_logs):
                log_time = timestamp + timedelta(minutes=np.random.randint(0, 5))
                
                # Determine log level based on failure scenario
                if failure_within_hour:
                    log_level = np.random.choice(['ERROR', 'WARN', 'INFO'], p=[0.7, 0.2, 0.1])
                elif system_state == 'critical':
                    log_level = np.random.choice(['ERROR', 'WARN', 'INFO'], p=[0.6, 0.3, 0.1])
                elif system_state == 'warning':
                    log_level = np.random.choice(['ERROR', 'WARN', 'INFO'], p=[0.3, 0.5, 0.2])
                else:
                    log_level = np.random.choice(['ERROR', 'WARN', 'INFO', 'DEBUG'], p=[0.1, 0.2, 0.5, 0.2])
                
                # Generate scenario-specific messages
                if failure_scenario != 'none' and log_level in ['ERROR', 'WARN']:
                    message = self._generate_scenario_log_message(failure_scenario, log_level)
                else:
                    message = self._generate_normal_log_message(log_level)
                
                log_entry = {
                    'timestamp': log_time,
                    'level': log_level,
                    'message': message,
                    'source': np.random.choice(['API', 'Database', 'Auth', 'Cache', 'Queue', 'Monitor', 'Scheduler']),
                    'user_id': f"user_{np.random.randint(1, 1000)}",
                    'session_id': f"session_{np.random.randint(1, 10000)}"
                }
                
                logs.append(log_entry)
        
        df_logs = pd.DataFrame(logs)
        
        # Save to CSV
        output_file = self.output_dir / "application_logs.csv"
        df_logs.to_csv(output_file, index=False)
        print(f"✅ Saved {len(df_logs)} enhanced log entries to {output_file}")
        
        return df_logs
    
    def _generate_scenario_log_message(self, scenario, level):
        """Generate log messages specific to failure scenarios"""
        scenario_messages = {
            'memory_leak_gradual': {
                'ERROR': ['OutOfMemoryError: Java heap space exhausted', 'Memory allocation failed for buffer', 'GC overhead limit exceeded', 'Native memory allocation failed'],
                'WARN': ['Memory usage critically high: 95%', 'Potential memory leak in session handler', 'GC pressure detected', 'Memory pool usage warning']
            },
            'cpu_spike_sudden': {
                'ERROR': ['CPU usage critical: 98% sustained', 'Process scheduling timeout', 'System overload: killing processes', 'CPU throttling activated'],
                'WARN': ['High CPU usage spike detected', 'CPU cores at maximum', 'Process queue backing up', 'Performance degradation warning']
            },
            'disk_full_critical': {
                'ERROR': ['No space left on device /var/log', 'Disk quota exceeded for user', 'Write operation failed: disk full', 'Unable to create temp files'],
                'WARN': ['Disk usage critical: 98% on /var', 'Log rotation failed: disk space', 'Database growth warning', 'Cleanup required immediately']
            },
            'network_degradation': {
                'ERROR': ['Connection timeout to external service', 'Network unreachable: packet loss 80%', 'Socket connection reset', 'API gateway timeout'],
                'WARN': ['High network latency: 800ms avg', 'Packet loss detected: 15%', 'Connection retry limit reached', 'CDN performance degraded']
            },
            'cascade_failure': {
                'ERROR': ['Multiple system failures detected', 'Service cascade failure initiated', 'Critical system instability', 'Emergency shutdown triggered'],
                'WARN': ['System instability cascade detected', 'Multiple subsystems failing', 'Failover mechanisms activating', 'Service degradation spreading']
            },
            'database_overload': {
                'ERROR': ['Database connection timeout: 30s', 'Query execution failed: timeout', 'Connection pool exhausted', 'Deadlock detected in transaction'],
                'WARN': ['Slow query detected: 15s execution', 'Database performance degraded', 'Connection pool 90% utilized', 'Index scan warning']
            },
            'high_traffic_overload': {
                'ERROR': ['Load balancer capacity exceeded', 'Request queue overflow', 'Circuit breaker opened', 'Rate limit exceeded'],
                'WARN': ['High traffic volume detected', 'Response time degradation', 'Server capacity warning', 'Queue depth increasing']
            }
        }
        
        scenarios = scenario.split(',')
        for s in scenarios:
            if s in scenario_messages and level in scenario_messages[s]:
                return np.random.choice(scenario_messages[s][level])
        
        return self._generate_normal_log_message(level)
    
    def _generate_normal_log_message(self, level):
        """Generate normal log messages"""
        messages = {
            'ERROR': ['Authentication failed for user', 'Invalid request format received', 'Service temporarily unavailable', 'Database connection lost'],
            'WARN': ['Rate limit approaching for API key', 'Cache miss rate elevated', 'Deprecated API endpoint used', 'SSL certificate expires soon'],
            'INFO': ['User session created successfully', 'Request processed in 120ms', 'Cache refresh completed', 'Health check passed', 'Backup completed'],
            'DEBUG': ['Request validation passed', 'Cache lookup miss', 'Session cleanup completed', 'Metrics collection cycle']
        }
        
        return np.random.choice(messages.get(level, messages['INFO']))
    
    def generate_all_data(self, n_days=30, failure_rate=0.35):
        """Generate all enhanced data with better failure patterns"""
        print("🚀 Starting enhanced data generation with realistic failure patterns...")
        
        # Generate enhanced system metrics
        metrics_df = self.generate_enhanced_failure_scenarios(n_days, failure_rate=failure_rate)
        
        # Generate corresponding logs
        logs_df = self.generate_corresponding_logs(metrics_df)
        
        # Create enhanced summary
        summary = {
            'generation_date': datetime.now().isoformat(),
            'days_generated': n_days,
            'total_metrics': len(metrics_df),
            'total_logs': len(logs_df),
            'failure_rate': metrics_df['failure_within_hour'].mean(),
            'failure_scenarios': metrics_df[metrics_df['failure_scenario'] != 'none']['failure_scenario'].value_counts().to_dict(),
            'system_state_distribution': metrics_df['system_state'].value_counts().to_dict(),
            'error_rate': len(logs_df[logs_df['level'] == 'ERROR']) / len(logs_df) if len(logs_df) > 0 else 0,
            'enhancement_features': [
                'Realistic failure scenario progression',
                'Multi-metric cascade failures',
                'Time-based scenario duration',
                'Correlated metric relationships',
                'Enhanced log message context'
            ]
        }
        
        # Save summary
        with open(self.output_dir / "enhanced_data_summary.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"✅ Enhanced data generation complete!")
        print(f"📊 Generated {summary['total_metrics']} metrics and {summary['total_logs']} logs")
        print(f"📈 Failure rate: {summary['failure_rate']:.2%}")
        print(f"🎯 Failure scenarios: {summary['failure_scenarios']}")
        print(f"📊 System states: {summary['system_state_distribution']}")
        print(f"🔥 Error rate in logs: {summary['error_rate']:.2%}")
        
        return metrics_df, logs_df

if __name__ == "__main__":
    generator = EnhancedSystemDataGenerator()
    # Generate data with 35% failure rate for better model training
    generator.generate_all_data(n_days=30, failure_rate=0.35)