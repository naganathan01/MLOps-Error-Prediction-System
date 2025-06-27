"""
Enhanced data generator for MLOps Error Prediction System.
File: src/data/data_generator.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedDataGenerator:
    def __init__(self, output_dir="data/raw"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_realistic_data(self, n_days=30, samples_per_hour=12):
        """Generate realistic system metrics with proper failure patterns"""
        logger.info(f"🔄 Generating {n_days} days of realistic system data...")
        
        total_samples = n_days * 24 * samples_per_hour
        start_time = datetime.now() - timedelta(days=n_days)
        
        data = []
        
        for i in range(total_samples):
            timestamp = start_time + timedelta(minutes=i*5)
            hour = timestamp.hour
            day_of_week = timestamp.weekday()
            
            # Time-based load factors
            business_hours = 1.3 if 9 <= hour <= 17 and day_of_week < 5 else 1.0
            weekend_factor = 0.7 if day_of_week >= 5 else 1.0
            night_factor = 0.5 if hour < 6 or hour > 22 else 1.0
            
            base_load = business_hours * weekend_factor * night_factor
            
            # Generate base metrics with realistic correlations
            cpu_base = np.random.normal(40, 15) * base_load
            memory_base = np.random.normal(50, 20) * base_load
            
            # Add realistic correlations
            if cpu_base > 80:
                memory_base += np.random.normal(10, 5)  # High CPU often means high memory
            
            # Generate failure scenarios (15% of time)
            is_failure_scenario = np.random.random() < 0.15
            failure_within_hour = 0
            
            if is_failure_scenario:
                # Create realistic failure patterns
                scenario_type = np.random.choice(['cpu_spike', 'memory_leak', 'cascade_failure'])
                
                if scenario_type == 'cpu_spike':
                    cpu_usage = np.random.uniform(85, 98)
                    memory_usage = memory_base + np.random.uniform(0, 20)
                    error_count = np.random.poisson(8)
                    response_time = np.random.uniform(1500, 4000)
                    failure_within_hour = 1 if np.random.random() < 0.8 else 0
                    
                elif scenario_type == 'memory_leak':
                    cpu_usage = cpu_base + np.random.uniform(0, 15)
                    memory_usage = np.random.uniform(88, 97)
                    error_count = np.random.poisson(5)
                    response_time = np.random.uniform(800, 2500)
                    failure_within_hour = 1 if np.random.random() < 0.85 else 0
                    
                else:  # cascade_failure
                    cpu_usage = np.random.uniform(80, 95)
                    memory_usage = np.random.uniform(85, 95)
                    error_count = np.random.poisson(12)
                    response_time = np.random.uniform(2000, 5000)
                    failure_within_hour = 1 if np.random.random() < 0.9 else 0
            else:
                # Normal operation
                cpu_usage = np.clip(cpu_base, 5, 75)
                memory_usage = np.clip(memory_base, 10, 80)
                error_count = np.random.poisson(2)
                response_time = np.random.normal(300, 100)
                failure_within_hour = 1 if np.random.random() < 0.02 else 0  # 2% background failure rate
            
            # Generate other metrics with realistic relationships
            disk_usage = np.clip(np.random.normal(45, 20), 10, 95)
            network_latency = np.random.exponential(50) + 20
            active_connections = np.random.poisson(50 * base_load)
            
            # Ensure response time is realistic
            response_time = max(100, response_time + network_latency * 0.3)
            if cpu_usage > 80:
                response_time *= 1.5
            if memory_usage > 85:
                response_time *= 1.3
            
            record = {
                'timestamp': timestamp,
                'cpu_usage': round(np.clip(cpu_usage, 0, 100), 2),
                'memory_usage': round(np.clip(memory_usage, 0, 100), 2),
                'disk_usage': round(disk_usage, 2),
                'network_latency_ms': round(network_latency, 2),
                'error_count': max(0, int(error_count)),
                'response_time_ms': round(max(50, response_time), 2),
                'active_connections': max(1, int(active_connections)),
                'hour': hour,
                'day_of_week': day_of_week,
                'is_weekend': int(day_of_week >= 5),
                'is_business_hours': int(9 <= hour <= 17 and day_of_week < 5),
                'failure_within_hour': failure_within_hour
            }
            
            data.append(record)
        
        df = pd.DataFrame(data)
        
        # Save data
        output_file = self.output_dir / "system_metrics.csv"
        df.to_csv(output_file, index=False)
        
        failure_rate = df['failure_within_hour'].mean()
        logger.info(f"✅ Generated {len(df)} records with {failure_rate:.1%} failure rate")
        logger.info(f"💾 Saved to {output_file}")
        
        return df

def main():
    """Main function for standalone execution"""
    logger.info("🚀 Starting data generation...")
    
    generator = EnhancedDataGenerator()
    df = generator.generate_realistic_data(n_days=30)
    
    logger.info("🎉 Data generation completed successfully!")
    return df

if __name__ == "__main__":
    main()
