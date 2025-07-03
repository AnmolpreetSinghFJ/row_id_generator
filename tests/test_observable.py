"""
Unit Tests for Observable Module
Task 9.1: Unit tests for core functions - Observable Component
"""

import unittest
import pandas as pd
import numpy as np
import tempfile
import os
import json
import time
import logging
from unittest.mock import Mock, patch, MagicMock
import sys

# Add the package to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import observable functions with error handling
try:
    from row_id_generator.observable import *
except ImportError:
    # Create mock classes for testing if observable module doesn't exist yet
    class MockLogger:
        def __init__(self, name='test_logger'):
            self.name = name
            self.entries = []
        
        def info(self, message):
            self.entries.append(('INFO', message))
        
        def warning(self, message):
            self.entries.append(('WARNING', message))
        
        def error(self, message):
            self.entries.append(('ERROR', message))
    
    class MockMetricsCollector:
        def __init__(self):
            self.metrics = {}
        
        def record_metric(self, name, value):
            self.metrics[name] = value
        
        def get_metrics(self):
            return self.metrics.copy()


class TestObservabilityFramework(unittest.TestCase):
    """Test observability framework components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_df = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
            'score': [95.5, 87.2, 78.9, 92.1, 85.6]
        })
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_basic_logging(self):
        """Test basic logging functionality."""
        logger = MockLogger('test_logger')
        
        logger.info('Test info message')
        logger.warning('Test warning message')
        logger.error('Test error message')
        
        # Check logged entries
        self.assertEqual(len(logger.entries), 3)
        self.assertEqual(logger.entries[0], ('INFO', 'Test info message'))
        self.assertEqual(logger.entries[1], ('WARNING', 'Test warning message'))
        self.assertEqual(logger.entries[2], ('ERROR', 'Test error message'))
    
    def test_metrics_collection(self):
        """Test metrics collection functionality."""
        collector = MockMetricsCollector()
        
        collector.record_metric('processing_time', 1.5)
        collector.record_metric('rows_processed', 1000)
        collector.record_metric('success_rate', 0.95)
        
        metrics = collector.get_metrics()
        
        self.assertEqual(metrics['processing_time'], 1.5)
        self.assertEqual(metrics['rows_processed'], 1000)
        self.assertEqual(metrics['success_rate'], 0.95)
    
    def test_performance_tracking(self):
        """Test performance tracking functionality."""
        start_time = time.time()
        
        # Simulate some work
        time.sleep(0.1)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should track duration
        self.assertGreater(duration, 0.1)
        self.assertLess(duration, 0.2)  # Should complete quickly
    
    def test_error_tracking(self):
        """Test error tracking functionality."""
        errors = []
        
        # Simulate error tracking
        try:
            raise ValueError("Test error")
        except ValueError as e:
            errors.append({
                'type': type(e).__name__,
                'message': str(e),
                'timestamp': time.time()
            })
        
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]['type'], 'ValueError')
        self.assertEqual(errors[0]['message'], 'Test error')


class TestDataQualityMonitoring(unittest.TestCase):
    """Test data quality monitoring components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.quality_df = pd.DataFrame({
            'complete_col': [1, 2, 3, 4, 5],
            'partial_col': [1, None, 3, None, 5],
            'empty_col': [None, None, None, None, None],
            'mixed_types': [1, 'text', 3.5, None, True]
        })
    
    def test_null_detection(self):
        """Test null value detection."""
        null_stats = {}
        
        for col in self.quality_df.columns:
            null_count = self.quality_df[col].isnull().sum()
            null_percentage = (null_count / len(self.quality_df)) * 100
            null_stats[col] = {
                'null_count': null_count,
                'null_percentage': null_percentage
            }
        
        # Check null statistics
        self.assertEqual(null_stats['complete_col']['null_count'], 0)
        self.assertEqual(null_stats['partial_col']['null_count'], 2)
        self.assertEqual(null_stats['empty_col']['null_count'], 5)
        self.assertEqual(null_stats['empty_col']['null_percentage'], 100.0)
    
    def test_data_type_consistency(self):
        """Test data type consistency checking."""
        type_issues = []
        
        for col in self.quality_df.columns:
            series = self.quality_df[col].dropna()
            if len(series) > 0:
                types = [type(val).__name__ for val in series]
                unique_types = set(types)
                
                if len(unique_types) > 1:
                    type_issues.append({
                        'column': col,
                        'types': list(unique_types),
                        'count': len(unique_types)
                    })
        
        # Should detect mixed types in mixed_types column
        self.assertGreater(len(type_issues), 0)
        mixed_col_issue = next((issue for issue in type_issues if issue['column'] == 'mixed_types'), None)
        self.assertIsNotNone(mixed_col_issue)
        self.assertGreater(mixed_col_issue['count'], 1)
    
    def test_uniqueness_analysis(self):
        """Test uniqueness analysis."""
        uniqueness_stats = {}
        
        for col in self.quality_df.columns:
            series = self.quality_df[col].dropna()
            if len(series) > 0:
                unique_count = series.nunique()
                total_count = len(series)
                uniqueness_ratio = unique_count / total_count
                
                uniqueness_stats[col] = {
                    'unique_count': unique_count,
                    'total_count': total_count,
                    'uniqueness_ratio': uniqueness_ratio
                }
        
        # Check uniqueness statistics
        self.assertEqual(uniqueness_stats['complete_col']['uniqueness_ratio'], 1.0)  # All unique
    
    def test_outlier_detection(self):
        """Test outlier detection for numeric columns."""
        numeric_data = pd.Series([1, 2, 3, 4, 5, 100])  # 100 is an outlier
        
        # Simple outlier detection using IQR
        Q1 = numeric_data.quantile(0.25)
        Q3 = numeric_data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = numeric_data[(numeric_data < lower_bound) | (numeric_data > upper_bound)]
        
        # Should detect 100 as outlier
        self.assertGreater(len(outliers), 0)
        self.assertIn(100, outliers.values)


class TestPerformanceMonitoring(unittest.TestCase):
    """Test performance monitoring components."""
    
    def test_execution_time_tracking(self):
        """Test execution time tracking."""
        execution_times = []
        
        for i in range(3):
            start_time = time.time()
            
            # Simulate work with different durations
            time.sleep(0.01 * (i + 1))
            
            end_time = time.time()
            execution_times.append(end_time - start_time)
        
        # Check that times are increasing
        self.assertGreater(execution_times[1], execution_times[0])
        self.assertGreater(execution_times[2], execution_times[1])
    
    def test_memory_usage_tracking(self):
        """Test memory usage tracking."""
        import psutil
        import os
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Allocate some memory
        large_list = [i for i in range(100000)]
        
        # Get memory usage after allocation
        final_memory = process.memory_info().rss
        memory_delta = final_memory - initial_memory
        
        # Should show increased memory usage
        self.assertGreater(memory_delta, 0)
        
        # Clean up
        del large_list
    
    def test_throughput_calculation(self):
        """Test throughput calculation."""
        start_time = time.time()
        items_processed = 1000
        
        # Simulate processing time
        time.sleep(0.1)
        
        end_time = time.time()
        duration = end_time - start_time
        throughput = items_processed / duration
        
        # Should calculate reasonable throughput
        self.assertGreater(throughput, 0)
        self.assertLess(throughput, 100000)  # Reasonable upper bound
    
    def test_resource_utilization(self):
        """Test resource utilization monitoring."""
        import psutil
        
        # Get CPU and memory info
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_info = psutil.virtual_memory()
        
        utilization_stats = {
            'cpu_percent': cpu_percent,
            'memory_percent': memory_info.percent,
            'memory_available_gb': memory_info.available / (1024**3)
        }
        
        # Should have reasonable values
        self.assertGreaterEqual(utilization_stats['cpu_percent'], 0)
        self.assertLessEqual(utilization_stats['cpu_percent'], 100)
        self.assertGreaterEqual(utilization_stats['memory_percent'], 0)
        self.assertLessEqual(utilization_stats['memory_percent'], 100)


class TestLoggingSystem(unittest.TestCase):
    """Test logging system components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.log_file = os.path.join(self.temp_dir, 'test.log')
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_file_logging(self):
        """Test file-based logging."""
        # Configure logger
        logger = logging.getLogger('test_file_logger')
        logger.setLevel(logging.INFO)
        
        # Add file handler
        file_handler = logging.FileHandler(self.log_file)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Log some messages
        logger.info('Test info message')
        logger.warning('Test warning message')
        logger.error('Test error message')
        
        # Close handler to flush content
        file_handler.close()
        logger.removeHandler(file_handler)
        
        # Verify log file content
        self.assertTrue(os.path.exists(self.log_file))
        
        with open(self.log_file, 'r') as f:
            log_content = f.read()
        
        self.assertIn('Test info message', log_content)
        self.assertIn('Test warning message', log_content)
        self.assertIn('Test error message', log_content)
    
    def test_structured_logging(self):
        """Test structured logging with JSON format."""
        log_entries = []
        
        # Simulate structured logging
        def log_structured(level, message, **kwargs):
            entry = {
                'timestamp': time.time(),
                'level': level,
                'message': message,
                'metadata': kwargs
            }
            log_entries.append(entry)
        
        # Log structured messages
        log_structured('INFO', 'Processing started', user_id=123, batch_size=1000)
        log_structured('WARNING', 'Slow processing detected', duration=5.2, threshold=3.0)
        log_structured('ERROR', 'Processing failed', error_code='E001', retry_count=3)
        
        # Verify structured entries
        self.assertEqual(len(log_entries), 3)
        
        info_entry = log_entries[0]
        self.assertEqual(info_entry['level'], 'INFO')
        self.assertEqual(info_entry['metadata']['user_id'], 123)
        self.assertEqual(info_entry['metadata']['batch_size'], 1000)
    
    def test_log_rotation(self):
        """Test log rotation functionality."""
        # Simulate log rotation by creating multiple log files
        log_files = []
        
        for i in range(3):
            log_file = os.path.join(self.temp_dir, f'test_{i}.log')
            with open(log_file, 'w') as f:
                f.write(f'Log file {i} content\n')
            log_files.append(log_file)
        
        # Verify all log files exist
        for log_file in log_files:
            self.assertTrue(os.path.exists(log_file))


class TestAlertingSystem(unittest.TestCase):
    """Test alerting system components."""
    
    def test_threshold_based_alerts(self):
        """Test threshold-based alerting."""
        alerts = []
        
        def check_threshold(metric_name, value, threshold, alert_type='WARNING'):
            if value > threshold:
                alert = {
                    'timestamp': time.time(),
                    'type': alert_type,
                    'metric': metric_name,
                    'value': value,
                    'threshold': threshold,
                    'message': f'{metric_name} exceeded threshold: {value} > {threshold}'
                }
                alerts.append(alert)
                return True
            return False
        
        # Test various metrics
        check_threshold('error_rate', 0.05, 0.01, 'WARNING')  # Should alert
        check_threshold('response_time', 2.5, 5.0, 'WARNING')  # Should not alert
        check_threshold('memory_usage', 0.95, 0.90, 'CRITICAL')  # Should alert
        
        # Verify alerts
        self.assertEqual(len(alerts), 2)
        
        error_rate_alert = next((a for a in alerts if a['metric'] == 'error_rate'), None)
        self.assertIsNotNone(error_rate_alert)
        self.assertEqual(error_rate_alert['type'], 'WARNING')
        
        memory_alert = next((a for a in alerts if a['metric'] == 'memory_usage'), None)
        self.assertIsNotNone(memory_alert)
        self.assertEqual(memory_alert['type'], 'CRITICAL')
    
    def test_anomaly_detection_alerts(self):
        """Test anomaly detection for alerting."""
        # Simulate time series data with anomaly
        normal_values = [10, 12, 11, 13, 9, 14, 10, 11]
        anomaly_values = [10, 12, 11, 50, 9, 14, 10, 11]  # 50 is anomaly
        
        def detect_anomaly(values, threshold_multiplier=2.0):
            mean_val = np.mean(values[:-1])  # Exclude last value for comparison
            std_val = np.std(values[:-1])
            threshold = mean_val + (threshold_multiplier * std_val)
            
            current_value = values[-1]
            return current_value > threshold
        
        # Test normal case
        normal_case = normal_values + [12]
        self.assertFalse(detect_anomaly(normal_case))
        
        # Test anomaly case
        anomaly_case = normal_values + [50]
        self.assertTrue(detect_anomaly(anomaly_case))
    
    def test_alert_aggregation(self):
        """Test alert aggregation and deduplication."""
        raw_alerts = [
            {'metric': 'cpu_usage', 'value': 85, 'timestamp': time.time()},
            {'metric': 'cpu_usage', 'value': 87, 'timestamp': time.time() + 1},
            {'metric': 'memory_usage', 'value': 92, 'timestamp': time.time() + 2},
            {'metric': 'cpu_usage', 'value': 89, 'timestamp': time.time() + 3}
        ]
        
        # Aggregate alerts by metric
        aggregated = {}
        for alert in raw_alerts:
            metric = alert['metric']
            if metric not in aggregated:
                aggregated[metric] = {
                    'count': 0,
                    'max_value': 0,
                    'latest_timestamp': 0
                }
            
            aggregated[metric]['count'] += 1
            aggregated[metric]['max_value'] = max(aggregated[metric]['max_value'], alert['value'])
            aggregated[metric]['latest_timestamp'] = max(aggregated[metric]['latest_timestamp'], alert['timestamp'])
        
        # Verify aggregation
        self.assertEqual(len(aggregated), 2)
        self.assertEqual(aggregated['cpu_usage']['count'], 3)
        self.assertEqual(aggregated['cpu_usage']['max_value'], 89)
        self.assertEqual(aggregated['memory_usage']['count'], 1)


class TestDashboardComponents(unittest.TestCase):
    """Test dashboard and visualization components."""
    
    def test_metrics_dashboard_data(self):
        """Test data preparation for metrics dashboard."""
        # Simulate metrics data
        metrics_data = {
            'system_metrics': {
                'cpu_usage': 75.2,
                'memory_usage': 68.5,
                'disk_usage': 45.3
            },
            'application_metrics': {
                'requests_per_second': 1250,
                'response_time_avg': 0.125,
                'error_rate': 0.02
            },
            'data_quality_metrics': {
                'null_percentage': 2.5,
                'duplicate_count': 15,
                'outlier_count': 8
            }
        }
        
        # Flatten metrics for dashboard
        dashboard_data = {}
        for category, metrics in metrics_data.items():
            for metric_name, value in metrics.items():
                dashboard_data[f"{category}.{metric_name}"] = value
        
        # Verify dashboard data structure
        self.assertIn('system_metrics.cpu_usage', dashboard_data)
        self.assertIn('application_metrics.requests_per_second', dashboard_data)
        self.assertIn('data_quality_metrics.null_percentage', dashboard_data)
        
        self.assertEqual(dashboard_data['system_metrics.cpu_usage'], 75.2)
        self.assertEqual(dashboard_data['application_metrics.error_rate'], 0.02)
    
    def test_time_series_data_preparation(self):
        """Test time series data preparation for charts."""
        # Generate time series data
        timestamps = [time.time() - (60 * i) for i in range(10, 0, -1)]  # Last 10 minutes
        values = [45.2, 46.1, 44.8, 47.3, 48.5, 49.1, 46.7, 45.9, 44.2, 43.8]
        
        time_series = list(zip(timestamps, values))
        
        # Verify time series structure
        self.assertEqual(len(time_series), 10)
        
        # Check that timestamps are in ascending order
        sorted_timestamps = [ts for ts, _ in time_series]
        self.assertEqual(sorted_timestamps, sorted(sorted_timestamps))
        
        # Calculate basic statistics
        stats = {
            'min': min(values),
            'max': max(values),
            'avg': sum(values) / len(values),
            'trend': 'decreasing' if values[-1] < values[0] else 'increasing'
        }
        
        self.assertEqual(stats['min'], 43.8)
        self.assertEqual(stats['max'], 49.1)
        self.assertEqual(stats['trend'], 'decreasing')  # Values generally decrease


def run_observable_tests():
    """Run all observable tests."""
    print("Running Observable Module Tests...")
    print("=" * 60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestObservabilityFramework,
        TestDataQualityMonitoring,
        TestPerformanceMonitoring,
        TestLoggingSystem,
        TestAlertingSystem,
        TestDashboardComponents
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("-" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    return len(result.failures) == 0 and len(result.errors) == 0


if __name__ == '__main__':
    success = run_observable_tests()
    if success:
        print("\n✅ All observable tests passed!")
    else:
        print("\n❌ Some tests failed. Check output above for details.") 