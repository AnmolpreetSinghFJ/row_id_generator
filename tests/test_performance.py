"""
Performance Testing Suite
Task 9.3: Performance tests - Load, Stress, and Benchmark Testing
"""

import unittest
import pandas as pd
import numpy as np
import time
import psutil
import gc
import threading
import sys
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from memory_profiler import profile as memory_profile
import hashlib
from io import StringIO
import tempfile

# Add the package to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class PerformanceMonitor:
    """Monitor system performance during tests."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all performance metrics."""
        self.start_time = None
        self.end_time = None
        self.start_memory = None
        self.end_memory = None
        self.peak_memory = None
        self.cpu_usage = []
        self.memory_usage = []
        self.monitoring = False
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start performance monitoring."""
        self.reset()
        self.start_time = time.time()
        self.start_memory = psutil.virtual_memory().used
        self.peak_memory = self.start_memory
        self.monitoring = True
        
        # Start background monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_resources)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop performance monitoring and return metrics."""
        self.monitoring = False
        self.end_time = time.time()
        self.end_memory = psutil.virtual_memory().used
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)
        
        return self.get_metrics()
    
    def _monitor_resources(self):
        """Background monitoring of CPU and memory."""
        while self.monitoring:
            try:
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory_info = psutil.virtual_memory()
                
                self.cpu_usage.append(cpu_percent)
                self.memory_usage.append(memory_info.used)
                
                if memory_info.used > self.peak_memory:
                    self.peak_memory = memory_info.used
                
                time.sleep(0.1)
            except:
                break
    
    def get_metrics(self):
        """Get comprehensive performance metrics."""
        duration = (self.end_time - self.start_time) if self.start_time and self.end_time else 0
        memory_delta = (self.end_memory - self.start_memory) if self.start_memory and self.end_memory else 0
        peak_memory_delta = (self.peak_memory - self.start_memory) if self.start_memory and self.peak_memory else 0
        
        return {
            'duration_seconds': duration,
            'memory_delta_mb': memory_delta / (1024 * 1024),
            'peak_memory_delta_mb': peak_memory_delta / (1024 * 1024),
            'average_cpu_percent': np.mean(self.cpu_usage) if self.cpu_usage else 0,
            'max_cpu_percent': max(self.cpu_usage) if self.cpu_usage else 0,
            'memory_samples': len(self.memory_usage),
            'cpu_samples': len(self.cpu_usage)
        }


class MockDataProcessor:
    """Mock data processor for performance testing."""
    
    @staticmethod
    def generate_test_data(size, complexity='medium'):
        """Generate test data of specified size and complexity."""
        np.random.seed(42)  # For reproducible results
        
        if complexity == 'simple':
            return pd.DataFrame({
                'id': range(size),
                'value': np.random.randn(size)
            })
        elif complexity == 'medium':
            return pd.DataFrame({
                'id': range(size),
                'name': [f'user_{i}' for i in range(size)],
                'value': np.random.randn(size),
                'category': np.random.choice(['A', 'B', 'C'], size),
                'timestamp': pd.date_range('2023-01-01', periods=size, freq='1min')
            })
        elif complexity == 'complex':
            return pd.DataFrame({
                'id': range(size),
                'name': [f'user_{i}' for i in range(size)],
                'email': [f'user{i}@example.com' for i in range(size)],
                'value': np.random.randn(size),
                'score': np.random.normal(75, 15, size),
                'category': np.random.choice(['A', 'B', 'C', 'D', 'E'], size),
                'subcategory': np.random.choice(['X', 'Y', 'Z'], size),
                'timestamp': pd.date_range('2023-01-01', periods=size, freq='1min'),
                'description': [f'Description text for record {i} with some content' for i in range(size)],
                'is_active': np.random.choice([True, False], size)
            })
    
    @staticmethod
    def preprocess_data(df):
        """Mock data preprocessing with realistic operations."""
        processed_df = df.copy()
        
        # String processing
        for col in processed_df.select_dtypes(include=['object']).columns:
            if col not in ['timestamp']:
                processed_df[col] = processed_df[col].astype(str).str.upper()
        
        # Numeric processing
        for col in processed_df.select_dtypes(include=[np.number]).columns:
            if col != 'id':
                processed_df[col] = processed_df[col].fillna(processed_df[col].mean())
        
        return processed_df
    
    @staticmethod
    def generate_row_ids(df, algorithm='sha256'):
        """Generate row IDs using specified algorithm."""
        row_ids = []
        
        for _, row in df.iterrows():
            # Create row string from all columns
            row_string = '_'.join(str(value) for value in row.values)
            
            if algorithm == 'sha256':
                row_id = hashlib.sha256(row_string.encode()).hexdigest()
            elif algorithm == 'md5':
                row_id = hashlib.md5(row_string.encode()).hexdigest()
            else:  # simple hash
                row_id = str(hash(row_string))
            
            row_ids.append(row_id)
        
        return row_ids


class TestLoadPerformance(unittest.TestCase):
    """Test system performance under various load conditions."""
    
    def setUp(self):
        """Set up performance monitoring."""
        self.monitor = PerformanceMonitor()
        self.processor = MockDataProcessor()
        
        # Performance thresholds
        self.thresholds = {
            'small_data_time': 0.1,      # 100ms for small datasets
            'medium_data_time': 1.0,     # 1s for medium datasets
            'large_data_time': 10.0,     # 10s for large datasets
            'memory_efficiency': 2.0,    # Max 2x memory usage
            'cpu_efficiency': 80.0       # Max 80% CPU usage
        }
    
    def test_small_dataset_performance(self):
        """Test performance with small datasets (1K records)."""
        print("\n⚡ Testing Small Dataset Performance (1K records)...")
        
        # Generate test data
        test_data = self.processor.generate_test_data(1000, complexity='medium')
        
        # Monitor performance
        self.monitor.start_monitoring()
        
        # Process data
        processed_data = self.processor.preprocess_data(test_data)
        row_ids = self.processor.generate_row_ids(processed_data)
        processed_data['row_id'] = row_ids
        
        # Get metrics
        metrics = self.monitor.stop_monitoring()
        
        # Validate performance
        self.assertLess(metrics['duration_seconds'], self.thresholds['small_data_time'])
        self.assertLess(metrics['average_cpu_percent'], self.thresholds['cpu_efficiency'])
        self.assertEqual(len(processed_data), 1000)
        self.assertEqual(processed_data['row_id'].nunique(), 1000)
        
        print(f"✅ Small dataset: {metrics['duration_seconds']:.3f}s, "
              f"{metrics['memory_delta_mb']:.1f}MB, "
              f"{metrics['average_cpu_percent']:.1f}% CPU")
    
    def test_medium_dataset_performance(self):
        """Test performance with medium datasets (10K records)."""
        print("\n⚡ Testing Medium Dataset Performance (10K records)...")
        
        # Generate test data
        test_data = self.processor.generate_test_data(10000, complexity='medium')
        
        # Monitor performance
        self.monitor.start_monitoring()
        
        # Process data
        processed_data = self.processor.preprocess_data(test_data)
        row_ids = self.processor.generate_row_ids(processed_data)
        processed_data['row_id'] = row_ids
        
        # Get metrics
        metrics = self.monitor.stop_monitoring()
        
        # Validate performance
        self.assertLess(metrics['duration_seconds'], self.thresholds['medium_data_time'])
        self.assertEqual(len(processed_data), 10000)
        self.assertEqual(processed_data['row_id'].nunique(), 10000)
        
        print(f"✅ Medium dataset: {metrics['duration_seconds']:.3f}s, "
              f"{metrics['memory_delta_mb']:.1f}MB, "
              f"{metrics['average_cpu_percent']:.1f}% CPU")
    
    def test_large_dataset_performance(self):
        """Test performance with large datasets (100K records)."""
        print("\n⚡ Testing Large Dataset Performance (100K records)...")
        
        # Generate test data
        test_data = self.processor.generate_test_data(100000, complexity='complex')
        
        # Monitor performance
        self.monitor.start_monitoring()
        
        # Process data
        processed_data = self.processor.preprocess_data(test_data)
        row_ids = self.processor.generate_row_ids(processed_data)
        processed_data['row_id'] = row_ids
        
        # Get metrics
        metrics = self.monitor.stop_monitoring()
        
        # Validate performance
        self.assertLess(metrics['duration_seconds'], self.thresholds['large_data_time'])
        self.assertEqual(len(processed_data), 100000)
        self.assertEqual(processed_data['row_id'].nunique(), 100000)
        
        print(f"✅ Large dataset: {metrics['duration_seconds']:.3f}s, "
              f"{metrics['memory_delta_mb']:.1f}MB, "
              f"{metrics['average_cpu_percent']:.1f}% CPU")
    
    def test_scalability_progression(self):
        """Test scalability across progressively larger datasets."""
        print("\n⚡ Testing Scalability Progression...")
        
        dataset_sizes = [1000, 5000, 10000, 25000, 50000]
        scalability_results = []
        
        for size in dataset_sizes:
            print(f"   Testing {size:,} records...")
            
            # Generate test data
            test_data = self.processor.generate_test_data(size, complexity='medium')
            
            # Monitor performance
            self.monitor.start_monitoring()
            
            # Process data
            processed_data = self.processor.preprocess_data(test_data)
            row_ids = self.processor.generate_row_ids(processed_data)
            processed_data['row_id'] = row_ids
            
            # Get metrics
            metrics = self.monitor.stop_monitoring()
            
            scalability_result = {
                'size': size,
                'duration': metrics['duration_seconds'],
                'memory_mb': metrics['memory_delta_mb'],
                'throughput_records_per_sec': size / metrics['duration_seconds'] if metrics['duration_seconds'] > 0 else 0,
                'memory_per_record_kb': (metrics['memory_delta_mb'] * 1024) / size if size > 0 else 0
            }
            scalability_results.append(scalability_result)
        
        # Analyze scalability
        throughputs = [r['throughput_records_per_sec'] for r in scalability_results]
        memory_per_record = [r['memory_per_record_kb'] for r in scalability_results]
        
        # Validate scalability characteristics
        self.assertGreater(min(throughputs), 1000)  # At least 1K records/sec
        self.assertLess(max(memory_per_record), 10)  # Less than 10KB per record
        
        print("✅ Scalability progression completed successfully")
        for result in scalability_results:
            print(f"   {result['size']:>6,} records: {result['duration']:.3f}s "
                  f"({result['throughput_records_per_sec']:,.0f} rec/s, "
                  f"{result['memory_per_record_kb']:.2f}KB/rec)")


class TestStressAndLimits(unittest.TestCase):
    """Test system under stress conditions and find limits."""
    
    def setUp(self):
        """Set up stress testing."""
        self.monitor = PerformanceMonitor()
        self.processor = MockDataProcessor()
    
    def test_memory_stress(self):
        """Test system behavior under memory stress."""
        print("\n🔥 Testing Memory Stress...")
        
        # Start with manageable size and increase until we hit limits
        current_size = 10000
        max_attempts = 5
        successful_sizes = []
        
        for attempt in range(max_attempts):
            try:
                print(f"   Attempting {current_size:,} records...")
                
                # Monitor memory before
                initial_memory = psutil.virtual_memory().used
                
                # Generate and process data
                self.monitor.start_monitoring()
                test_data = self.processor.generate_test_data(current_size, complexity='complex')
                processed_data = self.processor.preprocess_data(test_data)
                row_ids = self.processor.generate_row_ids(processed_data)
                processed_data['row_id'] = row_ids
                metrics = self.monitor.stop_monitoring()
                
                successful_sizes.append({
                    'size': current_size,
                    'duration': metrics['duration_seconds'],
                    'memory_mb': metrics['memory_delta_mb'],
                    'peak_memory_mb': metrics['peak_memory_delta_mb']
                })
                
                print(f"   ✅ Success: {metrics['duration_seconds']:.3f}s, "
                      f"{metrics['peak_memory_delta_mb']:.1f}MB peak")
                
                # Cleanup
                del test_data, processed_data, row_ids
                gc.collect()
                
                # Increase size for next attempt
                current_size *= 2
                
            except Exception as e:
                print(f"   ❌ Failed at {current_size:,} records: {str(e)}")
                break
        
        # Validate stress test results
        self.assertGreater(len(successful_sizes), 0)
        largest_successful = max(successful_sizes, key=lambda x: x['size'])
        self.assertGreater(largest_successful['size'], 10000)
        
        print(f"✅ Memory stress test completed. Largest dataset: {largest_successful['size']:,} records")
    
    def test_concurrent_processing_stress(self):
        """Test system under concurrent processing stress."""
        print("\n🔥 Testing Concurrent Processing Stress...")
        
        # Test different levels of concurrency
        concurrency_levels = [2, 4, 8]
        dataset_size = 5000
        
        results = []
        
        for num_workers in concurrency_levels:
            print(f"   Testing {num_workers} concurrent workers...")
            
            def process_batch(worker_id):
                """Process a batch of data."""
                test_data = self.processor.generate_test_data(dataset_size, complexity='medium')
                processed_data = self.processor.preprocess_data(test_data)
                row_ids = self.processor.generate_row_ids(processed_data)
                processed_data['row_id'] = row_ids
                return len(processed_data)
            
            # Monitor concurrent processing
            self.monitor.start_monitoring()
            
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(process_batch, i) for i in range(num_workers)]
                completed_batches = [future.result() for future in futures]
            
            metrics = self.monitor.stop_monitoring()
            
            total_records = sum(completed_batches)
            throughput = total_records / metrics['duration_seconds'] if metrics['duration_seconds'] > 0 else 0
            
            result = {
                'workers': num_workers,
                'total_records': total_records,
                'duration': metrics['duration_seconds'],
                'throughput': throughput,
                'avg_cpu': metrics['average_cpu_percent'],
                'memory_mb': metrics['memory_delta_mb']
            }
            results.append(result)
            
            print(f"   ✅ {num_workers} workers: {throughput:,.0f} rec/s, "
                  f"{metrics['average_cpu_percent']:.1f}% CPU")
        
        # Validate concurrent processing
        self.assertTrue(all(r['total_records'] > 0 for r in results))
        throughputs = [r['throughput'] for r in results]
        self.assertGreater(max(throughputs), 1000)  # At least 1K records/sec
        
        print("✅ Concurrent processing stress test completed")
    
    def test_algorithm_performance_comparison(self):
        """Test performance of different hashing algorithms."""
        print("\n🔥 Testing Algorithm Performance Comparison...")
        
        algorithms = ['sha256', 'md5', 'simple']
        dataset_size = 10000
        test_data = self.processor.generate_test_data(dataset_size, complexity='medium')
        
        algorithm_results = []
        
        for algorithm in algorithms:
            print(f"   Testing {algorithm} algorithm...")
            
            # Monitor algorithm performance
            self.monitor.start_monitoring()
            
            processed_data = self.processor.preprocess_data(test_data.copy())
            row_ids = self.processor.generate_row_ids(processed_data, algorithm=algorithm)
            processed_data['row_id'] = row_ids
            
            metrics = self.monitor.stop_monitoring()
            
            # Check uniqueness
            unique_ids = processed_data['row_id'].nunique()
            uniqueness_ratio = unique_ids / len(processed_data)
            
            result = {
                'algorithm': algorithm,
                'duration': metrics['duration_seconds'],
                'throughput': dataset_size / metrics['duration_seconds'] if metrics['duration_seconds'] > 0 else 0,
                'memory_mb': metrics['memory_delta_mb'],
                'uniqueness_ratio': uniqueness_ratio,
                'unique_ids': unique_ids
            }
            algorithm_results.append(result)
            
            print(f"   ✅ {algorithm}: {result['throughput']:,.0f} rec/s, "
                  f"{result['uniqueness_ratio']:.4f} uniqueness")
        
        # Validate algorithm comparison
        for result in algorithm_results:
            self.assertGreater(result['throughput'], 100)  # At least 100 rec/s
            self.assertEqual(result['uniqueness_ratio'], 1.0)  # Perfect uniqueness
        
        # Find fastest algorithm
        fastest = max(algorithm_results, key=lambda x: x['throughput'])
        print(f"✅ Fastest algorithm: {fastest['algorithm']} at {fastest['throughput']:,.0f} rec/s")


class TestBenchmarkBaselines(unittest.TestCase):
    """Establish performance benchmarks and baselines."""
    
    def setUp(self):
        """Set up benchmark testing."""
        self.monitor = PerformanceMonitor()
        self.processor = MockDataProcessor()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up benchmark testing."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_baseline_performance_metrics(self):
        """Establish baseline performance metrics for standard operations."""
        print("\n📊 Establishing Baseline Performance Metrics...")
        
        standard_dataset_size = 10000
        baselines = {}
        
        # Baseline 1: Data Generation
        self.monitor.start_monitoring()
        test_data = self.processor.generate_test_data(standard_dataset_size, complexity='medium')
        baselines['data_generation'] = self.monitor.stop_monitoring()
        
        # Baseline 2: Data Preprocessing
        self.monitor.start_monitoring()
        processed_data = self.processor.preprocess_data(test_data.copy())
        baselines['data_preprocessing'] = self.monitor.stop_monitoring()
        
        # Baseline 3: Row ID Generation
        self.monitor.start_monitoring()
        row_ids = self.processor.generate_row_ids(processed_data.copy())
        baselines['row_id_generation'] = self.monitor.stop_monitoring()
        
        # Baseline 4: Data Combination
        self.monitor.start_monitoring()
        final_data = processed_data.copy()
        final_data['row_id'] = row_ids
        baselines['data_combination'] = self.monitor.stop_monitoring()
        
        # Baseline 5: File I/O
        output_file = os.path.join(self.temp_dir, 'benchmark_output.csv')
        self.monitor.start_monitoring()
        final_data.to_csv(output_file, index=False)
        loaded_data = pd.read_csv(output_file)
        baselines['file_io'] = self.monitor.stop_monitoring()
        
        # Print baseline results
        print("   Baseline Performance Metrics:")
        for operation, metrics in baselines.items():
            throughput = standard_dataset_size / metrics['duration_seconds'] if metrics['duration_seconds'] > 0 else 0
            print(f"   {operation:>20}: {metrics['duration_seconds']:.4f}s "
                  f"({throughput:>8.0f} rec/s, {metrics['memory_delta_mb']:>6.1f}MB)")
        
        # Validate baselines
        total_time = sum(m['duration_seconds'] for m in baselines.values())
        total_memory = sum(m['memory_delta_mb'] for m in baselines.values())
        
        self.assertLess(total_time, 5.0)  # Total processing under 5 seconds
        self.assertLess(total_memory, 100)  # Total memory under 100MB
        self.assertEqual(len(final_data), standard_dataset_size)
        self.assertEqual(len(loaded_data), standard_dataset_size)
        
        print(f"✅ Baselines established: {total_time:.3f}s total, {total_memory:.1f}MB total")
    
    def test_performance_regression_detection(self):
        """Test performance regression detection capabilities."""
        print("\n📊 Testing Performance Regression Detection...")
        
        dataset_size = 5000
        
        # Run baseline measurement
        baseline_runs = []
        for run in range(3):
            test_data = self.processor.generate_test_data(dataset_size, complexity='medium')
            
            self.monitor.start_monitoring()
            processed_data = self.processor.preprocess_data(test_data)
            row_ids = self.processor.generate_row_ids(processed_data)
            processed_data['row_id'] = row_ids
            metrics = self.monitor.stop_monitoring()
            
            baseline_runs.append(metrics['duration_seconds'])
        
        baseline_avg = np.mean(baseline_runs)
        baseline_std = np.std(baseline_runs)
        
        # Simulate performance regression (add artificial delay)
        def slow_preprocess_data(df):
            """Simulated slow preprocessing for regression testing."""
            time.sleep(0.1)  # Add 100ms delay
            return self.processor.preprocess_data(df)
        
        # Run regression test
        test_data = self.processor.generate_test_data(dataset_size, complexity='medium')
        
        self.monitor.start_monitoring()
        processed_data = slow_preprocess_data(test_data)
        row_ids = self.processor.generate_row_ids(processed_data)
        processed_data['row_id'] = row_ids
        regression_metrics = self.monitor.stop_monitoring()
        
        # Detect regression
        regression_duration = regression_metrics['duration_seconds']
        performance_degradation = ((regression_duration - baseline_avg) / baseline_avg) * 100
        
        # Regression detection logic
        regression_threshold = 50  # 50% performance degradation threshold
        is_regression = performance_degradation > regression_threshold
        
        print(f"   Baseline average: {baseline_avg:.4f}s (±{baseline_std:.4f})")
        print(f"   Regression test: {regression_duration:.4f}s")
        print(f"   Performance change: {performance_degradation:+.1f}%")
        print(f"   Regression detected: {'Yes' if is_regression else 'No'}")
        
        # Validate regression detection
        self.assertTrue(is_regression)  # Should detect the artificial regression
        self.assertGreater(performance_degradation, regression_threshold)
        
        print("✅ Regression detection test completed successfully")


def run_performance_tests():
    """Run all performance tests."""
    print("Running Performance Test Suite...")
    print("=" * 70)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestLoadPerformance,
        TestStressAndLimits,
        TestBenchmarkBaselines
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("-" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    return len(result.failures) == 0 and len(result.errors) == 0


if __name__ == '__main__':
    success = run_performance_tests()
    if success:
        print("\n✅ All performance tests passed!")
    else:
        print("\n❌ Some performance tests failed. Check output above for details.") 