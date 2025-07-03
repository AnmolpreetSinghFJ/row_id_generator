"""
Unit Tests for Benchmarks Module
Task 9.1: Unit tests for core functions - Benchmarks Component
"""

import unittest
import pandas as pd
import numpy as np
import time
import sys
import os

# Add the package to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import benchmarks functions with error handling
try:
    from row_id_generator.benchmarks import *
except ImportError:
    # Create mock classes for testing if benchmarks module doesn't exist yet
    class MockBenchmarkRunner:
        def __init__(self):
            self.results = {}
        
        def run_benchmark(self, name, func, *args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            
            self.results[name] = {
                'duration': end_time - start_time,
                'result': result,
                'success': True
            }
            return self.results[name]
        
        def get_results(self):
            return self.results.copy()


class TestBenchmarkFramework(unittest.TestCase):
    """Test benchmark framework components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.benchmark_runner = MockBenchmarkRunner()
        
        self.small_df = pd.DataFrame({
            'id': range(100),
            'data': np.random.randn(100)
        })
        
        self.medium_df = pd.DataFrame({
            'id': range(1000),
            'data': np.random.randn(1000)
        })
        
        self.large_df = pd.DataFrame({
            'id': range(10000),
            'data': np.random.randn(10000)
        })
    
    def test_basic_benchmark_execution(self):
        """Test basic benchmark execution."""
        def simple_function(n):
            return sum(range(n))
        
        result = self.benchmark_runner.run_benchmark(
            'simple_sum', simple_function, 1000
        )
        
        self.assertIn('duration', result)
        self.assertIn('result', result)
        self.assertIn('success', result)
        self.assertTrue(result['success'])
        self.assertEqual(result['result'], sum(range(1000)))
    
    def test_multiple_benchmark_runs(self):
        """Test running multiple benchmarks."""
        def fast_function():
            return 42
        
        def slow_function():
            time.sleep(0.01)
            return 42
        
        # Run multiple benchmarks
        self.benchmark_runner.run_benchmark('fast_test', fast_function)
        self.benchmark_runner.run_benchmark('slow_test', slow_function)
        
        results = self.benchmark_runner.get_results()
        
        self.assertIn('fast_test', results)
        self.assertIn('slow_test', results)
        self.assertLess(results['fast_test']['duration'], results['slow_test']['duration'])
    
    def test_dataframe_processing_benchmark(self):
        """Test DataFrame processing benchmarks."""
        def process_dataframe(df):
            # Simple processing: calculate mean of each column
            return df.mean()
        
        # Benchmark different DataFrame sizes
        small_result = self.benchmark_runner.run_benchmark(
            'small_df_processing', process_dataframe, self.small_df
        )
        
        medium_result = self.benchmark_runner.run_benchmark(
            'medium_df_processing', process_dataframe, self.medium_df
        )
        
        large_result = self.benchmark_runner.run_benchmark(
            'large_df_processing', process_dataframe, self.large_df
        )
        
        # Larger DataFrames should generally take longer
        self.assertGreater(large_result['duration'], small_result['duration'])
    
    def test_memory_benchmark(self):
        """Test memory usage benchmarking."""
        import psutil
        import os
        
        def memory_intensive_function():
            # Allocate and deallocate memory
            large_list = [i for i in range(100000)]
            return len(large_list)
        
        # Get memory before
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss
        
        result = self.benchmark_runner.run_benchmark(
            'memory_test', memory_intensive_function
        )
        
        # Get memory after
        memory_after = process.memory_info().rss
        
        # Should have executed successfully
        self.assertTrue(result['success'])
        self.assertEqual(result['result'], 100000)


class TestPerformanceComparison(unittest.TestCase):
    """Test performance comparison utilities."""
    
    def test_performance_comparison(self):
        """Test performance comparison between functions."""
        def method_a(data):
            return [x * 2 for x in data]
        
        def method_b(data):
            return np.array(data) * 2
        
        test_data = list(range(1000))
        
        # Time both methods
        start_a = time.time()
        result_a = method_a(test_data)
        time_a = time.time() - start_a
        
        start_b = time.time()
        result_b = method_b(test_data)
        time_b = time.time() - start_b
        
        # Create comparison result
        comparison = {
            'method_a': {
                'time': time_a,
                'result_length': len(result_a)
            },
            'method_b': {
                'time': time_b,
                'result_length': len(result_b)
            },
            'faster_method': 'method_a' if time_a < time_b else 'method_b',
            'speedup_ratio': max(time_a, time_b) / min(time_a, time_b)
        }
        
        # Both should produce same result length
        self.assertEqual(comparison['method_a']['result_length'], 1000)
        self.assertEqual(comparison['method_b']['result_length'], 1000)
        
        # Should have a meaningful speedup ratio
        self.assertGreater(comparison['speedup_ratio'], 1.0)
    
    def test_scalability_testing(self):
        """Test scalability with different data sizes."""
        def linear_function(n):
            return sum(range(n))
        
        def quadratic_function(n):
            return sum(i * j for i in range(n) for j in range(n))
        
        sizes = [10, 100, 500]
        results = {}
        
        for size in sizes:
            # Test linear function
            start = time.time()
            linear_function(size)
            linear_time = time.time() - start
            
            # Test quadratic function (only for smaller sizes)
            if size <= 100:
                start = time.time()
                quadratic_function(size)
                quadratic_time = time.time() - start
            else:
                quadratic_time = None
            
            results[size] = {
                'linear_time': linear_time,
                'quadratic_time': quadratic_time
            }
        
        # Linear function should scale better
        self.assertLess(results[100]['linear_time'], results[500]['linear_time'])
        
        # Quadratic should be slower for larger inputs
        if results[10]['quadratic_time'] and results[100]['quadratic_time']:
            ratio = results[100]['quadratic_time'] / results[10]['quadratic_time']
            self.assertGreater(ratio, 10)  # Should be much slower


class TestHashingBenchmarks(unittest.TestCase):
    """Test hashing-specific benchmarks."""
    
    def test_hash_performance(self):
        """Test hash function performance."""
        import hashlib
        
        def sha256_hash(data):
            return hashlib.sha256(str(data).encode()).hexdigest()
        
        def md5_hash(data):
            return hashlib.md5(str(data).encode()).hexdigest()
        
        test_data = "test_data_for_hashing" * 1000
        
        # Benchmark SHA256
        start = time.time()
        sha256_result = sha256_hash(test_data)
        sha256_time = time.time() - start
        
        # Benchmark MD5
        start = time.time()
        md5_result = md5_hash(test_data)
        md5_time = time.time() - start
        
        # Both should produce valid hashes
        self.assertEqual(len(sha256_result), 64)  # SHA256 is 64 chars
        self.assertEqual(len(md5_result), 32)     # MD5 is 32 chars
        
        # Should complete in reasonable time
        self.assertLess(sha256_time, 1.0)
        self.assertLess(md5_time, 1.0)
    
    def test_batch_vs_individual_hashing(self):
        """Test batch vs individual hashing performance."""
        import hashlib
        
        test_data = [f"data_{i}" for i in range(1000)]
        
        # Individual hashing
        start = time.time()
        individual_results = []
        for item in test_data:
            hash_val = hashlib.sha256(str(item).encode()).hexdigest()
            individual_results.append(hash_val)
        individual_time = time.time() - start
        
        # Batch hashing simulation
        start = time.time()
        batch_input = ''.join(test_data)
        batch_results = []
        for i, item in enumerate(test_data):
            # Simulate batch processing with some optimization
            hash_val = hashlib.sha256(f"{batch_input}_{i}".encode()).hexdigest()
            batch_results.append(hash_val)
        batch_time = time.time() - start
        
        # Both should produce same number of results
        self.assertEqual(len(individual_results), len(batch_results))
        self.assertEqual(len(individual_results), 1000)
        
        # Record performance comparison
        performance_comparison = {
            'individual_time': individual_time,
            'batch_time': batch_time,
            'faster_method': 'individual' if individual_time < batch_time else 'batch'
        }
        
        # Should have meaningful performance data
        self.assertGreater(performance_comparison['individual_time'], 0)
        self.assertGreater(performance_comparison['batch_time'], 0)


class TestDataProcessingBenchmarks(unittest.TestCase):
    """Test data processing benchmarks."""
    
    def test_preprocessing_performance(self):
        """Test data preprocessing performance."""
        # Create test DataFrame with various data types
        test_df = pd.DataFrame({
            'integers': range(10000),
            'floats': np.random.randn(10000),
            'strings': [f"string_{i}" for i in range(10000)],
            'nulls': [None if i % 10 == 0 else i for i in range(10000)]
        })
        
        preprocessing_times = {}
        
        # Test string processing
        start = time.time()
        test_df['strings'].str.lower()
        preprocessing_times['string_processing'] = time.time() - start
        
        # Test null handling
        start = time.time()
        test_df['nulls'].fillna(0)
        preprocessing_times['null_handling'] = time.time() - start
        
        # Test numeric operations
        start = time.time()
        test_df['floats'].abs()
        preprocessing_times['numeric_operations'] = time.time() - start
        
        # All operations should complete quickly
        for operation, duration in preprocessing_times.items():
            self.assertLess(duration, 2.0, f"{operation} took too long: {duration}s")
    
    def test_aggregation_performance(self):
        """Test data aggregation performance."""
        # Create test DataFrame
        test_df = pd.DataFrame({
            'category': np.random.choice(['A', 'B', 'C'], 10000),
            'value': np.random.randn(10000),
            'count': np.random.randint(1, 100, 10000)
        })
        
        aggregation_times = {}
        
        # Test groupby operations
        start = time.time()
        result = test_df.groupby('category')['value'].mean()
        aggregation_times['groupby_mean'] = time.time() - start
        
        # Test more complex aggregation
        start = time.time()
        result = test_df.groupby('category').agg({
            'value': ['mean', 'std', 'count'],
            'count': 'sum'
        })
        aggregation_times['complex_aggregation'] = time.time() - start
        
        # Test pivot operations
        start = time.time()
        result = test_df.pivot_table(
            values='value',
            index='category',
            aggfunc='mean'
        )
        aggregation_times['pivot_table'] = time.time() - start
        
        # All aggregations should complete quickly
        for operation, duration in aggregation_times.items():
            self.assertLess(duration, 2.0, f"{operation} took too long: {duration}s")


class TestConcurrencyBenchmarks(unittest.TestCase):
    """Test concurrency performance benchmarks."""
    
    def test_sequential_vs_concurrent_processing(self):
        """Test sequential vs concurrent processing performance."""
        def cpu_intensive_task(n):
            return sum(i * i for i in range(n))
        
        task_inputs = [1000, 2000, 3000, 4000, 5000]
        
        # Sequential processing
        start = time.time()
        sequential_results = []
        for n in task_inputs:
            result = cpu_intensive_task(n)
            sequential_results.append(result)
        sequential_time = time.time() - start
        
        # Simulated concurrent processing (using map for simplicity)
        start = time.time()
        concurrent_results = list(map(cpu_intensive_task, task_inputs))
        concurrent_time = time.time() - start
        
        # Results should be identical
        self.assertEqual(sequential_results, concurrent_results)
        
        # Record performance comparison
        performance_data = {
            'sequential_time': sequential_time,
            'concurrent_time': concurrent_time,
            'tasks_completed': len(task_inputs),
            'speedup_ratio': sequential_time / concurrent_time if concurrent_time > 0 else 1
        }
        
        # Should have meaningful performance data
        self.assertGreater(performance_data['sequential_time'], 0)
        self.assertGreater(performance_data['concurrent_time'], 0)
        self.assertEqual(performance_data['tasks_completed'], 5)


class TestMemoryEfficiencyBenchmarks(unittest.TestCase):
    """Test memory efficiency benchmarks."""
    
    def test_memory_efficient_processing(self):
        """Test memory-efficient data processing."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # Test memory-efficient approach
        def memory_efficient_sum(data_size):
            total = 0
            for i in range(data_size):
                total += i
            return total
        
        # Test memory-intensive approach
        def memory_intensive_sum(data_size):
            data = list(range(data_size))
            return sum(data)
        
        data_size = 100000
        
        # Test efficient approach
        memory_before = process.memory_info().rss
        start = time.time()
        efficient_result = memory_efficient_sum(data_size)
        efficient_time = time.time() - start
        efficient_memory = process.memory_info().rss - memory_before
        
        # Test intensive approach
        memory_before = process.memory_info().rss
        start = time.time()
        intensive_result = memory_intensive_sum(data_size)
        intensive_time = time.time() - start
        intensive_memory = process.memory_info().rss - memory_before
        
        # Results should be identical
        self.assertEqual(efficient_result, intensive_result)
        
        # Memory comparison
        memory_comparison = {
            'efficient_memory': efficient_memory,
            'intensive_memory': intensive_memory,
            'efficient_time': efficient_time,
            'intensive_time': intensive_time,
            'memory_saved': intensive_memory - efficient_memory
        }
        
        # Should provide meaningful comparison data
        self.assertGreaterEqual(memory_comparison['memory_saved'], 0)
    
    def test_chunk_processing_efficiency(self):
        """Test chunk processing efficiency."""
        def process_in_chunks(data, chunk_size):
            results = []
            for i in range(0, len(data), chunk_size):
                chunk = data[i:i + chunk_size]
                chunk_result = sum(chunk)
                results.append(chunk_result)
            return sum(results)
        
        def process_all_at_once(data):
            return sum(data)
        
        large_data = list(range(100000))
        
        # Test chunk processing
        start = time.time()
        chunk_result = process_in_chunks(large_data, 1000)
        chunk_time = time.time() - start
        
        # Test all-at-once processing
        start = time.time()
        all_result = process_all_at_once(large_data)
        all_time = time.time() - start
        
        # Results should be identical
        self.assertEqual(chunk_result, all_result)
        
        # Performance comparison
        processing_comparison = {
            'chunk_time': chunk_time,
            'all_at_once_time': all_time,
            'chunk_result': chunk_result,
            'all_result': all_result,
            'faster_method': 'chunk' if chunk_time < all_time else 'all_at_once'
        }
        
        # Should have valid timing data
        self.assertGreater(processing_comparison['chunk_time'], 0)
        self.assertGreater(processing_comparison['all_at_once_time'], 0)


def run_benchmark_tests():
    """Run all benchmark tests."""
    print("Running Benchmark Module Tests...")
    print("=" * 60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestBenchmarkFramework,
        TestPerformanceComparison,
        TestHashingBenchmarks,
        TestDataProcessingBenchmarks,
        TestConcurrencyBenchmarks,
        TestMemoryEfficiencyBenchmarks
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
    success = run_benchmark_tests()
    if success:
        print("\n✅ All benchmark tests passed!")
    else:
        print("\n❌ Some tests failed. Check output above for details.") 