#!/usr/bin/env python3
"""
Comprehensive Benchmarking System for Row ID Generation Performance Optimization
Task 7.4 - Benchmark performance

This module provides comprehensive benchmarking capabilities to validate
performance optimizations and compare baseline vs optimized implementations.
"""

import sys
import time
import gc
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
import pandas as pd
import numpy as np
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not available, memory monitoring will be limited")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerformanceBenchmarkSuite:
    """
    Comprehensive benchmarking suite for performance optimization validation.
    Tests various data sizes, processing scenarios, and edge cases.
    """
    
    def __init__(self, 
                 test_data_sizes: List[int] = [1000, 10000, 50000, 100000, 500000],
                 warmup_runs: int = 2,
                 benchmark_runs: int = 5,
                 enable_memory_profiling: bool = True):
        self.test_data_sizes = test_data_sizes
        self.warmup_runs = warmup_runs
        self.benchmark_runs = benchmark_runs
        self.enable_memory_profiling = enable_memory_profiling
        
        # Benchmark results storage
        self.benchmark_results = {}
        self.baseline_results = {}
        self.optimization_metrics = {}
        
        # System info
        self.system_info = self._collect_system_info()
        
    def _collect_system_info(self) -> Dict[str, Any]:
        """Collect system information for benchmark context."""
        try:
            import platform
            
            info = {
                'platform': platform.platform(),
                'processor': platform.processor(),
                'python_version': platform.python_version(),
                'pandas_version': pd.__version__,
                'numpy_version': np.__version__
            }
            
            if PSUTIL_AVAILABLE:
                info.update({
                    'cpu_count': psutil.cpu_count(),
                    'total_memory_gb': psutil.virtual_memory().total / (1024**3)
                })
            else:
                import multiprocessing
                info['cpu_count'] = multiprocessing.cpu_count()
                info['total_memory_gb'] = 'unknown'
                
            return info
        except Exception as e:
            logger.warning(f"Error collecting system info: {e}")
            return {'error': 'System info collection failed'}
    
    def generate_test_data(self, size: int, complexity: str = 'medium') -> pd.DataFrame:
        """
        Generate test data with varying complexity characteristics.
        
        Args:
            size: Number of rows to generate
            complexity: Data complexity level ('simple', 'medium', 'complex')
            
        Returns:
            Generated test DataFrame
        """
        np.random.seed(42)  # For reproducible results
        
        if complexity == 'simple':
            return pd.DataFrame({
                'id': range(size),
                'category': np.random.choice(['A', 'B', 'C'], size),
                'value': np.random.randint(1, 100, size)
            })
        
        elif complexity == 'medium':
            return pd.DataFrame({
                'id': range(size),
                'category': np.random.choice(['Cat1', 'Cat2', 'Cat3', 'Cat4', 'Cat5'], size),
                'subcategory': np.random.choice(['Sub1', 'Sub2', 'Sub3'], size),
                'value1': np.random.randn(size),
                'value2': np.random.randint(1, 1000, size),
                'timestamp': pd.date_range(start='2023-01-01', periods=size, freq='1H')
            })
        
        elif complexity == 'complex':
            # Complex data with varying string lengths and special characters
            long_strings = [f"complex_string_{i}_{'x' * np.random.randint(10, 50)}" 
                           for i in range(size)]
            
            return pd.DataFrame({
                'id': range(size),
                'category': np.random.choice(['Category_A_Long', 'Category_B_Medium', 'Cat_C'], size),
                'subcategory': np.random.choice(['SubCat_1', 'SubCat_2', 'SubCat_3'], size),
                'description': long_strings,
                'value1': np.random.randn(size),
                'value2': np.random.randint(1, 10000, size),
                'value3': np.random.uniform(-1000, 1000, size),
                'timestamp': pd.date_range(start='2023-01-01', periods=size, freq='1min'),
                'flag': np.random.choice([True, False], size)
            })
        
        else:
            raise ValueError(f"Unknown complexity level: {complexity}")
    
    def benchmark_function(self, 
                         func: Callable,
                         test_data: pd.DataFrame,
                         function_name: str,
                         **func_kwargs) -> Dict[str, Any]:
        """
        Benchmark a single function with comprehensive metrics collection.
        
        Args:
            func: Function to benchmark
            test_data: Test data to use
            function_name: Name for the function in results
            **func_kwargs: Additional arguments for the function
            
        Returns:
            Benchmark results dictionary
        """
        # Warmup runs
        for _ in range(self.warmup_runs):
            try:
                _ = func(test_data.copy(), **func_kwargs)
                gc.collect()
            except Exception:
                pass  # Ignore warmup errors
        
        # Benchmark runs
        execution_times = []
        memory_usages = []
        rows_per_second_values = []
        
        for run in range(self.benchmark_runs):
            # Prepare clean test data
            test_df = test_data.copy()
            
            # Time the execution
            start_time = time.time()
            start_memory = self._get_memory_usage()
            
            try:
                result = func(test_df, **func_kwargs)
                success = True
                result_rows = len(result) if hasattr(result, '__len__') else len(test_df)
            except Exception as e:
                success = False
                result_rows = 0
                logger.error(f"Benchmark run {run} failed for {function_name}: {e}")
            
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            execution_time = end_time - start_time
            memory_delta = max(0, end_memory - start_memory)
            
            if success and execution_time > 0:
                execution_times.append(execution_time)
                memory_usages.append(memory_delta)
                rows_per_second = result_rows / execution_time
                rows_per_second_values.append(rows_per_second)
            
            # Cleanup
            del test_df
            if 'result' in locals():
                del result
            gc.collect()
        
        # Calculate statistics
        if execution_times:
            return {
                'function_name': function_name,
                'success_rate': len(execution_times) / self.benchmark_runs,
                'execution_time': {
                    'mean': np.mean(execution_times),
                    'std': np.std(execution_times),
                    'min': min(execution_times),
                    'max': max(execution_times),
                    'median': np.median(execution_times)
                },
                'memory_usage_mb': {
                    'mean': np.mean(memory_usages) if memory_usages else 0,
                    'std': np.std(memory_usages) if memory_usages else 0,
                    'min': min(memory_usages) if memory_usages else 0,
                    'max': max(memory_usages) if memory_usages else 0
                },
                'throughput_rows_per_second': {
                    'mean': np.mean(rows_per_second_values),
                    'std': np.std(rows_per_second_values),
                    'min': min(rows_per_second_values),
                    'max': max(rows_per_second_values)
                },
                'test_data_size': len(test_data)
            }
        else:
            return {
                'function_name': function_name,
                'success_rate': 0,
                'error': 'All benchmark runs failed',
                'test_data_size': len(test_data)
            }
    
    def run_comparative_benchmark(self,
                                baseline_func: Callable,
                                optimized_func: Callable,
                                complexity_levels: List[str] = ['simple', 'medium', 'complex'],
                                **func_kwargs) -> Dict[str, Any]:
        """
        Run comparative benchmarks between baseline and optimized functions.
        
        Args:
            baseline_func: Original/baseline function
            optimized_func: Optimized function to compare
            complexity_levels: List of data complexity levels to test
            **func_kwargs: Additional arguments for functions
            
        Returns:
            Comprehensive comparison results
        """
        comparison_results = {
            'system_info': self.system_info,
            'test_configuration': {
                'data_sizes': self.test_data_sizes,
                'complexity_levels': complexity_levels,
                'warmup_runs': self.warmup_runs,
                'benchmark_runs': self.benchmark_runs
            },
            'results_by_size': {},
            'overall_comparison': {},
            'recommendations': []
        }
        
        print(f"🚀 Running comparative benchmarks...")
        print(f"   Data sizes: {self.test_data_sizes}")
        print(f"   Complexity levels: {complexity_levels}")
        print(f"   Benchmark runs per test: {self.benchmark_runs}")
        
        for size in self.test_data_sizes:
            comparison_results['results_by_size'][size] = {}
            
            for complexity in complexity_levels:
                print(f"\n📊 Testing {size:,} rows, {complexity} complexity...")
                
                # Generate test data
                test_data = self.generate_test_data(size, complexity)
                
                # Benchmark baseline function
                print(f"   Benchmarking baseline function...")
                baseline_results = self.benchmark_function(
                    baseline_func, test_data, f'baseline_{complexity}', **func_kwargs
                )
                
                # Benchmark optimized function
                print(f"   Benchmarking optimized function...")
                optimized_results = self.benchmark_function(
                    optimized_func, test_data, f'optimized_{complexity}', **func_kwargs
                )
                
                # Calculate improvement metrics
                improvement_metrics = self._calculate_improvement_metrics(
                    baseline_results, optimized_results
                )
                
                # Store results
                comparison_results['results_by_size'][size][complexity] = {
                    'baseline': baseline_results,
                    'optimized': optimized_results,
                    'improvement': improvement_metrics
                }
                
                # Print summary
                if improvement_metrics.get('speed_improvement_factor', 0) > 0:
                    print(f"   ✅ Speed improvement: {improvement_metrics['speed_improvement_factor']:.2f}x")
                    print(f"   📈 Throughput: {optimized_results['throughput_rows_per_second']['mean']:,.0f} rows/sec")
        
        # Calculate overall comparison metrics
        comparison_results['overall_comparison'] = self._calculate_overall_comparison(
            comparison_results['results_by_size']
        )
        
        # Generate recommendations
        comparison_results['recommendations'] = self._generate_performance_recommendations(
            comparison_results
        )
        
        return comparison_results
    
    def _calculate_improvement_metrics(self,
                                     baseline: Dict[str, Any],
                                     optimized: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate performance improvement metrics."""
        if baseline.get('success_rate', 0) == 0 or optimized.get('success_rate', 0) == 0:
            return {'error': 'One or both functions failed'}
        
        baseline_time = baseline['execution_time']['mean']
        optimized_time = optimized['execution_time']['mean']
        
        baseline_memory = baseline['memory_usage_mb']['mean']
        optimized_memory = optimized['memory_usage_mb']['mean']
        
        baseline_throughput = baseline['throughput_rows_per_second']['mean']
        optimized_throughput = optimized['throughput_rows_per_second']['mean']
        
        return {
            'speed_improvement_factor': baseline_time / optimized_time if optimized_time > 0 else 0,
            'speed_improvement_percent': ((baseline_time - optimized_time) / baseline_time * 100) if baseline_time > 0 else 0,
            'memory_reduction_percent': ((baseline_memory - optimized_memory) / baseline_memory * 100) if baseline_memory > 0 else 0,
            'throughput_improvement_factor': optimized_throughput / baseline_throughput if baseline_throughput > 0 else 0,
            'throughput_improvement_percent': ((optimized_throughput - baseline_throughput) / baseline_throughput * 100) if baseline_throughput > 0 else 0,
            'efficiency_score': (optimized_throughput / max(optimized_memory, 1)) / (baseline_throughput / max(baseline_memory, 1)) if baseline_memory > 0 and optimized_memory > 0 else 0
        }
    
    def _calculate_overall_comparison(self, results_by_size: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall comparison metrics across all tests."""
        speed_improvements = []
        memory_reductions = []
        throughput_improvements = []
        
        for size_results in results_by_size.values():
            for complexity_results in size_results.values():
                improvement = complexity_results.get('improvement', {})
                
                if 'speed_improvement_factor' in improvement:
                    speed_improvements.append(improvement['speed_improvement_factor'])
                    memory_reductions.append(improvement.get('memory_reduction_percent', 0))
                    throughput_improvements.append(improvement.get('throughput_improvement_factor', 0))
        
        if speed_improvements:
            return {
                'average_speed_improvement': np.mean(speed_improvements),
                'median_speed_improvement': np.median(speed_improvements),
                'best_speed_improvement': max(speed_improvements),
                'average_memory_reduction_percent': np.mean(memory_reductions),
                'average_throughput_improvement': np.mean(throughput_improvements),
                'consistent_improvement': min(speed_improvements) > 1.0,
                'significant_improvement': np.mean(speed_improvements) > 1.5
            }
        else:
            return {'error': 'No valid improvement metrics calculated'}
    
    def _generate_performance_recommendations(self, comparison_results: Dict[str, Any]) -> List[str]:
        """Generate performance recommendations based on benchmark results."""
        recommendations = []
        overall = comparison_results.get('overall_comparison', {})
        
        if overall.get('average_speed_improvement', 0) > 2.0:
            recommendations.append("✅ Significant performance improvement achieved (>2x speed gain)")
        elif overall.get('average_speed_improvement', 0) > 1.5:
            recommendations.append("✅ Good performance improvement achieved (>1.5x speed gain)")
        elif overall.get('average_speed_improvement', 0) > 1.1:
            recommendations.append("⚠️ Modest performance improvement (>1.1x speed gain)")
        else:
            recommendations.append("❌ Limited or no performance improvement detected")
        
        if overall.get('average_memory_reduction_percent', 0) > 10:
            recommendations.append(f"✅ Good memory usage reduction ({overall['average_memory_reduction_percent']:.1f}%)")
        
        if not overall.get('consistent_improvement', False):
            recommendations.append("⚠️ Performance improvements are not consistent across all test cases")
            recommendations.append("🔍 Consider further optimization for edge cases")
        
        # Check for specific patterns in results
        large_data_performance = self._analyze_large_data_performance(comparison_results)
        if large_data_performance:
            recommendations.extend(large_data_performance)
        
        return recommendations
    
    def _analyze_large_data_performance(self, comparison_results: Dict[str, Any]) -> List[str]:
        """Analyze performance patterns for large datasets."""
        recommendations = []
        results_by_size = comparison_results.get('results_by_size', {})
        
        # Check if performance scales well with data size
        large_sizes = [size for size in self.test_data_sizes if size >= 100000]
        
        if large_sizes:
            large_improvements = []
            for size in large_sizes:
                if size in results_by_size:
                    for complexity_data in results_by_size[size].values():
                        improvement = complexity_data.get('improvement', {})
                        if 'speed_improvement_factor' in improvement:
                            large_improvements.append(improvement['speed_improvement_factor'])
            
            if large_improvements:
                avg_large_improvement = np.mean(large_improvements)
                if avg_large_improvement > 2.0:
                    recommendations.append("🚀 Excellent scalability for large datasets (>2x improvement)")
                elif avg_large_improvement < 1.2:
                    recommendations.append("⚠️ Limited scalability benefits for large datasets")
                    recommendations.append("💡 Consider implementing more aggressive chunking or parallel processing")
        
        return recommendations
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if PSUTIL_AVAILABLE:
            try:
                process = psutil.Process()
                return process.memory_info().rss / (1024 * 1024)
            except:
                pass
        return 0.0
    
    def save_benchmark_results(self, results: Dict[str, Any], filename: str = None) -> str:
        """Save benchmark results to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_benchmark_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        return filename
    
    def generate_benchmark_report(self, results: Dict[str, Any]) -> str:
        """Generate a formatted benchmark report."""
        report = []
        report.append("=" * 80)
        report.append("PERFORMANCE BENCHMARK REPORT")
        report.append("=" * 80)
        
        # System information
        system_info = results.get('system_info', {})
        report.append(f"\nSystem Information:")
        report.append(f"  Platform: {system_info.get('platform', 'Unknown')}")
        report.append(f"  CPU Count: {system_info.get('cpu_count', 'Unknown')}")
        report.append(f"  Total Memory: {system_info.get('total_memory_gb', 'Unknown')} GB")
        
        # Test configuration
        config = results.get('test_configuration', {})
        report.append(f"\nTest Configuration:")
        report.append(f"  Data sizes tested: {config.get('data_sizes', [])}")
        report.append(f"  Complexity levels: {config.get('complexity_levels', [])}")
        report.append(f"  Benchmark runs: {config.get('benchmark_runs', 'Unknown')}")
        
        # Overall comparison
        overall = results.get('overall_comparison', {})
        if 'average_speed_improvement' in overall:
            report.append(f"\nOverall Performance Comparison:")
            report.append(f"  Average Speed Improvement: {overall['average_speed_improvement']:.2f}x")
            report.append(f"  Median Speed Improvement: {overall['median_speed_improvement']:.2f}x")
            report.append(f"  Best Speed Improvement: {overall['best_speed_improvement']:.2f}x")
            report.append(f"  Average Memory Reduction: {overall['average_memory_reduction_percent']:.1f}%")
            report.append(f"  Consistent Improvement: {'Yes' if overall['consistent_improvement'] else 'No'}")
        
        # Recommendations
        recommendations = results.get('recommendations', [])
        if recommendations:
            report.append(f"\nRecommendations:")
            for rec in recommendations:
                report.append(f"  {rec}")
        
        # Detailed results summary
        report.append(f"\nDetailed Results by Data Size:")
        results_by_size = results.get('results_by_size', {})
        for size, size_results in results_by_size.items():
            report.append(f"\n  {size:,} rows:")
            for complexity, complexity_results in size_results.items():
                improvement = complexity_results.get('improvement', {})
                if 'speed_improvement_factor' in improvement:
                    report.append(f"    {complexity}: {improvement['speed_improvement_factor']:.2f}x faster, "
                                f"{improvement['throughput_improvement_percent']:.1f}% better throughput")
        
        return "\n".join(report)


def run_comprehensive_performance_benchmark(
    baseline_function: Optional[Callable] = None,
    optimized_function: Optional[Callable] = None,
    test_sizes: List[int] = [1000, 10000, 50000, 100000],
    save_results: bool = True
) -> Dict[str, Any]:
    """
    Run a comprehensive performance benchmark comparing baseline vs optimized implementations.
    
    Args:
        baseline_function: Baseline function to test
        optimized_function: Optimized function to test
        test_sizes: List of data sizes to test
        save_results: Whether to save results to file
        
    Returns:
        Comprehensive benchmark results
    """
    # Import functions here to avoid circular imports
    from .core import generate_unique_row_ids, create_optimized_row_id_function
    
    # Use default functions if not provided
    if baseline_function is None:
        baseline_function = generate_unique_row_ids
    
    if optimized_function is None:
        optimized_function = create_optimized_row_id_function()
    
    # Initialize benchmark suite
    benchmark_suite = PerformanceBenchmarkSuite(
        test_data_sizes=test_sizes,
        warmup_runs=2,
        benchmark_runs=5,
        enable_memory_profiling=True
    )
    
    # Run comparative benchmark
    results = benchmark_suite.run_comparative_benchmark(
        baseline_func=baseline_function,
        optimized_func=optimized_function,
        complexity_levels=['simple', 'medium', 'complex'],
        columns=None,  # Use all columns
        show_progress=False,  # Disable progress for benchmarking
        enable_monitoring=False  # Disable internal monitoring to avoid interference
    )
    
    # Generate and print report
    report = benchmark_suite.generate_benchmark_report(results)
    print(f"\n{report}")
    
    # Save results if requested
    if save_results:
        filename = benchmark_suite.save_benchmark_results(results)
        print(f"\n📊 Benchmark results saved to: {filename}")
    
    return results


def create_simple_benchmark() -> Dict[str, Any]:
    """Create a simple benchmark for quick testing."""
    from .core import generate_unique_row_ids, create_optimized_row_id_function
    
    # Create simple test data
    test_data = pd.DataFrame({
        'id': range(10000),
        'category': np.random.choice(['A', 'B', 'C'], 10000),
        'value': np.random.randint(1, 1000, 10000)
    })
    
    # Create functions
    baseline_func = generate_unique_row_ids
    optimized_func = create_optimized_row_id_function()
    
    # Create benchmark suite
    benchmark_suite = PerformanceBenchmarkSuite(
        test_data_sizes=[10000],
        benchmark_runs=3
    )
    
    # Run quick benchmark
    print("Running simple benchmark...")
    
    baseline_result = benchmark_suite.benchmark_function(
        baseline_func, test_data, 'baseline', show_progress=False
    )
    
    optimized_result = benchmark_suite.benchmark_function(
        optimized_func, test_data, 'optimized', show_progress=False
    )
    
    # Calculate improvement
    improvement = benchmark_suite._calculate_improvement_metrics(baseline_result, optimized_result)
    
    print(f"\nSimple Benchmark Results:")
    print(f"Baseline time: {baseline_result['execution_time']['mean']:.3f}s")
    print(f"Optimized time: {optimized_result['execution_time']['mean']:.3f}s")
    print(f"Speed improvement: {improvement['speed_improvement_factor']:.2f}x")
    print(f"Throughput improvement: {improvement['throughput_improvement_percent']:.1f}%")
    
    return {
        'baseline': baseline_result,
        'optimized': optimized_result,
        'improvement': improvement
    }


if __name__ == "__main__":
    # Run simple benchmark when script is executed directly
    create_simple_benchmark() 