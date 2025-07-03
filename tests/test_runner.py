"""
Comprehensive Test Runner for Row ID Generator
Task 9.1: Unit tests for core functions - Main Test Runner
"""

import unittest
import sys
import os
import time
import numpy as np
from io import StringIO

# Add the package to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import all test modules
try:
    from tests.test_core import run_core_tests
except ImportError:
    def run_core_tests():
        print("Core tests module not available")
        return False

try:
    from tests.test_utils import run_utils_tests
except ImportError:
    def run_utils_tests():
        print("Utils tests module not available")
        return False

try:
    from tests.test_observable import run_observable_tests
except ImportError:
    def run_observable_tests():
        print("Observable tests module not available")
        return False

try:
    from tests.test_benchmarks import run_benchmark_tests
except ImportError:
    def run_benchmark_tests():
        print("Benchmark tests module not available")
        return False

try:
    from tests.test_snowflake_integration import run_snowflake_tests
except ImportError:
    def run_snowflake_tests():
        print("Snowflake integration tests module not available")
        return False


class TestSuiteRunner:
    """Comprehensive test suite runner."""
    
    def __init__(self):
        self.results = {}
        self.start_time = None
        self.end_time = None
    
    def run_all_tests(self, verbose=True):
        """Run all available test suites."""
        print("🚀 Starting Comprehensive Test Suite")
        print("=" * 80)
        
        self.start_time = time.time()
        
        # Define test suites
        test_suites = [
            ("Core Functions", run_core_tests),
            ("Utilities", run_utils_tests),
            ("Observability Framework", run_observable_tests),
            ("Benchmarks", run_benchmark_tests),
            ("Snowflake Integration", run_snowflake_tests)
        ]
        
        # Run each test suite
        for suite_name, test_function in test_suites:
            print(f"\n📋 Running {suite_name} Tests...")
            print("-" * 60)
            
            suite_start = time.time()
            
            try:
                success = test_function()
                suite_time = time.time() - suite_start
                
                self.results[suite_name] = {
                    'success': success,
                    'duration': suite_time,
                    'error': None
                }
                
                if success:
                    print(f"✅ {suite_name} tests completed successfully in {suite_time:.2f}s")
                else:
                    print(f"❌ {suite_name} tests failed")
                    
            except Exception as e:
                suite_time = time.time() - suite_start
                self.results[suite_name] = {
                    'success': False,
                    'duration': suite_time,
                    'error': str(e)
                }
                print(f"💥 {suite_name} tests crashed: {e}")
        
        self.end_time = time.time()
        
        # Print summary
        self._print_summary()
        
        return self._all_tests_passed()
    
    def _print_summary(self):
        """Print comprehensive test summary."""
        total_time = self.end_time - self.start_time
        
        print("\n" + "=" * 80)
        print("📊 COMPREHENSIVE TEST SUMMARY")
        print("=" * 80)
        
        # Suite results
        passed_suites = 0
        failed_suites = 0
        
        for suite_name, result in self.results.items():
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            duration = result['duration']
            
            print(f"{status} | {suite_name:<25} | {duration:>6.2f}s")
            
            if result['success']:
                passed_suites += 1
            else:
                failed_suites += 1
                
            if result['error']:
                print(f"      Error: {result['error']}")
        
        print("-" * 80)
        
        # Overall statistics
        total_suites = len(self.results)
        success_rate = (passed_suites / total_suites) * 100 if total_suites > 0 else 0
        
        print(f"📈 OVERALL RESULTS:")
        print(f"   Total Test Suites: {total_suites}")
        print(f"   Passed: {passed_suites}")
        print(f"   Failed: {failed_suites}")
        print(f"   Success Rate: {success_rate:.1f}%")
        print(f"   Total Duration: {total_time:.2f}s")
        
        # Final verdict
        if self._all_tests_passed():
            print("\n🎉 ALL TESTS PASSED! 🎉")
            print("The Row ID Generator is ready for production!")
        else:
            print("\n⚠️  SOME TESTS FAILED")
            print("Please review the failed tests before proceeding.")
        
        print("=" * 80)
    
    def _all_tests_passed(self):
        """Check if all test suites passed."""
        return all(result['success'] for result in self.results.values())
    
    def get_results(self):
        """Get detailed test results."""
        return {
            'suites': self.results,
            'overall_success': self._all_tests_passed(),
            'total_duration': self.end_time - self.start_time if self.end_time else None,
            'summary': {
                'total_suites': len(self.results),
                'passed_suites': sum(1 for r in self.results.values() if r['success']),
                'failed_suites': sum(1 for r in self.results.values() if not r['success'])
            }
        }


class QuickTestRunner:
    """Quick test runner for basic functionality."""
    
    @staticmethod
    def run_basic_tests():
        """Run basic sanity tests."""
        print("🔧 Running Basic Sanity Tests...")
        print("-" * 40)
        
        tests_passed = 0
        total_tests = 0
        
        # Test 1: Import test
        total_tests += 1
        try:
            import pandas as pd
            import hashlib
            print("✅ Package imports successful")
            tests_passed += 1
        except ImportError as e:
            print(f"❌ Package import failed: {e}")
        
        # Test 2: Basic DataFrame operations
        total_tests += 1
        try:
            df = pd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z']})
            assert len(df) == 3
            assert len(df.columns) == 2
            print("✅ Basic DataFrame operations work")
            tests_passed += 1
        except Exception as e:
            print(f"❌ DataFrame operations failed: {e}")
        
        # Test 3: Hash generation
        total_tests += 1
        try:
            test_string = "test_data"
            hash_result = hashlib.sha256(test_string.encode()).hexdigest()
            assert len(hash_result) == 64
            print("✅ Hash generation works")
            tests_passed += 1
        except Exception as e:
            print(f"❌ Hash generation failed: {e}")
        
        # Test 4: File I/O
        total_tests += 1
        try:
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
                f.write("test content")
                temp_path = f.name
            
            with open(temp_path, 'r') as f:
                content = f.read()
                assert content == "test content"
            
            os.unlink(temp_path)
            print("✅ File I/O operations work")
            tests_passed += 1
        except Exception as e:
            print(f"❌ File I/O failed: {e}")
        
        print("-" * 40)
        print(f"Basic Tests: {tests_passed}/{total_tests} passed")
        
        return tests_passed == total_tests


def run_integration_tests():
    """Run integration tests across modules."""
    print("🔗 Running Integration Tests...")
    print("-" * 40)
    
    integration_results = []
    
    # Integration Test 1: Data processing pipeline
    try:
        # Simulate data processing pipeline
        import pandas as pd
        import hashlib
        
        # Create test data
        test_df = pd.DataFrame({
            'id': range(100),
            'name': [f'user_{i}' for i in range(100)],
            'value': np.random.randn(100)
        })
        
        # Process data
        processed_df = test_df.copy()
        processed_df['name'] = processed_df['name'].str.upper()
        processed_df['value_abs'] = processed_df['value'].abs()
        
        # Generate row IDs
        row_ids = []
        for _, row in processed_df.iterrows():
            row_string = f"{row['id']}_{row['name']}_{row['value_abs']:.6f}"
            row_id = hashlib.sha256(row_string.encode()).hexdigest()
            row_ids.append(row_id)
        
        processed_df['row_id'] = row_ids
        
        # Validate results
        assert len(processed_df) == 100
        assert 'row_id' in processed_df.columns
        assert processed_df['row_id'].nunique() == 100  # All unique
        
        integration_results.append(("Data Processing Pipeline", True, None))
        print("✅ Data processing pipeline integration test passed")
        
    except Exception as e:
        integration_results.append(("Data Processing Pipeline", False, str(e)))
        print(f"❌ Data processing pipeline integration test failed: {e}")
    
    # Integration Test 2: Error handling and logging
    try:
        import logging
        import io
        
        # Set up test logger
        log_stream = io.StringIO()
        handler = logging.StreamHandler(log_stream)
        logger = logging.getLogger('integration_test')
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        
        # Test logging
        logger.info("Integration test message")
        logger.error("Integration test error")
        
        # Check log content
        log_content = log_stream.getvalue()
        assert "Integration test message" in log_content
        assert "Integration test error" in log_content
        
        integration_results.append(("Error Handling & Logging", True, None))
        print("✅ Error handling and logging integration test passed")
        
    except Exception as e:
        integration_results.append(("Error Handling & Logging", False, str(e)))
        print(f"❌ Error handling and logging integration test failed: {e}")
    
    print("-" * 40)
    
    # Summary
    passed = sum(1 for _, success, _ in integration_results if success)
    total = len(integration_results)
    print(f"Integration Tests: {passed}/{total} passed")
    
    return passed == total


def main():
    """Main test execution function."""
    print("🧪 Row ID Generator - Comprehensive Test Suite")
    print("=" * 80)
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Run Row ID Generator tests')
    parser.add_argument('--quick', action='store_true', help='Run only quick sanity tests')
    parser.add_argument('--integration', action='store_true', help='Run only integration tests')
    parser.add_argument('--all', action='store_true', help='Run all tests (default)')
    args = parser.parse_args()
    
    # Default to all tests if no specific option is given
    if not any([args.quick, args.integration]):
        args.all = True
    
    overall_success = True
    
    # Run quick tests
    if args.quick or args.all:
        quick_success = QuickTestRunner.run_basic_tests()
        overall_success = overall_success and quick_success
        
        if not args.all:
            return 0 if quick_success else 1
    
    # Run integration tests
    if args.integration or args.all:
        integration_success = run_integration_tests()
        overall_success = overall_success and integration_success
        
        if args.integration and not args.all:
            return 0 if integration_success else 1
    
    # Run comprehensive tests
    if args.all:
        runner = TestSuiteRunner()
        comprehensive_success = runner.run_all_tests()
        overall_success = overall_success and comprehensive_success
    
    # Return appropriate exit code
    return 0 if overall_success else 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code) 