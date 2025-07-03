#!/usr/bin/env python3
"""
Comprehensive Test Automation Script
Task 9.7: Automated test execution pipeline

Provides multiple test execution modes with parallel processing, coverage reporting,
and comprehensive test management for local development and CI/CD environments.
"""

import argparse
import subprocess
import sys
import os
import time
import json
import concurrent.futures
from pathlib import Path
from datetime import datetime


class TestAutomation:
    """Automated test execution and management system."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.test_dir = self.project_root / "tests"
        self.results_dir = self.project_root / "test_results"
        self.coverage_dir = self.project_root / "coverage_reports"
        
        # Ensure directories exist
        self.results_dir.mkdir(exist_ok=True)
        self.coverage_dir.mkdir(exist_ok=True)
        
        self.test_suites = {
            'unit': [
                'test_core.py',
                'test_utils.py', 
                'test_observable.py',
                'test_benchmarks.py'
            ],
            'integration': [
                'test_integration_simple.py',
                'test_integration_workflows.py'
            ],
            'performance': [
                'test_performance.py'
            ],
            'edge': [
                'test_edge_cases.py'
            ],
            'snowflake': [
                'test_snowflake_integration.py'
            ]
        }
    
    def run_test_suite(self, suite_name, parallel=True, coverage=True, verbose=True):
        """Run a specific test suite with optional parallel execution and coverage."""
        if suite_name not in self.test_suites:
            print(f"❌ Unknown test suite: {suite_name}")
            return False
        
        test_files = self.test_suites[suite_name]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print(f"🚀 Running {suite_name} test suite...")
        print(f"📁 Test files: {', '.join(test_files)}")
        
        if parallel and len(test_files) > 1:
            return self._run_parallel_tests(suite_name, test_files, timestamp, coverage, verbose)
        else:
            return self._run_sequential_tests(suite_name, test_files, timestamp, coverage, verbose)
    
    def _run_parallel_tests(self, suite_name, test_files, timestamp, coverage, verbose):
        """Execute tests in parallel using multiple workers."""
        print(f"⚡ Running {len(test_files)} test files in parallel...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(test_files))) as executor:
            futures = []
            
            for test_file in test_files:
                future = executor.submit(
                    self._run_single_test_file, 
                    test_file, 
                    suite_name, 
                    timestamp, 
                    coverage, 
                    verbose
                )
                futures.append((test_file, future))
            
            results = {}
            for test_file, future in futures:
                try:
                    result = future.result(timeout=300)  # 5 minute timeout
                    results[test_file] = result
                    status = "✅ PASSED" if result['success'] else "❌ FAILED"
                    print(f"{status} {test_file} ({result['duration']:.2f}s)")
                except concurrent.futures.TimeoutError:
                    results[test_file] = {'success': False, 'error': 'Timeout', 'duration': 300}
                    print(f"⏱️ TIMEOUT {test_file}")
                except Exception as e:
                    results[test_file] = {'success': False, 'error': str(e), 'duration': 0}
                    print(f"💥 ERROR {test_file}: {e}")
        
        # Generate summary
        successful = sum(1 for r in results.values() if r['success'])
        total = len(results)
        total_time = sum(r['duration'] for r in results.values())
        
        print(f"\n📊 Parallel Execution Summary:")
        print(f"   Successful: {successful}/{total}")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Success rate: {successful/total*100:.1f}%")
        
        return successful == total
    
    def _run_sequential_tests(self, suite_name, test_files, timestamp, coverage, verbose):
        """Execute tests sequentially."""
        print(f"📝 Running {len(test_files)} test files sequentially...")
        
        results = []
        total_time = 0
        
        for test_file in test_files:
            result = self._run_single_test_file(test_file, suite_name, timestamp, coverage, verbose)
            results.append(result)
            total_time += result['duration']
            
            status = "✅ PASSED" if result['success'] else "❌ FAILED"
            print(f"{status} {test_file} ({result['duration']:.2f}s)")
        
        successful = sum(1 for r in results if r['success'])
        print(f"\n📊 Sequential Execution Summary:")
        print(f"   Successful: {successful}/{len(results)}")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Success rate: {successful/len(results)*100:.1f}%")
        
        return successful == len(results)
    
    def _run_single_test_file(self, test_file, suite_name, timestamp, coverage, verbose):
        """Run a single test file and return results."""
        start_time = time.time()
        test_path = self.test_dir / test_file
        
        if not test_path.exists():
            return {
                'success': False,
                'error': f'Test file not found: {test_file}',
                'duration': 0
            }
        
        # Build pytest command
        cmd = [
            sys.executable, '-m', 'pytest', str(test_path),
            '--tb=short',
            f'--junit-xml={self.results_dir}/{suite_name}_{test_file}_{timestamp}.xml'
        ]
        
        if coverage:
            cmd.extend([
                '--cov=row_id_generator',
                f'--cov-report=xml:{self.coverage_dir}/{suite_name}_{test_file}_{timestamp}.xml',
                '--cov-report=term-missing'
            ])
        
        if verbose:
            cmd.append('-v')
        else:
            cmd.append('-q')
        
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=180,  # 3 minute timeout per file
                cwd=self.project_root
            )
            
            duration = time.time() - start_time
            success = result.returncode == 0
            
            return {
                'success': success,
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'duration': duration
            }
            
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': 'Test execution timeout',
                'duration': time.time() - start_time
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'duration': time.time() - start_time
            }
    
    def run_all_tests(self, parallel=True, coverage=True, fast_fail=False):
        """Run all test suites with comprehensive reporting."""
        print("🎯 Running ALL test suites...")
        print("=" * 60)
        
        all_results = {}
        start_time = time.time()
        
        for suite_name in self.test_suites.keys():
            print(f"\n🔄 Starting {suite_name.upper()} tests...")
            result = self.run_test_suite(suite_name, parallel, coverage, verbose=True)
            all_results[suite_name] = result
            
            if not result and fast_fail:
                print(f"💥 Fast fail enabled - stopping due to {suite_name} failure")
                break
        
        total_time = time.time() - start_time
        successful_suites = sum(1 for r in all_results.values() if r)
        total_suites = len(all_results)
        
        print("\n" + "=" * 60)
        print("🏆 FINAL TEST EXECUTION SUMMARY")
        print("=" * 60)
        
        for suite_name, success in all_results.items():
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"   {status} {suite_name.upper()} suite")
        
        print(f"\n📈 Overall Statistics:")
        print(f"   Successful suites: {successful_suites}/{total_suites}")
        print(f"   Total execution time: {total_time:.2f}s")
        print(f"   Success rate: {successful_suites/total_suites*100:.1f}%")
        
        if successful_suites == total_suites:
            print("\n🎉 ALL TESTS PASSED! ✨")
            return True
        else:
            print(f"\n⚠️  {total_suites - successful_suites} test suite(s) failed")
            return False
    
    def run_coverage_analysis(self):
        """Run comprehensive coverage analysis across all modules."""
        print("📊 Running comprehensive coverage analysis...")
        
        cmd = [
            sys.executable, '-m', 'pytest',
            'tests/',
            '--cov=row_id_generator',
            '--cov-report=html:coverage_reports/html_report',
            '--cov-report=xml:coverage_reports/coverage.xml',
            '--cov-report=term-missing',
            '--cov-report=json:coverage_reports/coverage.json'
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
            print("Coverage analysis completed!")
            print(f"📁 HTML report: coverage_reports/html_report/index.html")
            print(f"📄 XML report: coverage_reports/coverage.xml")
            print(f"📋 JSON report: coverage_reports/coverage.json")
            
            return result.returncode == 0
            
        except Exception as e:
            print(f"❌ Coverage analysis failed: {e}")
            return False
    
    def run_performance_benchmarks(self):
        """Run dedicated performance benchmarks with detailed reporting."""
        print("⚡ Running performance benchmarks...")
        
        cmd = [
            sys.executable, '-m', 'pytest',
            'tests/test_performance.py::TestBenchmarkBaselines',
            '-v',
            '--tb=short',
            f'--junit-xml={self.results_dir}/performance_benchmarks_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xml'
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
            print("Performance benchmarks completed!")
            
            if result.returncode == 0:
                print("✅ All performance benchmarks passed")
            else:
                print("❌ Some performance benchmarks failed")
                print(result.stdout)
            
            return result.returncode == 0
            
        except Exception as e:
            print(f"❌ Performance benchmarks failed: {e}")
            return False
    
    def cleanup_old_results(self, days=7):
        """Clean up old test results and coverage reports."""
        print(f"🧹 Cleaning up test results older than {days} days...")
        
        cutoff_time = time.time() - (days * 24 * 60 * 60)
        cleaned_count = 0
        
        for directory in [self.results_dir, self.coverage_dir]:
            for file_path in directory.rglob('*'):
                if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                    try:
                        file_path.unlink()
                        cleaned_count += 1
                    except Exception as e:
                        print(f"⚠️  Could not delete {file_path}: {e}")
        
        print(f"🗑️  Cleaned up {cleaned_count} old files")


def main():
    """Main entry point for test automation."""
    parser = argparse.ArgumentParser(description='Comprehensive Test Automation System')
    
    parser.add_argument('--suite', choices=list(TestAutomation().test_suites.keys()) + ['all'],
                       default='all', help='Test suite to run')
    parser.add_argument('--no-parallel', action='store_true', help='Disable parallel execution')
    parser.add_argument('--no-coverage', action='store_true', help='Disable coverage reporting')
    parser.add_argument('--fast-fail', action='store_true', help='Stop on first suite failure')
    parser.add_argument('--coverage-only', action='store_true', help='Run only coverage analysis')
    parser.add_argument('--benchmarks-only', action='store_true', help='Run only performance benchmarks')
    parser.add_argument('--cleanup', type=int, metavar='DAYS', help='Clean up old results (specify days)')
    parser.add_argument('--quiet', action='store_true', help='Minimal output')
    
    args = parser.parse_args()
    
    automation = TestAutomation()
    
    if args.cleanup:
        automation.cleanup_old_results(args.cleanup)
        return
    
    if args.coverage_only:
        success = automation.run_coverage_analysis()
        sys.exit(0 if success else 1)
    
    if args.benchmarks_only:
        success = automation.run_performance_benchmarks()
        sys.exit(0 if success else 1)
    
    # Main test execution
    parallel = not args.no_parallel
    coverage = not args.no_coverage
    
    if args.suite == 'all':
        success = automation.run_all_tests(parallel, coverage, args.fast_fail)
    else:
        success = automation.run_test_suite(args.suite, parallel, coverage, not args.quiet)
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main() 