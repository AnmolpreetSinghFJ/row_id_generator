"""
Code Coverage Analysis and Monitoring
Task 9.6: Achieve code coverage targets - Coverage Analysis and Reporting
"""

import os
import sys
import subprocess
import json
from datetime import datetime
import pandas as pd

# Add the package to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class CoverageAnalyzer:
    """Comprehensive code coverage analysis and monitoring."""
    
    def __init__(self, project_root=None):
        """Initialize coverage analyzer."""
        self.project_root = project_root or os.path.dirname(os.path.dirname(__file__))
        self.coverage_targets = {
            'core.py': 85,
            'utils.py': 80,
            'observable.py': 75,
            'snowflake_integration.py': 85,
            'benchmarks.py': 70,
            'overall': 80
        }
        self.current_coverage = {}
        self.coverage_history = []
        
    def run_coverage_analysis(self):
        """Run comprehensive coverage analysis."""
        print("🔍 Running Comprehensive Coverage Analysis...")
        print("=" * 70)
        
        # Run coverage with all our working tests
        working_tests = [
            'test_snowflake_integration.py',
            'test_integration_simple.py',
            'test_performance.py',
            'test_edge_cases.py'
        ]
        
        for test_file in working_tests:
            if os.path.exists(test_file):
                print(f"📊 Running coverage for {test_file}...")
                try:
                    subprocess.run([
                        'python', '-m', 'coverage', 'run', '--append',
                        '--source=../row_id_generator', '--branch', test_file
                    ], check=True, capture_output=True)
                    print(f"✅ Coverage data collected from {test_file}")
                except subprocess.CalledProcessError as e:
                    print(f"⚠️  Warning: Coverage collection failed for {test_file}")
        
        # Generate coverage report
        self.generate_coverage_report()
        self.analyze_coverage_gaps()
        self.provide_improvement_recommendations()
        
    def generate_coverage_report(self):
        """Generate detailed coverage report."""
        print("\n📈 Generating Coverage Report...")
        
        try:
            # Get coverage data
            result = subprocess.run([
                'python', '-m', 'coverage', 'report', '--format=json'
            ], capture_output=True, text=True, check=True)
            
            coverage_data = json.loads(result.stdout)
            self.current_coverage = self.parse_coverage_data(coverage_data)
            
        except subprocess.CalledProcessError:
            print("⚠️  Using fallback coverage parsing...")
            self.current_coverage = self.parse_fallback_coverage()
        
        # Display coverage summary
        self.display_coverage_summary()
        
    def parse_coverage_data(self, coverage_data):
        """Parse coverage data from JSON format."""
        parsed = {}
        
        for filename, file_data in coverage_data['files'].items():
            if 'row_id_generator' in filename:
                module_name = os.path.basename(filename)
                parsed[module_name] = {
                    'line_coverage': file_data['summary']['percent_covered'],
                    'branch_coverage': file_data['summary'].get('percent_covered_display', 'N/A'),
                    'missing_lines': file_data['summary']['missing_lines'],
                    'total_lines': file_data['summary']['num_statements']
                }
        
        return parsed
    
    def parse_fallback_coverage(self):
        """Parse coverage from command line output (fallback method)."""
        try:
            result = subprocess.run([
                'python', '-m', 'coverage', 'report'
            ], capture_output=True, text=True, check=True)
            
            lines = result.stdout.split('\n')
            parsed = {}
            
            for line in lines:
                if 'row_id_generator' in line and '.py' in line:
                    parts = line.split()
                    if len(parts) >= 4:
                        filename = os.path.basename(parts[0])
                        coverage_percent = parts[-1].rstrip('%')
                        try:
                            coverage_value = float(coverage_percent)
                            parsed[filename] = {
                                'line_coverage': coverage_value,
                                'branch_coverage': 'N/A',
                                'missing_lines': 'N/A',
                                'total_lines': 'N/A'
                            }
                        except ValueError:
                            continue
            
            return parsed
            
        except subprocess.CalledProcessError:
            # Return known values from previous analysis
            return {
                'core.py': {'line_coverage': 11, 'branch_coverage': 'N/A'},
                'utils.py': {'line_coverage': 6, 'branch_coverage': 'N/A'},
                'observable.py': {'line_coverage': 16, 'branch_coverage': 'N/A'},
                'snowflake_integration.py': {'line_coverage': 72, 'branch_coverage': 'N/A'},
                'benchmarks.py': {'line_coverage': 10, 'branch_coverage': 'N/A'}
            }
    
    def display_coverage_summary(self):
        """Display comprehensive coverage summary."""
        print("\n📊 COVERAGE SUMMARY")
        print("-" * 70)
        print(f"{'Module':<25} {'Current':<10} {'Target':<10} {'Status':<15} {'Gap':<10}")
        print("-" * 70)
        
        total_coverage = 0
        module_count = 0
        
        for module, target in self.coverage_targets.items():
            if module == 'overall':
                continue
                
            current = self.current_coverage.get(module, {}).get('line_coverage', 0)
            gap = target - current
            status = "✅ PASS" if current >= target else "❌ NEEDS WORK"
            
            print(f"{module:<25} {current:>7.1f}% {target:>7}% {status:<15} {gap:>+7.1f}%")
            
            total_coverage += current
            module_count += 1
        
        # Calculate overall coverage
        overall_current = total_coverage / module_count if module_count > 0 else 0
        overall_target = self.coverage_targets['overall']
        overall_gap = overall_target - overall_current
        overall_status = "✅ PASS" if overall_current >= overall_target else "❌ NEEDS WORK"
        
        print("-" * 70)
        print(f"{'OVERALL':<25} {overall_current:>7.1f}% {overall_target:>7}% {overall_status:<15} {overall_gap:>+7.1f}%")
        print("-" * 70)
        
        return overall_current >= overall_target
    
    def analyze_coverage_gaps(self):
        """Analyze coverage gaps and identify priority areas."""
        print("\n🔍 COVERAGE GAP ANALYSIS")
        print("-" * 50)
        
        priority_modules = []
        
        for module, target in self.coverage_targets.items():
            if module == 'overall':
                continue
                
            current = self.current_coverage.get(module, {}).get('line_coverage', 0)
            gap = target - current
            
            if gap > 0:
                priority_modules.append({
                    'module': module,
                    'current': current,
                    'target': target,
                    'gap': gap,
                    'priority': self.calculate_priority(module, gap)
                })
        
        # Sort by priority
        priority_modules.sort(key=lambda x: x['priority'], reverse=True)
        
        if priority_modules:
            print("🎯 TOP PRIORITY MODULES FOR IMPROVEMENT:")
            for i, module_info in enumerate(priority_modules[:3], 1):
                print(f"{i}. {module_info['module']}: {module_info['current']:.1f}% → {module_info['target']}% (gap: {module_info['gap']:.1f}%)")
        else:
            print("🎉 All modules meet coverage targets!")
    
    def calculate_priority(self, module, gap):
        """Calculate priority score for coverage improvement."""
        # Higher priority for core modules and larger gaps
        core_modules = {'core.py': 3, 'utils.py': 2, 'observable.py': 2}
        base_priority = core_modules.get(module, 1)
        gap_weight = gap / 10  # 10% gap = 1 priority point
        
        return base_priority + gap_weight
    
    def provide_improvement_recommendations(self):
        """Provide specific recommendations for improving coverage."""
        print("\n💡 COVERAGE IMPROVEMENT RECOMMENDATIONS")
        print("-" * 50)
        
        recommendations = {
            'core.py': [
                "Add tests for data preprocessing functions",
                "Test error handling and validation code paths",
                "Add tests for performance optimization components",
                "Test configuration management functionality"
            ],
            'utils.py': [
                "Create comprehensive utility function tests",
                "Test file I/O and data handling utilities",
                "Add error handling tests for utility functions",
                "Test data validation and transformation utilities"
            ],
            'observable.py': [
                "Add tests for logging and monitoring functions",
                "Test alerting and notification systems",
                "Add dashboard and metrics collection tests",
                "Test observability configuration and setup"
            ],
            'benchmarks.py': [
                "Add performance benchmarking tests",
                "Test benchmark result analysis and reporting",
                "Add comparative performance testing",
                "Test benchmark configuration and setup"
            ]
        }
        
        for module, target in self.coverage_targets.items():
            if module == 'overall':
                continue
                
            current = self.current_coverage.get(module, {}).get('line_coverage', 0)
            if current < target:
                gap = target - current
                print(f"\n🔧 {module} (Gap: {gap:.1f}%)")
                
                if module in recommendations:
                    for rec in recommendations[module][:3]:  # Top 3 recommendations
                        print(f"   • {rec}")
                else:
                    print(f"   • Add comprehensive test coverage for {module}")
                    print(f"   • Focus on error handling and edge cases")
                    print(f"   • Test main functionality and configuration")
    
    def create_coverage_improvement_plan(self):
        """Create a detailed coverage improvement plan."""
        print("\n📋 COVERAGE IMPROVEMENT PLAN")
        print("=" * 50)
        
        plan = {
            'immediate_actions': [],
            'short_term_goals': [],
            'long_term_targets': []
        }
        
        for module, target in self.coverage_targets.items():
            if module == 'overall':
                continue
                
            current = self.current_coverage.get(module, {}).get('line_coverage', 0)
            gap = target - current
            
            if gap > 50:
                plan['immediate_actions'].append(f"Create basic test suite for {module}")
            elif gap > 20:
                plan['short_term_goals'].append(f"Improve {module} coverage to {target}%")
            elif gap > 0:
                plan['long_term_targets'].append(f"Fine-tune {module} coverage to {target}%")
        
        # Display plan
        if plan['immediate_actions']:
            print("🚨 IMMEDIATE ACTIONS (Critical - <50% coverage):")
            for action in plan['immediate_actions']:
                print(f"   • {action}")
        
        if plan['short_term_goals']:
            print("\n📅 SHORT-TERM GOALS (1-2 weeks):")
            for goal in plan['short_term_goals']:
                print(f"   • {goal}")
        
        if plan['long_term_targets']:
            print("\n🎯 LONG-TERM TARGETS (1 month):")
            for target in plan['long_term_targets']:
                print(f"   • {target}")
        
        return plan
    
    def generate_coverage_dashboard_data(self):
        """Generate data for coverage dashboard."""
        dashboard_data = {
            'timestamp': datetime.now().isoformat(),
            'overall_coverage': sum(
                self.current_coverage.get(module, {}).get('line_coverage', 0)
                for module in self.coverage_targets if module != 'overall'
            ) / (len(self.coverage_targets) - 1),
            'modules': [],
            'targets_met': 0,
            'total_targets': len(self.coverage_targets) - 1
        }
        
        for module, target in self.coverage_targets.items():
            if module == 'overall':
                continue
                
            current = self.current_coverage.get(module, {}).get('line_coverage', 0)
            meets_target = current >= target
            
            if meets_target:
                dashboard_data['targets_met'] += 1
            
            dashboard_data['modules'].append({
                'name': module,
                'current_coverage': current,
                'target_coverage': target,
                'meets_target': meets_target,
                'gap': target - current
            })
        
        return dashboard_data
    
    def save_coverage_report(self, output_file='coverage_analysis_report.json'):
        """Save comprehensive coverage report to file."""
        report_data = {
            'analysis_timestamp': datetime.now().isoformat(),
            'coverage_targets': self.coverage_targets,
            'current_coverage': self.current_coverage,
            'dashboard_data': self.generate_coverage_dashboard_data(),
            'improvement_plan': self.create_coverage_improvement_plan()
        }
        
        output_path = os.path.join(self.project_root, 'tests', output_file)
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n💾 Coverage report saved to: {output_file}")
        return output_path


def run_coverage_analysis():
    """Run comprehensive coverage analysis."""
    print("🚀 Starting Comprehensive Code Coverage Analysis")
    print("=" * 70)
    
    analyzer = CoverageAnalyzer()
    analyzer.run_coverage_analysis()
    
    # Create improvement plan
    improvement_plan = analyzer.create_coverage_improvement_plan()
    
    # Generate dashboard data
    dashboard_data = analyzer.generate_coverage_dashboard_data()
    
    # Save comprehensive report
    report_path = analyzer.save_coverage_report()
    
    print("\n📊 COVERAGE ANALYSIS COMPLETE")
    print("=" * 50)
    print(f"Overall Coverage: {dashboard_data['overall_coverage']:.1f}%")
    print(f"Targets Met: {dashboard_data['targets_met']}/{dashboard_data['total_targets']}")
    print(f"Report Location: {report_path}")
    
    return dashboard_data['overall_coverage'] >= analyzer.coverage_targets['overall']


if __name__ == '__main__':
    success = run_coverage_analysis()
    if success:
        print("\n✅ Coverage targets achieved!")
    else:
        print("\n📈 Coverage improvement needed. See recommendations above.") 