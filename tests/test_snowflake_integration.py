"""
Test Suite for Snowflake Integration Module
Task 8: Integrate Snowflake Connector Compatibility
"""

import unittest
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add the package to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from row_id_generator.snowflake_integration import (
    SnowflakeDataFrameCompatibilityChecker,
    SnowflakeConnectionManager,
    SnowflakeConnectionError,
    SnowflakeDataLoadError,
    prepare_for_snowflake,
    load_to_snowflake,
    validate_snowflake_connection_params,
    get_snowflake_integration_status,
    SNOWFLAKE_AVAILABLE
)


class TestSnowflakeDataFrameCompatibilityChecker(unittest.TestCase):
    """Test the DataFrame compatibility checker."""
    
    def setUp(self):
        self.checker = SnowflakeDataFrameCompatibilityChecker()
        
        # Create test DataFrame
        self.test_df = pd.DataFrame({
            'user_id': [1, 2, 3, 4, 5],
            'user_name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
            'email': ['alice@test.com', 'bob@test.com', 'charlie@test.com', 'diana@test.com', 'eve@test.com'],
            'age': [25, 30, 35, 28, 32],
            'is_active': [True, True, False, True, True],
            'signup_date': pd.to_datetime(['2023-01-01', '2023-02-01', '2023-03-01', '2023-04-01', '2023-05-01']),
            'score': [95.5, 87.2, 78.9, 92.1, 85.6]
        })
    
    def test_basic_compatibility_check(self):
        """Test basic compatibility checking."""
        result = self.checker.check_dataframe_compatibility(self.test_df, 'users')
        
        self.assertIsInstance(result, dict)
        self.assertIn('is_compatible', result)
        self.assertIn('issues', result)
        self.assertIn('warnings', result)
        self.assertIn('data_type_mappings', result)
        self.assertEqual(result['row_count'], 5)
        self.assertEqual(result['column_count'], 7)
    
    def test_table_name_validation(self):
        """Test table name validation."""
        # Valid table name
        result = self.checker.check_dataframe_compatibility(self.test_df, 'valid_table_name')
        table_issues = [issue for issue in result['issues'] if 'table name' in issue.lower()]
        self.assertEqual(len(table_issues), 0)
        
        # Invalid table name (starts with number)
        result = self.checker.check_dataframe_compatibility(self.test_df, '123invalid')
        table_issues = [issue for issue in result['issues'] if 'table name' in issue.lower()]
        self.assertGreater(len(table_issues), 0)
        
        # Reserved word
        result = self.checker.check_dataframe_compatibility(self.test_df, 'user')
        reserved_issues = [issue for issue in result['issues'] if 'reserved word' in issue.lower()]
        self.assertGreater(len(reserved_issues), 0)
    
    def test_column_name_validation(self):
        """Test column name validation."""
        # DataFrame with problematic column names
        problematic_df = pd.DataFrame({
            'user id': [1, 2, 3],  # Space in name
            'user-name': ['A', 'B', 'C'],  # Hyphen in name
            '123column': [1, 2, 3],  # Starts with number
            'a' * 300: [1, 2, 3]  # Too long
        })
        
        result = self.checker.check_dataframe_compatibility(problematic_df, 'test_table')
        self.assertFalse(result['is_compatible'])
        self.assertGreater(len(result['issues']), 0)
    
    def test_data_type_mapping(self):
        """Test data type mapping functionality."""
        result = self.checker.check_dataframe_compatibility(self.test_df, 'users')
        mappings = result['data_type_mappings']
        
        # Check specific mappings
        self.assertEqual(mappings['user_id']['snowflake_type'], 'NUMBER')
        self.assertEqual(mappings['user_name']['snowflake_type'], 'VARCHAR')
        self.assertEqual(mappings['is_active']['snowflake_type'], 'BOOLEAN')
        self.assertEqual(mappings['signup_date']['snowflake_type'], 'TIMESTAMP')
        self.assertEqual(mappings['score']['snowflake_type'], 'FLOAT')
        
        # Check metadata
        self.assertIn('nullable', mappings['user_id'])
        self.assertIn('unique_values', mappings['user_id'])
        self.assertIn('sample_values', mappings['user_id'])
    
    def test_data_quality_checks(self):
        """Test data quality checking."""
        # DataFrame with quality issues
        quality_df = pd.DataFrame({
            'mostly_null': [1, None, None, None, None],  # High null percentage
            'single_value': [1, 1, 1, 1, 1],  # Only one unique value
            'normal_column': [1, 2, 3, 4, 5]
        })
        
        result = self.checker.check_dataframe_compatibility(quality_df, 'quality_test')
        self.assertGreater(len(result['warnings']), 0)
        
        # Check for null warning
        null_warnings = [w for w in result['warnings'] if 'null values' in w]
        self.assertGreater(len(null_warnings), 0)
        
        # Check for single value warning
        single_value_warnings = [w for w in result['warnings'] if 'one unique value' in w]
        self.assertGreater(len(single_value_warnings), 0)
    
    def test_compatibility_statistics(self):
        """Test compatibility statistics tracking."""
        initial_stats = self.checker.get_compatibility_statistics()
        
        # Run some checks
        self.checker.check_dataframe_compatibility(self.test_df, 'users')
        
        # Create problematic DataFrame
        bad_df = pd.DataFrame({'user id': [1, 2, 3]})  # Space in column name
        self.checker.check_dataframe_compatibility(bad_df, 'bad_table')
        
        final_stats = self.checker.get_compatibility_statistics()
        
        # Check that statistics were updated
        self.assertGreater(final_stats['total_checks'], initial_stats['total_checks'])
        self.assertGreaterEqual(final_stats['compatibility_issues'], initial_stats['compatibility_issues'])


class TestSnowflakeConnectionManager(unittest.TestCase):
    """Test the Snowflake connection manager."""
    
    def setUp(self):
        self.connection_params = {
            'user': 'test_user',
            'password': 'test_password',
            'account': 'test_account',
            'warehouse': 'TEST_WH',
            'database': 'TEST_DB',
            'schema': 'TEST_SCHEMA'
        }
        self.manager = SnowflakeConnectionManager(self.connection_params)
    
    def test_connection_creation_mock(self):
        """Test connection creation in mock mode."""
        # Since SNOWFLAKE_AVAILABLE is likely False in test environment,
        # this should create a mock connection
        result = self.manager.create_connection('test_conn')
        
        self.assertIsInstance(result, dict)
        self.assertIn('connection_id', result)
        self.assertIn('success', result)
        self.assertIn('connection_time', result)
        
        # In mock mode, should succeed
        if not SNOWFLAKE_AVAILABLE:
            self.assertTrue(result['success'])
            self.assertEqual(result['connection'], 'mock_connection')
    
    def test_health_check(self):
        """Test health check functionality."""
        health_result = self.manager.perform_health_check()
        
        self.assertIsInstance(health_result, dict)
        self.assertIn('timestamp', health_result)
        self.assertIn('is_healthy', health_result)
        self.assertIn('connection_test', health_result)
        self.assertIn('query_test', health_result)
        self.assertIn('response_time', health_result)
        
        # Response time should be measured
        self.assertGreater(health_result['response_time'], 0)
    
    def test_connection_statistics(self):
        """Test connection statistics tracking."""
        initial_stats = self.manager.get_connection_statistics()
        
        # Create a connection
        self.manager.create_connection('test_stats')
        
        updated_stats = self.manager.get_connection_statistics()
        
        # Check that statistics were updated
        self.assertGreater(updated_stats['total_connections'], initial_stats['total_connections'])
        self.assertIn('health_status', updated_stats)
    
    def test_parameter_sanitization(self):
        """Test connection parameter sanitization."""
        sanitized = self.manager._sanitize_connection_params()
        
        self.assertIn('user', sanitized)
        self.assertIn('account', sanitized)
        self.assertEqual(sanitized['password'], '***')


class TestSnowflakeUtilityFunctions(unittest.TestCase):
    """Test utility functions."""
    
    def test_prepare_for_snowflake(self):
        """Test DataFrame preparation function."""
        test_df = pd.DataFrame({
            'user id': [1, 2, 3],  # Space in name - should be fixed
            'valid_column': ['A', 'B', 'C'],
            'row_id': ['hash1', 'hash2', 'hash3']
        })
        
        # Test with auto-fix enabled
        prepared_df = prepare_for_snowflake(test_df, 'test_table', auto_fix_issues=True)
        
        self.assertIsInstance(prepared_df, pd.DataFrame)
        self.assertEqual(len(prepared_df), 3)
        self.assertNotIn('user id', prepared_df.columns)  # Should be fixed
        self.assertIn('row_id', prepared_df.columns)
    
    def test_load_to_snowflake_mock(self):
        """Test Snowflake loading function in mock mode."""
        test_df = pd.DataFrame({
            'user_id': [1, 2, 3],
            'name': ['A', 'B', 'C'],
            'row_id': ['hash1', 'hash2', 'hash3']
        })
        
        connection_params = {
            'user': 'test',
            'password': 'test',
            'account': 'test'
        }
        
        # This should work in mock mode
        success, rows_loaded = load_to_snowflake(test_df, connection_params, 'test_table')
        
        if not SNOWFLAKE_AVAILABLE:
            self.assertTrue(success)
            self.assertEqual(rows_loaded, 3)
    
    def test_validate_connection_params(self):
        """Test connection parameter validation."""
        # Valid parameters
        valid_params = {
            'user': 'test_user',
            'password': 'test_password',
            'account': 'test_account'
        }
        
        result = validate_snowflake_connection_params(valid_params)
        self.assertTrue(result['is_valid'])
        self.assertEqual(len(result['missing_params']), 0)
        
        # Missing required parameters
        invalid_params = {
            'user': 'test_user'
            # Missing password and account
        }
        
        result = validate_snowflake_connection_params(invalid_params)
        self.assertFalse(result['is_valid'])
        self.assertGreater(len(result['missing_params']), 0)
        self.assertIn('password', result['missing_params'])
        self.assertIn('account', result['missing_params'])
        
        # Invalid account format
        invalid_format_params = {
            'user': 'test',
            'password': 'test',
            'account': 'invalid@account!'
        }
        
        result = validate_snowflake_connection_params(invalid_format_params)
        self.assertFalse(result['is_valid'])
        self.assertIn('account', result['invalid_params'])
    
    def test_integration_status(self):
        """Test integration status function."""
        status = get_snowflake_integration_status()
        
        self.assertIsInstance(status, dict)
        self.assertIn('snowflake_available', status)
        self.assertIn('version', status)
        self.assertIn('features', status)
        self.assertIn('mock_mode', status)
        
        # Check features
        features = status['features']
        self.assertTrue(features['dataframe_compatibility_checking'])
        self.assertTrue(features['connection_management'])
        self.assertTrue(features['health_monitoring'])
        self.assertTrue(features['data_loading'])
        self.assertTrue(features['error_handling'])
        self.assertTrue(features['logging'])


class TestSnowflakeExceptions(unittest.TestCase):
    """Test custom exception classes."""
    
    def test_connection_error(self):
        """Test SnowflakeConnectionError."""
        context = {'connection_param': 'test_value'}
        error = SnowflakeConnectionError(
            "Test connection error",
            error_code="SF001",
            context=context
        )
        
        self.assertEqual(str(error), "Test connection error")
        self.assertEqual(error.error_code, "SF001")
        self.assertEqual(error.context, context)
        self.assertIsInstance(error.timestamp, datetime)
        
        # Test dictionary conversion
        error_dict = error.to_dict()
        self.assertEqual(error_dict['error_type'], 'SnowflakeConnectionError')
        self.assertEqual(error_dict['message'], "Test connection error")
        self.assertEqual(error_dict['error_code'], "SF001")
        self.assertEqual(error_dict['context'], context)
    
    def test_data_load_error(self):
        """Test SnowflakeDataLoadError."""
        context = {'table_name': 'test_table', 'rows': 100}
        error = SnowflakeDataLoadError(
            "Test data load error",
            error_code="SF002",
            context=context
        )
        
        self.assertEqual(str(error), "Test data load error")
        self.assertEqual(error.error_code, "SF002")
        self.assertEqual(error.context, context)
        
        # Test dictionary conversion
        error_dict = error.to_dict()
        self.assertEqual(error_dict['error_type'], 'SnowflakeDataLoadError')
        self.assertEqual(error_dict['context'], context)


class TestSnowflakeIntegrationEndToEnd(unittest.TestCase):
    """End-to-end integration tests."""
    
    def test_complete_workflow(self):
        """Test complete workflow from DataFrame preparation to loading."""
        # Create test data
        test_df = pd.DataFrame({
            'user_id': [1, 2, 3, 4, 5],
            'user name': ['Alice Smith', 'Bob Jones', 'Charlie Brown', 'Diana Prince', 'Eve Adams'],
            'email': ['alice@test.com', 'bob@test.com', 'charlie@test.com', 'diana@test.com', 'eve@test.com'],
            'age': [25, 30, 35, 28, 32],
            'is_active': [True, True, False, True, True],
            'row_id': ['hash1', 'hash2', 'hash3', 'hash4', 'hash5']
        })
        
        # Step 1: Check compatibility
        checker = SnowflakeDataFrameCompatibilityChecker()
        compatibility_result = checker.check_dataframe_compatibility(test_df, 'users')
        
        # Should have issues due to space in column name
        self.assertFalse(compatibility_result['is_compatible'])
        
        # Step 2: Prepare DataFrame
        prepared_df = prepare_for_snowflake(test_df, 'users', auto_fix_issues=True)
        
        # Should be fixed now
        self.assertNotIn('user name', prepared_df.columns)
        self.assertIn('row_id', prepared_df.columns)
        
        # Step 3: Validate connection parameters
        connection_params = {
            'user': 'test_user',
            'password': 'test_password',
            'account': 'test_account',
            'warehouse': 'TEST_WH'
        }
        
        validation_result = validate_snowflake_connection_params(connection_params)
        self.assertTrue(validation_result['is_valid'])
        
        # Step 4: Load to Snowflake (mock mode)
        if not SNOWFLAKE_AVAILABLE:
            success, rows_loaded = load_to_snowflake(prepared_df, connection_params, 'users')
            self.assertTrue(success)
            self.assertEqual(rows_loaded, 5)


def run_snowflake_tests():
    """Run all Snowflake integration tests."""
    print("Running Snowflake Integration Tests...")
    print(f"Snowflake Connector Available: {SNOWFLAKE_AVAILABLE}")
    print(f"Running in Mock Mode: {not SNOWFLAKE_AVAILABLE}")
    print("-" * 60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestSnowflakeDataFrameCompatibilityChecker,
        TestSnowflakeConnectionManager,
        TestSnowflakeUtilityFunctions,
        TestSnowflakeExceptions,
        TestSnowflakeIntegrationEndToEnd
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
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback.split('Error:')[-1].strip()}")
    
    # Return success status
    return len(result.failures) == 0 and len(result.errors) == 0


if __name__ == '__main__':
    success = run_snowflake_tests()
    
    if success:
        print("\n✅ All Snowflake integration tests passed!")
    else:
        print("\n❌ Some tests failed. Check the output above for details.")
    
    # Get integration status
    print("\nSnowflake Integration Status:")
    status = get_snowflake_integration_status()
    for key, value in status.items():
        if key == 'features':
            print(f"  {key}:")
            for feature, enabled in value.items():
                print(f"    - {feature}: {'✅' if enabled else '❌'}")
        else:
            print(f"  {key}: {value}") 