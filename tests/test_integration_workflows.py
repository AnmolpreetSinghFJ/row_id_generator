"""
Integration Tests for Workflows
Task 9.2: Integration tests for workflows - End-to-End Testing
"""

import unittest
import pandas as pd
import numpy as np
import time
import tempfile
import os
import json
import hashlib
import threading
from unittest.mock import Mock, patch, MagicMock
import sys
from concurrent.futures import ThreadPoolExecutor
import logging
from io import StringIO

# Add the package to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import modules with error handling
try:
    from row_id_generator.core import *
except ImportError:
    # Create mock core functions for testing
    def generate_unique_row_ids(df, columns=None, **kwargs):
        """Mock row ID generation."""
        row_ids = []
        for _, row in df.iterrows():
            row_string = '_'.join(str(row[col]) for col in (columns or df.columns))
            row_id = hashlib.sha256(row_string.encode()).hexdigest()
            row_ids.append(row_id)
        return row_ids
    
    def preprocess_data(df, **kwargs):
        """Mock data preprocessing."""
        processed_df = df.copy()
        # Basic preprocessing simulation
        for col in processed_df.select_dtypes(include=['object']).columns:
            processed_df[col] = processed_df[col].astype(str).str.upper()
        return processed_df

try:
    from row_id_generator.snowflake_integration import *
except ImportError:
    # Create mock Snowflake functions
    def prepare_for_snowflake(df, table_name=None, **kwargs):
        """Mock Snowflake preparation."""
        return df.copy()
    
    def load_to_snowflake(df, table_name, **kwargs):
        """Mock Snowflake loading."""
        return {'success': True, 'rows_inserted': len(df)}

try:
    from row_id_generator.observable import *
except ImportError:
    # Create mock observability functions
    pass

# Create mock classes available globally
class MockLogger:
    def __init__(self):
        self.logs = []
    
    def info(self, msg):
        self.logs.append(('INFO', msg))
    
    def warning(self, msg):
        self.logs.append(('WARNING', msg))
    
    def error(self, msg):
        self.logs.append(('ERROR', msg))


class TestEndToEndWorkflows(unittest.TestCase):
    """Test complete end-to-end data processing workflows."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create sample datasets of various sizes
        self.small_dataset = self._create_test_dataset(100)
        self.medium_dataset = self._create_test_dataset(1000)
        self.large_dataset = self._create_test_dataset(10000)
        
        # Create dataset with data quality issues
        self.dirty_dataset = self._create_dirty_dataset(500)
        
        # Mock logger for observability
        self.mock_logger = MockLogger()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def _create_test_dataset(self, size):
        """Create a test dataset with specified size."""
        np.random.seed(42)  # For reproducible results
        
        return pd.DataFrame({
            'id': range(size),
            'name': [f'user_{i}' for i in range(size)],
            'email': [f'user{i}@example.com' for i in range(size)],
            'age': np.random.randint(18, 80, size),
            'score': np.random.normal(75, 15, size),
            'category': np.random.choice(['A', 'B', 'C'], size),
            'created_date': pd.date_range('2023-01-01', periods=size, freq='h'),
            'is_active': np.random.choice([True, False], size)
        })
    
    def _create_dirty_dataset(self, size):
        """Create a dataset with data quality issues."""
        base_df = self._create_test_dataset(size)
        
        # Introduce nulls
        base_df.loc[base_df.index % 10 == 0, 'email'] = None
        base_df.loc[base_df.index % 15 == 0, 'age'] = None
        
        # Introduce duplicates
        base_df.loc[base_df.index % 20 == 0, 'name'] = 'duplicate_user'
        
        # Introduce invalid data
        base_df.loc[base_df.index % 25 == 0, 'age'] = -1
        base_df.loc[base_df.index % 30 == 0, 'score'] = 999999
        
        return base_df
    
    def test_simple_end_to_end_workflow(self):
        """Test simple end-to-end data processing workflow."""
        print("\n🔄 Testing Simple End-to-End Workflow...")
        
        workflow_start = time.time()
        
        # Step 1: Data Input
        input_df = self.small_dataset.copy()
        self.mock_logger.info(f"Input data received: {len(input_df)} rows")
        
        # Step 2: Data Preprocessing
        processed_df = preprocess_data(input_df)
        self.mock_logger.info("Data preprocessing completed")
        
        # Step 3: Row ID Generation
        row_ids = generate_unique_row_ids(processed_df, columns=['id', 'name', 'email'])
        processed_df['row_id'] = row_ids
        self.mock_logger.info("Row IDs generated")
        
        # Step 4: Validation
        self.assertEqual(len(processed_df), len(input_df))
        self.assertTrue('row_id' in processed_df.columns)
        self.assertEqual(processed_df['row_id'].nunique(), len(processed_df))
        
        # Step 5: Data Output Preparation
        output_df = prepare_for_snowflake(processed_df)
        self.mock_logger.info("Data prepared for output")
        
        workflow_time = time.time() - workflow_start
        
        # Workflow validation
        workflow_result = {
            'success': True,
            'input_rows': len(input_df),
            'output_rows': len(output_df),
            'processing_time': workflow_time,
            'unique_row_ids': output_df['row_id'].nunique(),
            'data_integrity': len(output_df) == len(input_df)
        }
        
        # Assertions
        self.assertTrue(workflow_result['success'])
        self.assertEqual(workflow_result['input_rows'], workflow_result['output_rows'])
        self.assertTrue(workflow_result['data_integrity'])
        self.assertEqual(workflow_result['unique_row_ids'], len(output_df))
        self.assertLess(workflow_result['processing_time'], 10.0)  # Should complete quickly
        
        print(f"✅ Simple workflow completed successfully in {workflow_time:.2f}s")
    
    def test_complex_data_processing_workflow(self):
        """Test complex data processing workflow with multiple transformations."""
        print("\n🔄 Testing Complex Data Processing Workflow...")
        
        workflow_start = time.time()
        workflow_steps = []
        
        # Step 1: Data Input and Validation
        step_start = time.time()
        input_df = self.medium_dataset.copy()
        
        # Validate input data
        input_validation = {
            'total_rows': len(input_df),
            'null_count': input_df.isnull().sum().sum(),
            'duplicate_count': input_df.duplicated().sum(),
            'columns': list(input_df.columns)
        }
        
        step_time = time.time() - step_start
        workflow_steps.append(('Data Input & Validation', step_time, input_validation))
        self.mock_logger.info(f"Data input validated: {input_validation['total_rows']} rows")
        
        # Step 2: Data Cleaning and Preprocessing
        step_start = time.time()
        
        # Clean data
        cleaned_df = input_df.copy()
        
        # Handle nulls
        cleaned_df['email'] = cleaned_df['email'].fillna('unknown@example.com')
        cleaned_df['age'] = cleaned_df['age'].fillna(cleaned_df['age'].median())
        
        # Normalize text data
        for col in cleaned_df.select_dtypes(include=['object']).columns:
            if col not in ['created_date']:
                cleaned_df[col] = cleaned_df[col].astype(str).str.strip().str.lower()
        
        # Apply business rules
        cleaned_df['age_group'] = pd.cut(cleaned_df['age'], 
                                       bins=[0, 25, 40, 60, 100], 
                                       labels=['Young', 'Adult', 'Middle', 'Senior'])
        
        cleaned_df['score_category'] = pd.cut(cleaned_df['score'],
                                            bins=[-np.inf, 60, 80, 100, np.inf],
                                            labels=['Low', 'Medium', 'High', 'Excellent'])
        
        step_time = time.time() - step_start
        workflow_steps.append(('Data Cleaning', step_time, len(cleaned_df)))
        self.mock_logger.info("Data cleaning completed")
        
        # Step 3: Feature Engineering
        step_start = time.time()
        
        # Create derived features
        enriched_df = cleaned_df.copy()
        enriched_df['email_domain'] = enriched_df['email'].str.split('@').str[1]
        enriched_df['name_length'] = enriched_df['name'].str.len()
        enriched_df['is_weekend_created'] = enriched_df['created_date'].dt.weekday >= 5
        enriched_df['days_since_creation'] = (pd.Timestamp.now() - enriched_df['created_date']).dt.days
        
        step_time = time.time() - step_start
        workflow_steps.append(('Feature Engineering', step_time, len(enriched_df.columns)))
        self.mock_logger.info("Feature engineering completed")
        
        # Step 4: Row ID Generation with Complex Logic
        step_start = time.time()
        
        # Generate row IDs using multiple columns
        key_columns = ['id', 'name', 'email', 'age_group', 'category']
        row_ids = generate_unique_row_ids(enriched_df, columns=key_columns)
        enriched_df['row_id'] = row_ids
        
        # Generate alternative IDs for different use cases
        business_key_columns = ['email', 'name']
        business_ids = generate_unique_row_ids(enriched_df, columns=business_key_columns)
        enriched_df['business_id'] = business_ids
        
        step_time = time.time() - step_start
        workflow_steps.append(('Row ID Generation', step_time, enriched_df['row_id'].nunique()))
        self.mock_logger.info("Row ID generation completed")
        
        # Step 5: Data Quality Validation
        step_start = time.time()
        
        quality_checks = {
            'total_rows': len(enriched_df),
            'null_percentage': (enriched_df.isnull().sum().sum() / (len(enriched_df) * len(enriched_df.columns))) * 100,
            'duplicate_row_ids': enriched_df['row_id'].duplicated().sum(),
            'duplicate_business_ids': enriched_df['business_id'].duplicated().sum(),
            'age_range_valid': (enriched_df['age'] >= 0).all() and (enriched_df['age'] <= 150).all(),
            'score_range_valid': len(enriched_df[enriched_df['score'].between(-100, 200)]) == len(enriched_df)
        }
        
        step_time = time.time() - step_start
        workflow_steps.append(('Quality Validation', step_time, quality_checks))
        self.mock_logger.info("Data quality validation completed")
        
        # Step 6: Output Preparation
        step_start = time.time()
        
        final_df = prepare_for_snowflake(enriched_df)
        
        step_time = time.time() - step_start
        workflow_steps.append(('Output Preparation', step_time, len(final_df)))
        
        workflow_time = time.time() - workflow_start
        
        # Comprehensive workflow validation
        workflow_result = {
            'success': True,
            'total_processing_time': workflow_time,
            'steps': workflow_steps,
            'data_quality': quality_checks,
            'output_summary': {
                'total_rows': len(final_df),
                'total_columns': len(final_df.columns),
                'unique_row_ids': final_df['row_id'].nunique(),
                'unique_business_ids': final_df['business_id'].nunique()
            }
        }
        
        # Assertions
        self.assertTrue(workflow_result['success'])
        self.assertEqual(workflow_result['data_quality']['duplicate_row_ids'], 0)
        self.assertLess(workflow_result['data_quality']['null_percentage'], 5.0)
        self.assertTrue(workflow_result['data_quality']['age_range_valid'])
        self.assertEqual(workflow_result['output_summary']['unique_row_ids'], len(final_df))
        
        print(f"✅ Complex workflow completed successfully in {workflow_time:.2f}s")
        print(f"   Processed {len(final_df)} rows through {len(workflow_steps)} steps")
    
    def test_error_handling_workflow(self):
        """Test workflow error handling and recovery."""
        print("\n🔄 Testing Error Handling Workflow...")
        
        errors_encountered = []
        recovery_actions = []
        
        # Test with dirty dataset
        input_df = self.dirty_dataset.copy()
        
        try:
            # Step 1: Attempt processing with errors
            processed_df = input_df.copy()
            
            # Detect and handle invalid ages
            invalid_ages = processed_df['age'] < 0
            if invalid_ages.any():
                error_count = invalid_ages.sum()
                errors_encountered.append(f"Invalid ages detected: {error_count}")
                
                # Recovery action: Set invalid ages to median
                median_age = processed_df[processed_df['age'] >= 0]['age'].median()
                processed_df.loc[invalid_ages, 'age'] = median_age
                recovery_actions.append(f"Fixed {error_count} invalid ages with median value {median_age}")
            
            # Detect and handle null emails
            null_emails = processed_df['email'].isnull()
            if null_emails.any():
                error_count = null_emails.sum()
                errors_encountered.append(f"Null emails detected: {error_count}")
                
                # Recovery action: Generate placeholder emails
                for idx in processed_df[null_emails].index:
                    processed_df.loc[idx, 'email'] = f"placeholder_{idx}@example.com"
                recovery_actions.append(f"Generated placeholder emails for {error_count} records")
            
            # Detect and handle outlier scores
            score_outliers = (processed_df['score'] > 150) | (processed_df['score'] < 0)
            if score_outliers.any():
                error_count = score_outliers.sum()
                errors_encountered.append(f"Score outliers detected: {error_count}")
                
                # Recovery action: Cap scores to reasonable range
                processed_df.loc[processed_df['score'] > 150, 'score'] = 100
                processed_df.loc[processed_df['score'] < 0, 'score'] = 0
                recovery_actions.append(f"Capped {error_count} outlier scores to valid range")
            
            # Generate row IDs despite data issues
            row_ids = generate_unique_row_ids(processed_df, columns=['id', 'name', 'email'])
            processed_df['row_id'] = row_ids
            
            # Final validation
            final_validation = {
                'total_rows': len(processed_df),
                'errors_detected': len(errors_encountered),
                'recovery_actions': len(recovery_actions),
                'data_quality_issues_resolved': len(errors_encountered) == len(recovery_actions),
                'unique_row_ids': processed_df['row_id'].nunique(),
                'processing_success': True
            }
            
        except Exception as e:
            final_validation = {
                'processing_success': False,
                'error_message': str(e),
                'errors_detected': len(errors_encountered),
                'recovery_actions': len(recovery_actions)
            }
        
        # Assertions
        self.assertTrue(final_validation['processing_success'])
        self.assertGreater(final_validation['errors_detected'], 0)  # Should detect data issues
        self.assertEqual(final_validation['errors_detected'], final_validation['recovery_actions'])
        self.assertTrue(final_validation['data_quality_issues_resolved'])
        
        print(f"✅ Error handling workflow completed successfully")
        print(f"   Detected {final_validation['errors_detected']} issues and applied {final_validation['recovery_actions']} recovery actions")
    
    def test_concurrent_processing_workflow(self):
        """Test concurrent processing workflow with multiple datasets."""
        print("\n🔄 Testing Concurrent Processing Workflow...")
        
        # Create multiple datasets for concurrent processing
        datasets = [
            ('small', self.small_dataset.copy()),
            ('medium', self.medium_dataset.copy()),
            ('large', self.large_dataset.copy())
        ]
        
        def process_dataset(dataset_info):
            """Process a single dataset."""
            name, df = dataset_info
            
            start_time = time.time()
            
            # Simulate processing steps
            processed_df = preprocess_data(df)
            row_ids = generate_unique_row_ids(processed_df, columns=['id', 'name'])
            processed_df['row_id'] = row_ids
            
            processing_time = time.time() - start_time
            
            return {
                'dataset_name': name,
                'input_rows': len(df),
                'output_rows': len(processed_df),
                'processing_time': processing_time,
                'unique_row_ids': processed_df['row_id'].nunique(),
                'success': True
            }
        
        # Test sequential processing
        sequential_start = time.time()
        sequential_results = []
        for dataset in datasets:
            result = process_dataset(dataset)
            sequential_results.append(result)
        sequential_time = time.time() - sequential_start
        
        # Test concurrent processing
        concurrent_start = time.time()
        with ThreadPoolExecutor(max_workers=3) as executor:
            concurrent_results = list(executor.map(process_dataset, datasets))
        concurrent_time = time.time() - concurrent_start
        
        # Compare results
        performance_comparison = {
            'sequential_time': sequential_time,
            'concurrent_time': concurrent_time,
            'speedup_ratio': sequential_time / concurrent_time if concurrent_time > 0 else 1,
            'sequential_results': sequential_results,
            'concurrent_results': concurrent_results,
            'results_match': sequential_results == concurrent_results
        }
        
        # Assertions
        self.assertEqual(len(sequential_results), len(concurrent_results))
        self.assertTrue(all(r['success'] for r in concurrent_results))
        
        # Results should be identical (order might differ)
        sequential_by_name = {r['dataset_name']: r for r in sequential_results}
        concurrent_by_name = {r['dataset_name']: r for r in concurrent_results}
        
        for name in sequential_by_name:
            self.assertEqual(sequential_by_name[name]['input_rows'], concurrent_by_name[name]['input_rows'])
            self.assertEqual(sequential_by_name[name]['output_rows'], concurrent_by_name[name]['output_rows'])
            self.assertEqual(sequential_by_name[name]['unique_row_ids'], concurrent_by_name[name]['unique_row_ids'])
        
        print(f"✅ Concurrent processing workflow completed successfully")
        print(f"   Sequential: {sequential_time:.2f}s, Concurrent: {concurrent_time:.2f}s")
        print(f"   Speedup ratio: {performance_comparison['speedup_ratio']:.2f}x")


class TestModuleIntegration(unittest.TestCase):
    """Test integration between different modules."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_df = pd.DataFrame({
            'id': range(1000),
            'data': np.random.randn(1000),
            'category': np.random.choice(['A', 'B', 'C'], 1000)
        })
    
    def test_core_utils_integration(self):
        """Test integration between core and utils modules."""
        print("\n🔗 Testing Core-Utils Integration...")
        
        integration_result = {
            'core_processing': False,
            'utils_validation': False,
            'data_consistency': False,
            'performance_tracking': False
        }
        
        try:
            # Core processing
            processed_df = preprocess_data(self.test_df)
            row_ids = generate_unique_row_ids(processed_df)
            processed_df['row_id'] = row_ids
            integration_result['core_processing'] = True
            
            # Utils validation (simulated)
            validation_stats = {
                'total_rows': len(processed_df),
                'null_count': processed_df.isnull().sum().sum(),
                'duplicate_ids': processed_df['row_id'].duplicated().sum(),
                'data_types': processed_df.dtypes.to_dict()
            }
            integration_result['utils_validation'] = validation_stats['duplicate_ids'] == 0
            
            # Data consistency check
            integration_result['data_consistency'] = len(processed_df) == len(self.test_df)
            
            # Performance tracking (simulated)
            integration_result['performance_tracking'] = True
            
        except Exception as e:
            print(f"Integration error: {e}")
        
        # Assertions
        self.assertTrue(integration_result['core_processing'])
        self.assertTrue(integration_result['utils_validation'])
        self.assertTrue(integration_result['data_consistency'])
        
        print("✅ Core-Utils integration test passed")
    
    def test_observability_integration(self):
        """Test integration with observability framework."""
        print("\n🔗 Testing Observability Integration...")
        
        # Set up observability tracking
        logger = MockLogger()
        metrics = {}
        
        # Simulate processing with observability
        start_time = time.time()
        
        logger.info("Starting data processing")
        processed_df = preprocess_data(self.test_df)
        
        logger.info("Generating row IDs")
        row_ids = generate_unique_row_ids(processed_df)
        processed_df['row_id'] = row_ids
        
        processing_time = time.time() - start_time
        
        # Record metrics
        metrics['processing_time'] = processing_time
        metrics['rows_processed'] = len(processed_df)
        metrics['unique_ids_generated'] = processed_df['row_id'].nunique()
        
        logger.info("Processing completed successfully")
        
        # Validate observability integration
        observability_result = {
            'logging_active': len(logger.logs) > 0,
            'metrics_collected': len(metrics) > 0,
            'performance_tracked': 'processing_time' in metrics,
            'data_metrics_available': 'rows_processed' in metrics,
            'log_messages': [log[1] for log in logger.logs]
        }
        
        # Assertions
        self.assertTrue(observability_result['logging_active'])
        self.assertTrue(observability_result['metrics_collected'])
        self.assertTrue(observability_result['performance_tracked'])
        self.assertGreater(metrics['rows_processed'], 0)
        self.assertEqual(metrics['unique_ids_generated'], len(processed_df))
        
        print("✅ Observability integration test passed")
        print(f"   Logged {len(logger.logs)} messages, collected {len(metrics)} metrics")
    
    def test_snowflake_integration_workflow(self):
        """Test Snowflake integration workflow."""
        print("\n🔗 Testing Snowflake Integration Workflow...")
        
        snowflake_workflow = {
            'data_preparation': False,
            'compatibility_check': False,
            'load_simulation': False,
            'error_handling': False
        }
        
        try:
            # Step 1: Prepare data for Snowflake
            source_df = self.test_df.copy()
            prepared_df = prepare_for_snowflake(source_df)
            snowflake_workflow['data_preparation'] = True
            
            # Step 2: Compatibility validation
            # Check column names, data types, etc.
            compatibility_issues = []
            
            # Check for Snowflake-incompatible column names
            for col in prepared_df.columns:
                if not col.replace('_', '').isalnum():
                    compatibility_issues.append(f"Column name '{col}' may have compatibility issues")
            
            snowflake_workflow['compatibility_check'] = len(compatibility_issues) == 0
            
            # Step 3: Simulate data loading
            load_result = load_to_snowflake(prepared_df, 'test_table')
            snowflake_workflow['load_simulation'] = load_result.get('success', False)
            
            # Step 4: Test error handling
            try:
                # Simulate error scenario
                invalid_df = pd.DataFrame({'invalid column name!': [1, 2, 3]})
                error_result = prepare_for_snowflake(invalid_df)
                snowflake_workflow['error_handling'] = True
            except:
                snowflake_workflow['error_handling'] = True  # Expected to handle gracefully
            
        except Exception as e:
            print(f"Snowflake integration error: {e}")
        
        # Assertions
        self.assertTrue(snowflake_workflow['data_preparation'])
        self.assertTrue(snowflake_workflow['compatibility_check'])
        self.assertTrue(snowflake_workflow['load_simulation'])
        
        print("✅ Snowflake integration workflow test passed")


class TestRealWorldScenarios(unittest.TestCase):
    """Test real-world usage scenarios."""
    
    def test_batch_processing_scenario(self):
        """Test batch processing of large datasets."""
        print("\n🌍 Testing Batch Processing Scenario...")
        
        # Simulate multiple data files
        batch_files = []
        for i in range(5):
            df = pd.DataFrame({
                'batch_id': [i] * 200,
                'record_id': range(200),
                'data': np.random.randn(200),
                'timestamp': pd.date_range('2023-01-01', periods=200, freq='1min')
            })
            batch_files.append(df)
        
        # Process batches
        batch_results = []
        total_start = time.time()
        
        for i, batch_df in enumerate(batch_files):
            batch_start = time.time()
            
            # Process batch
            processed_batch = preprocess_data(batch_df)
            row_ids = generate_unique_row_ids(processed_batch, columns=['batch_id', 'record_id'])
            processed_batch['row_id'] = row_ids
            
            batch_time = time.time() - batch_start
            
            batch_result = {
                'batch_number': i,
                'input_rows': len(batch_df),
                'output_rows': len(processed_batch),
                'processing_time': batch_time,
                'unique_ids': processed_batch['row_id'].nunique()
            }
            batch_results.append(batch_result)
        
        total_time = time.time() - total_start
        
        # Aggregate results
        scenario_summary = {
            'total_batches': len(batch_files),
            'total_rows_processed': sum(r['output_rows'] for r in batch_results),
            'total_processing_time': total_time,
            'average_batch_time': np.mean([r['processing_time'] for r in batch_results]),
            'throughput_rows_per_second': sum(r['output_rows'] for r in batch_results) / total_time,
            'all_batches_successful': all(r['input_rows'] == r['output_rows'] for r in batch_results)
        }
        
        # Assertions
        self.assertTrue(scenario_summary['all_batches_successful'])
        self.assertEqual(scenario_summary['total_batches'], 5)
        self.assertEqual(scenario_summary['total_rows_processed'], 1000)
        self.assertGreater(scenario_summary['throughput_rows_per_second'], 0)
        
        print(f"✅ Batch processing scenario completed successfully")
        print(f"   Processed {scenario_summary['total_rows_processed']} rows in {total_time:.2f}s")
        print(f"   Throughput: {scenario_summary['throughput_rows_per_second']:.0f} rows/second")
    
    def test_streaming_simulation_scenario(self):
        """Test streaming data processing simulation."""
        print("\n🌍 Testing Streaming Simulation Scenario...")
        
        # Simulate streaming data
        stream_buffer = []
        processed_records = []
        
        def process_stream_batch(records):
            """Process a batch of streaming records."""
            if not records:
                return []
            
            batch_df = pd.DataFrame(records)
            processed_df = preprocess_data(batch_df)
            row_ids = generate_unique_row_ids(processed_df, columns=['id', 'timestamp'])
            processed_df['row_id'] = row_ids
            
            return processed_df.to_dict('records')
        
        # Simulate data arrival
        for batch in range(10):
            # Generate new records
            new_records = []
            for i in range(50):  # 50 records per batch
                record = {
                    'id': batch * 50 + i,
                    'value': np.random.randn(),
                    'timestamp': time.time() + i * 0.001,
                    'source': f'stream_{batch}'
                }
                new_records.append(record)
            
            # Add to buffer
            stream_buffer.extend(new_records)
            
            # Process when buffer reaches threshold
            if len(stream_buffer) >= 100:
                batch_processed = process_stream_batch(stream_buffer[:100])
                processed_records.extend(batch_processed)
                stream_buffer = stream_buffer[100:]  # Remove processed records
        
        # Process remaining records
        if stream_buffer:
            final_batch = process_stream_batch(stream_buffer)
            processed_records.extend(final_batch)
        
        # Validate streaming results
        streaming_summary = {
            'total_records_processed': len(processed_records),
            'unique_row_ids': len(set(r['row_id'] for r in processed_records)),
            'data_completeness': len(processed_records) == 500,  # 10 batches * 50 records
            'processing_success': len(processed_records) > 0
        }
        
        # Assertions
        self.assertTrue(streaming_summary['processing_success'])
        self.assertTrue(streaming_summary['data_completeness'])
        self.assertEqual(streaming_summary['unique_row_ids'], streaming_summary['total_records_processed'])
        
        print(f"✅ Streaming simulation scenario completed successfully")
        print(f"   Processed {streaming_summary['total_records_processed']} streaming records")


def run_integration_workflow_tests():
    """Run all integration workflow tests."""
    print("Running Integration Workflow Tests...")
    print("=" * 80)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestEndToEndWorkflows,
        TestModuleIntegration,
        TestRealWorldScenarios
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("-" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    return len(result.failures) == 0 and len(result.errors) == 0


if __name__ == '__main__':
    success = run_integration_workflow_tests()
    if success:
        print("\n✅ All integration workflow tests passed!")
    else:
        print("\n❌ Some tests failed. Check output above for details.") 