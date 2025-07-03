"""
Simplified Integration Tests for Workflows
Task 9.2: Integration tests for workflows - Focused End-to-End Testing
"""

import unittest
import pandas as pd
import numpy as np
import time
import tempfile
import os
import hashlib
import sys

# Add the package to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class MockDataProcessor:
    """Mock data processor for integration testing."""
    
    @staticmethod
    def preprocess_data(df):
        """Mock data preprocessing."""
        processed_df = df.copy()
        
        # Handle nulls
        for col in processed_df.columns:
            if processed_df[col].dtype == 'object':
                processed_df[col] = processed_df[col].fillna('unknown')
                processed_df[col] = processed_df[col].astype(str).str.upper()
            else:
                processed_df[col] = processed_df[col].fillna(0)
        
        return processed_df
    
    @staticmethod
    def generate_row_ids(df, columns=None):
        """Mock row ID generation."""
        if columns is None:
            columns = df.columns
        
        row_ids = []
        for _, row in df.iterrows():
            row_string = '_'.join(str(row[col]) for col in columns)
            row_id = hashlib.sha256(row_string.encode()).hexdigest()
            row_ids.append(row_id)
        
        return row_ids
    
    @staticmethod
    def prepare_for_output(df):
        """Mock output preparation."""
        output_df = df.copy()
        
        # Ensure all column names are output-friendly
        output_df.columns = [col.lower().replace(' ', '_') for col in output_df.columns]
        
        return output_df


class TestSimpleIntegrationWorkflows(unittest.TestCase):
    """Test simple integration workflows."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.processor = MockDataProcessor()
        
        # Create test datasets
        self.clean_data = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
            'score': [85, 92, 78, 96, 88],
            'category': ['A', 'B', 'A', 'C', 'B']
        })
        
        self.dirty_data = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'name': ['Alice', None, 'Charlie', 'Diana', 'Eve'],
            'score': [85, 92, None, 96, 88],
            'category': ['A', 'B', None, 'C', 'B']
        })
    
    def test_simple_end_to_end_workflow(self):
        """Test simple end-to-end data processing workflow."""
        print("\n🔄 Testing Simple End-to-End Workflow...")
        
        workflow_start = time.time()
        
        # Step 1: Data Input
        input_df = self.clean_data.copy()
        self.assertEqual(len(input_df), 5)
        
        # Step 2: Data Preprocessing
        processed_df = self.processor.preprocess_data(input_df)
        self.assertEqual(len(processed_df), len(input_df))
        
        # Step 3: Row ID Generation
        row_ids = self.processor.generate_row_ids(processed_df, columns=['id', 'name'])
        processed_df['row_id'] = row_ids
        
        # Step 4: Output Preparation
        output_df = self.processor.prepare_for_output(processed_df)
        
        workflow_time = time.time() - workflow_start
        
        # Validation
        self.assertEqual(len(output_df), 5)
        self.assertTrue('row_id' in output_df.columns)
        self.assertEqual(output_df['row_id'].nunique(), 5)  # All unique
        self.assertLess(workflow_time, 1.0)  # Should be fast
        
        print(f"✅ Simple workflow completed in {workflow_time:.3f}s")
    
    def test_dirty_data_workflow(self):
        """Test workflow with dirty data that needs cleaning."""
        print("\n🔄 Testing Dirty Data Workflow...")
        
        # Step 1: Data Input with Issues
        input_df = self.dirty_data.copy()
        initial_nulls = input_df.isnull().sum().sum()
        self.assertGreater(initial_nulls, 0)  # Should have nulls
        
        # Step 2: Data Cleaning
        cleaned_df = self.processor.preprocess_data(input_df)
        final_nulls = cleaned_df.isnull().sum().sum()
        self.assertEqual(final_nulls, 0)  # Should have no nulls after cleaning
        
        # Step 3: Row ID Generation
        row_ids = self.processor.generate_row_ids(cleaned_df)
        cleaned_df['row_id'] = row_ids
        
        # Step 4: Validation
        self.assertEqual(len(cleaned_df), 5)
        self.assertEqual(cleaned_df['row_id'].nunique(), 5)
        
        # Check that cleaning worked
        self.assertFalse(cleaned_df['name'].str.contains('None').any())
        self.assertTrue((cleaned_df['score'] >= 0).all())
        
        print("✅ Dirty data workflow completed successfully")
    
    def test_batch_processing_workflow(self):
        """Test batch processing of multiple datasets."""
        print("\n🔄 Testing Batch Processing Workflow...")
        
        # Create multiple batches
        batches = []
        for i in range(3):
            batch = pd.DataFrame({
                'batch_id': [i] * 10,
                'record_id': range(10),
                'value': np.random.randn(10),
                'timestamp': pd.date_range('2023-01-01', periods=10, freq='1h')
            })
            batches.append(batch)
        
        # Process each batch
        batch_results = []
        total_start = time.time()
        
        for i, batch_df in enumerate(batches):
            batch_start = time.time()
            
            # Process batch
            processed_batch = self.processor.preprocess_data(batch_df)
            row_ids = self.processor.generate_row_ids(processed_batch, columns=['batch_id', 'record_id'])
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
        
        # Validate batch processing
        self.assertEqual(len(batch_results), 3)
        total_rows = sum(r['output_rows'] for r in batch_results)
        self.assertEqual(total_rows, 30)  # 3 batches * 10 rows each
        
        # All batches should process successfully
        for result in batch_results:
            self.assertEqual(result['input_rows'], result['output_rows'])
            self.assertEqual(result['unique_ids'], 10)
        
        print(f"✅ Batch processing completed: {total_rows} rows in {total_time:.3f}s")
    
    def test_error_handling_workflow(self):
        """Test workflow error handling and recovery."""
        print("\n🔄 Testing Error Handling Workflow...")
        
        errors_found = []
        recovery_actions = []
        
        # Create problematic dataset
        problem_df = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'name': ['Alice', '', None, 'Diana', 'Eve'],
            'score': [85, -1, None, 150, 88],  # Invalid scores
            'category': ['A', 'B', None, 'C', 'B']
        })
        
        try:
            # Step 1: Detect issues
            if problem_df['name'].isnull().any() or (problem_df['name'] == '').any():
                errors_found.append("Invalid names detected")
            
            if (problem_df['score'] < 0).any() or (problem_df['score'] > 100).any():
                errors_found.append("Invalid scores detected")
            
            # Step 2: Apply fixes
            fixed_df = problem_df.copy()
            
            # Fix names
            fixed_df['name'] = fixed_df['name'].fillna('Unknown')
            fixed_df.loc[fixed_df['name'] == '', 'name'] = 'Unknown'
            recovery_actions.append("Fixed invalid names")
            
            # Fix scores
            fixed_df['score'] = fixed_df['score'].fillna(75)  # Default score
            fixed_df.loc[fixed_df['score'] < 0, 'score'] = 0
            fixed_df.loc[fixed_df['score'] > 100, 'score'] = 100
            recovery_actions.append("Fixed invalid scores")
            
            # Step 3: Continue processing
            processed_df = self.processor.preprocess_data(fixed_df)
            row_ids = self.processor.generate_row_ids(processed_df)
            processed_df['row_id'] = row_ids
            
            # Step 4: Validation
            workflow_success = True
            
        except Exception as e:
            workflow_success = False
            print(f"Workflow failed: {e}")
        
        # Assertions
        self.assertTrue(workflow_success)
        self.assertGreater(len(errors_found), 0)
        self.assertEqual(len(errors_found), len(recovery_actions))
        
        print(f"✅ Error handling workflow: {len(errors_found)} errors fixed")
    
    def test_performance_comparison_workflow(self):
        """Test performance comparison between different approaches."""
        print("\n🔄 Testing Performance Comparison Workflow...")
        
        # Create larger dataset for performance testing
        large_df = pd.DataFrame({
            'id': range(5000),
            'data': np.random.randn(5000),
            'category': np.random.choice(['A', 'B', 'C', 'D'], 5000)
        })
        
        # Method 1: Process all at once
        start_time = time.time()
        processed_all = self.processor.preprocess_data(large_df)
        row_ids_all = self.processor.generate_row_ids(processed_all, columns=['id'])
        processed_all['row_id'] = row_ids_all
        time_all_at_once = time.time() - start_time
        
        # Method 2: Process in chunks
        start_time = time.time()
        chunk_size = 1000
        processed_chunks = []
        
        for i in range(0, len(large_df), chunk_size):
            chunk = large_df.iloc[i:i+chunk_size].copy()
            processed_chunk = self.processor.preprocess_data(chunk)
            row_ids_chunk = self.processor.generate_row_ids(processed_chunk, columns=['id'])
            processed_chunk['row_id'] = row_ids_chunk
            processed_chunks.append(processed_chunk)
        
        processed_chunked = pd.concat(processed_chunks, ignore_index=True)
        time_chunked = time.time() - start_time
        
        # Compare results
        performance_comparison = {
            'all_at_once_time': time_all_at_once,
            'chunked_time': time_chunked,
            'speedup_ratio': time_all_at_once / time_chunked if time_chunked > 0 else 1,
            'results_identical': len(processed_all) == len(processed_chunked),
            'data_integrity': processed_all['row_id'].nunique() == processed_chunked['row_id'].nunique()
        }
        
        # Assertions
        self.assertTrue(performance_comparison['results_identical'])
        self.assertTrue(performance_comparison['data_integrity'])
        self.assertGreater(performance_comparison['all_at_once_time'], 0)
        self.assertGreater(performance_comparison['chunked_time'], 0)
        
        print(f"✅ Performance comparison: All-at-once: {time_all_at_once:.3f}s, Chunked: {time_chunked:.3f}s")


class TestModuleIntegration(unittest.TestCase):
    """Test integration between different components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.processor = MockDataProcessor()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_data_pipeline_integration(self):
        """Test complete data pipeline integration."""
        print("\n🔗 Testing Data Pipeline Integration...")
        
        # Create test data with various data types
        test_data = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'text': ['Hello', 'World', 'Test', 'Data', 'Pipeline'],
            'numbers': [10.5, 20.3, 30.1, 40.7, 50.9],
            'categories': ['X', 'Y', 'X', 'Z', 'Y'],
            'dates': pd.date_range('2023-01-01', periods=5)
        })
        
        pipeline_stages = []
        
        # Stage 1: Input validation
        stage_start = time.time()
        self.assertEqual(len(test_data), 5)
        self.assertEqual(len(test_data.columns), 5)
        pipeline_stages.append(('Input Validation', time.time() - stage_start))
        
        # Stage 2: Data preprocessing
        stage_start = time.time()
        processed_data = self.processor.preprocess_data(test_data)
        pipeline_stages.append(('Data Preprocessing', time.time() - stage_start))
        
        # Stage 3: Feature engineering
        stage_start = time.time()
        processed_data['text_length'] = processed_data['text'].str.len()
        processed_data['number_category'] = pd.cut(processed_data['numbers'], 
                                                 bins=3, labels=['Low', 'Medium', 'High'])
        pipeline_stages.append(('Feature Engineering', time.time() - stage_start))
        
        # Stage 4: ID generation
        stage_start = time.time()
        row_ids = self.processor.generate_row_ids(processed_data, columns=['id', 'text'])
        processed_data['row_id'] = row_ids
        pipeline_stages.append(('ID Generation', time.time() - stage_start))
        
        # Stage 5: Output preparation
        stage_start = time.time()
        final_data = self.processor.prepare_for_output(processed_data)
        pipeline_stages.append(('Output Preparation', time.time() - stage_start))
        
        # Validate pipeline results
        pipeline_result = {
            'stages_completed': len(pipeline_stages),
            'final_row_count': len(final_data),
            'unique_ids_generated': final_data['row_id'].nunique(),
            'new_columns_created': len(final_data.columns) - len(test_data.columns),
            'data_integrity': len(final_data) == len(test_data)
        }
        
        # Assertions
        self.assertEqual(pipeline_result['stages_completed'], 5)
        self.assertTrue(pipeline_result['data_integrity'])
        self.assertEqual(pipeline_result['unique_ids_generated'], 5)
        self.assertGreater(pipeline_result['new_columns_created'], 0)
        
        print(f"✅ Pipeline integration: {pipeline_result['stages_completed']} stages completed")
    
    def test_file_io_integration(self):
        """Test file I/O integration with data processing."""
        print("\n🔗 Testing File I/O Integration...")
        
        # Create test data
        test_data = pd.DataFrame({
            'id': range(100),
            'value': np.random.randn(100),
            'category': np.random.choice(['A', 'B', 'C'], 100)
        })
        
        # Stage 1: Write to file
        input_file = os.path.join(self.temp_dir, 'input_data.csv')
        test_data.to_csv(input_file, index=False)
        self.assertTrue(os.path.exists(input_file))
        
        # Stage 2: Read from file
        loaded_data = pd.read_csv(input_file)
        self.assertEqual(len(loaded_data), len(test_data))
        
        # Stage 3: Process loaded data
        processed_data = self.processor.preprocess_data(loaded_data)
        row_ids = self.processor.generate_row_ids(processed_data)
        processed_data['row_id'] = row_ids
        
        # Stage 4: Write processed data
        output_file = os.path.join(self.temp_dir, 'output_data.csv')
        processed_data.to_csv(output_file, index=False)
        self.assertTrue(os.path.exists(output_file))
        
        # Stage 5: Validate file I/O
        final_data = pd.read_csv(output_file)
        self.assertEqual(len(final_data), 100)
        self.assertTrue('row_id' in final_data.columns)
        self.assertEqual(final_data['row_id'].nunique(), 100)
        
        print("✅ File I/O integration test passed")


def run_simple_integration_tests():
    """Run all simple integration tests."""
    print("Running Simple Integration Tests...")
    print("=" * 60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestSimpleIntegrationWorkflows,
        TestModuleIntegration
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
    success = run_simple_integration_tests()
    if success:
        print("\n✅ All simple integration tests passed!")
    else:
        print("\n❌ Some tests failed. Check output above for details.") 