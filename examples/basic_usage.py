"""
Basic usage examples for row-id-generator package.

This comprehensive example demonstrates all major features of the package:
- Basic row ID generation
- Advanced configuration options
- Observable API for production monitoring
- Error handling and validation
- Performance optimization techniques
"""

import pandas as pd
import time
import numpy as np
from row_id_generator import (
    generate_unique_row_ids,
    generate_row_ids_simple,
    generate_row_ids_fast,
    select_columns_for_hashing,
    prepare_data_for_hashing,
    create_observable_engine,
    create_minimal_observable_engine
)

def basic_usage_example():
    """Demonstrate basic row ID generation."""
    print("🔥 BASIC USAGE EXAMPLE")
    print("=" * 50)
    
    # Create sample data
    print("\n1. Creating sample DataFrame...")
    df = pd.DataFrame({
        'email': [
            'alice@example.com',
            'bob@example.com', 
            'charlie@example.com',
            'diana@example.com',
            'eve@example.com'
        ],
        'name': ['Alice Johnson', 'Bob Smith', 'Charlie Brown', 'Diana Prince', 'Eve Davis'],
        'age': [28, 34, 29, 32, 26],
        'department': ['Engineering', 'Marketing', 'Engineering', 'Sales', 'Marketing'],
        'salary': [85000, 72000, 90000, 68000, 75000]
    })
    
    print(f"Original DataFrame shape: {df.shape}")
    print("\nOriginal DataFrame:")
    print(df.to_string(index=False))
    
    # Generate row IDs with default settings
    print("\n2. Generating row IDs with intelligent defaults...")
    try:
        result_df = generate_unique_row_ids(df)
        
        print(f"✅ Row IDs generated successfully!")
        print(f"Result DataFrame shape: {result_df.shape}")
        
        print("\nDataFrame with row IDs:")
        print(result_df[['row_id', 'email', 'name', 'department']].to_string(index=False))
        
        # Check uniqueness
        unique_ids = result_df['row_id'].nunique()
        total_rows = len(result_df)
        print(f"\nUniqueness check: {unique_ids}/{total_rows} unique IDs ({unique_ids/total_rows:.1%})")
        
    except Exception as e:
        print(f"❌ Error generating row IDs: {e}")
        return None
    
    return result_df

def advanced_configuration_example():
    """Demonstrate advanced configuration options."""
    print("\n\n🎯 ADVANCED CONFIGURATION EXAMPLE")
    print("=" * 50)
    
    # Create more complex sample data
    df = pd.DataFrame({
        'user_id': ['U001', 'U002', 'U003', 'U004'],
        'email': ['alice@corp.com', 'bob@startup.io', None, 'diana@agency.co'],
        'first_name': ['Alice', 'Bob', 'Charlie', 'Diana'],
        'last_name': ['Johnson', None, 'Brown', 'Prince'],
        'phone': ['+1-555-0001', '+1-555-0002', '+1-555-0003', None],
        'created_at': pd.to_datetime(['2024-01-15', '2024-01-16', '2024-01-17', '2024-01-18']),
        'score': [95.5, 87.2, None, 92.8]
    })
    
    print("\nSample DataFrame with mixed data types and NULLs:")
    print(df.to_string(index=False))
    
    # Advanced configuration with comprehensive audit trail
    print("\n1. Advanced configuration with detailed control...")
    try:
        result_df = generate_unique_row_ids(
            df=df,
            columns=['user_id', 'email', 'phone'],  # Manually specify high-quality columns
            id_column_name='customer_hash_id',      # Custom ID column name
            uniqueness_threshold=0.90,              # Slightly lower threshold for demo
            separator='::',                         # Custom separator
            show_progress=True,                     # Show progress
            enable_monitoring=True,                 # Enable observability
            enable_quality_checks=True,             # Data quality validation
            return_audit_trail=True                 # Return detailed info
        )
        
        # Handle results based on return type
        if isinstance(result_df, dict):
            df_with_ids = result_df['dataframe']
            audit_trail = result_df.get('audit_trail', {})
            selected_columns = result_df.get('selected_columns', [])
            
            print(f"✅ Advanced row IDs generated successfully!")
            print(f"Selected columns: {selected_columns}")
            print(f"Processing details: {audit_trail}")
            
            print(f"\nResult DataFrame:")
            print(df_with_ids[['customer_hash_id', 'user_id', 'email']].to_string(index=False))
        else:
            print(f"✅ Simple DataFrame returned")
            print(result_df[['customer_hash_id', 'user_id', 'email']].to_string(index=False))
            
    except Exception as e:
        print(f"❌ Error with advanced configuration: {e}")
        import traceback
        traceback.print_exc()

def performance_variants_example():
    """Demonstrate different performance variants."""
    print("\n\n⚡ PERFORMANCE VARIANTS EXAMPLE")
    print("=" * 50)
    
    # Create larger sample dataset
    np.random.seed(42)
    size = 1000
    df_large = pd.DataFrame({
        'id': range(size),
        'email': [f'user{i}@example.com' for i in range(size)],
        'name': [f'User {i}' for i in range(size)],
        'category': np.random.choice(['A', 'B', 'C', 'D'], size),
        'value': np.random.randn(size),
        'timestamp': pd.date_range('2024-01-01', periods=size, freq='H')
    })
    
    print(f"Large DataFrame: {df_large.shape[0]:,} rows")
    
    # Test different performance variants
    variants = [
        ("Simple (minimal overhead)", lambda: generate_row_ids_simple(df_large, columns=['email'])),
        ("Fast (maximum performance)", lambda: generate_row_ids_fast(df_large, columns=['email'])),
        ("Full (all features)", lambda: generate_unique_row_ids(df_large, columns=['email'], show_progress=False))
    ]
    
    results = {}
    for name, func in variants:
        print(f"\n{name}:")
        start_time = time.time()
        try:
            result = func()
            duration = time.time() - start_time
            results[name] = {
                'duration': duration,
                'rows': len(result),
                'throughput': len(result) / duration if duration > 0 else 0
            }
            print(f"  ✅ {len(result):,} rows processed in {duration:.3f}s ({len(result)/duration:,.0f} rows/sec)")
        except Exception as e:
            print(f"  ❌ Error: {e}")
            results[name] = {'error': str(e)}
    
    return results

def utility_functions_example():
    """Demonstrate utility functions."""
    print("\n\n🛠️ UTILITY FUNCTIONS EXAMPLE")
    print("=" * 50)
    
    # Sample data with quality issues
    df = pd.DataFrame({
        'email': ['alice@example.com', 'bob@EXAMPLE.COM', '  charlie@test.org  '],
        'name': ['Alice Johnson', 'BOB SMITH', 'charlie brown'],
        'description': ['Great customer', None, 'VIP customer'],
        'amount': [99.999, 150.0, 75.123456],
        'created_at': ['2024-01-15', '2024/01/16', '15-Jan-2024']
    })
    
    print("Original data with quality issues:")
    print(df.to_string(index=False))
    
    # 1. Column selection
    print("\n1. Intelligent column selection:")
    try:
        selected_columns = select_columns_for_hashing(
            df=df,
            uniqueness_threshold=0.8,
            include_email=True
        )
        print(f"Selected columns: {selected_columns}")
    except Exception as e:
        print(f"Error in column selection: {e}")
    
    # 2. Data preprocessing
    print("\n2. Data preprocessing:")
    try:
        processed_df = prepare_data_for_hashing(df, columns=['email', 'name'])
        print("Processed data:")
        print(processed_df.to_string(index=False))
    except Exception as e:
        print(f"Error in data preprocessing: {e}")

def observable_api_example():
    """Demonstrate Observable API for production monitoring."""
    print("\n\n📊 OBSERVABLE API EXAMPLE")
    print("=" * 50)
    
    # Create sample data
    df = pd.DataFrame({
        'transaction_id': ['T001', 'T002', 'T003', 'T004', 'T005'],
        'user_email': ['alice@corp.com', 'bob@startup.io', 'charlie@agency.co', 'diana@enterprise.com', 'eve@small.biz'],
        'amount': [99.99, 149.50, 79.99, 299.99, 49.99],
        'currency': ['USD', 'USD', 'EUR', 'USD', 'USD'],
        'timestamp': pd.to_datetime(['2024-01-15 10:30', '2024-01-15 11:15', '2024-01-15 14:22', '2024-01-15 15:45', '2024-01-15 16:30'])
    })
    
    print("Transaction data:")
    print(df.to_string(index=False))
    
    # Create observable engine
    print("\n1. Creating observable engine with monitoring...")
    try:
        engine = create_minimal_observable_engine()
        print("✅ Observable engine created successfully!")
        
        # Generate row IDs with observability
        print("\n2. Generating row IDs with full observability...")
        result_df, selected_columns, audit_trail = engine.generate_unique_row_ids(
            df=df,
            columns=['transaction_id', 'user_email'],
            show_progress=True
        )
        
        print(f"✅ Observable row ID generation completed!")
        print(f"Selected columns: {selected_columns}")
        print(f"Audit trail keys: {list(audit_trail.keys()) if audit_trail else 'None'}")
        
        print("\nResult:")
        print(result_df[['row_id', 'transaction_id', 'user_email', 'amount']].to_string(index=False))
        
        # Get system health report
        print("\n3. System health report...")
        health_report = engine.get_system_health_report()
        print(f"Session metrics: {health_report.get('session_metrics', {})}")
        print(f"System resources: {health_report.get('system_resources', {})}")
        
        # Get session summary
        print("\n4. Session summary...")
        session_summary = engine.get_session_summary()
        print(f"Operations completed: {session_summary.get('operation_count', 0)}")
        print(f"Success rate: {session_summary.get('success_rate', 0):.1%}")
        
    except Exception as e:
        print(f"❌ Error with observable API: {e}")
        import traceback
        traceback.print_exc()

def error_handling_example():
    """Demonstrate error handling capabilities."""
    print("\n\n🚨 ERROR HANDLING EXAMPLE")
    print("=" * 50)
    
    # Test various error conditions
    error_cases = [
        ("Empty DataFrame", pd.DataFrame()),
        ("All NULL values", pd.DataFrame({'col1': [None, None], 'col2': [None, None]})),
        ("No valid columns", pd.DataFrame({'col1': ['duplicate', 'duplicate'], 'col2': ['same', 'same']})),
    ]
    
    for case_name, test_df in error_cases:
        print(f"\nTesting: {case_name}")
        try:
            result = generate_unique_row_ids(
                test_df,
                show_warnings=True,
                enable_quality_checks=True
            )
            print(f"  ✅ Handled gracefully - {len(result)} rows")
        except Exception as e:
            print(f"  ⚠️ Expected error: {type(e).__name__}: {e}")

def main():
    """Run all examples."""
    print("🚀 ROW ID GENERATOR - COMPREHENSIVE EXAMPLES")
    print("=" * 60)
    print("This example demonstrates all major features of the row-id-generator package")
    print("including basic usage, advanced configuration, performance variants,")
    print("utility functions, observable API, and error handling.")
    print("=" * 60)
    
    try:
        # Run all examples
        basic_usage_example()
        advanced_configuration_example()
        performance_results = performance_variants_example()
        utility_functions_example()
        observable_api_example()
        error_handling_example()
        
        print("\n\n🎉 ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        # Summary
        if performance_results:
            print("\nPerformance Summary:")
            for variant, metrics in performance_results.items():
                if 'error' not in metrics:
                    print(f"  {variant}: {metrics['throughput']:,.0f} rows/sec")
        
        print("\nNext Steps:")
        print("1. Check out the full API documentation")
        print("2. Explore advanced configuration options")
        print("3. Set up observability for production use")
        print("4. Integrate with your Snowflake workflow")
        
    except Exception as e:
        print(f"\n❌ Example execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 