"""
Advanced usage examples for row-id-generator package.

This file demonstrates advanced features including:
- Observable engines for production monitoring
- Error handling and recovery patterns
- Performance optimization techniques
- Integration with external systems
- Custom configuration management
"""

import pandas as pd
import time
import logging
import os
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

from row_id_generator import (
    generate_unique_row_ids,
    generate_row_ids_simple,
    generate_row_ids_fast,
    create_observable_engine,
    create_minimal_observable_engine,
    create_full_observable_engine,
    select_columns_for_hashing,
    prepare_data_for_hashing,
    analyze_dataframe_quality,
    get_column_quality_score,
    validate_dataframe_input,
    DataValidationError,
    HashGenerationError,
    RowIDGenerationError
)


def production_monitoring_example():
    """Demonstrate production-ready monitoring and observability."""
    print("🔍 PRODUCTION MONITORING EXAMPLE")
    print("=" * 50)
    
    # Create comprehensive observable engine
    engine = create_observable_engine()
    
    # Sample production data
    df = pd.DataFrame({
        'customer_id': ['CUST001', 'CUST002', 'CUST003', 'CUST004'],
        'email': ['alice@corp.com', 'bob@startup.io', 'charlie@agency.co', 'diana@enterprise.net'],
        'transaction_amount': [99.99, 149.50, 79.99, 299.99],
        'timestamp': pd.to_datetime([
            '2024-01-15 10:30:00',
            '2024-01-15 11:15:00', 
            '2024-01-15 14:22:00',
            '2024-01-15 16:45:00'
        ]),
        'payment_method': ['credit_card', 'paypal', 'bank_transfer', 'credit_card']
    })
    
    print(f"Processing {len(df)} transaction records...")
    
    # Process with full observability
    start_time = time.time()
    result_df, selected_columns, audit_trail = engine.generate_unique_row_ids(
        df=df,
        show_progress=True
    )
    processing_time = time.time() - start_time
    
    print(f"✅ Processed in {processing_time:.3f} seconds")
    print(f"📊 Selected columns: {selected_columns}")
    print(f"🔍 Audit trail keys: {list(audit_trail.keys())}")
    
    # System health monitoring
    health_report = engine.get_system_health_report()
    print(f"\n📈 SYSTEM HEALTH:")
    print(f"   CPU Usage: {health_report['system_resources']['cpu_percent']:.1f}%")
    print(f"   Memory Usage: {health_report['system_resources']['memory_percent']:.1f}%")
    
    # Session metrics
    session_summary = engine.get_session_summary()
    print(f"\n📋 SESSION METRICS:")
    print(f"   Operations: {session_summary['operation_count']}")
    print(f"   Success Rate: {session_summary['success_rate']:.1%}")
    
    # Export metrics for external monitoring
    try:
        metrics_json = engine.export_metrics("json")
        prometheus_metrics = engine.export_metrics("prometheus")
        print(f"\n📊 Exported metrics (JSON keys): {list(metrics_json.keys())}")
        print(f"📊 Prometheus metrics length: {len(prometheus_metrics)} characters")
    except Exception as e:
        print(f"⚠️ Metrics export error: {e}")
    
    # Generate performance dashboard
    try:
        dashboard_html = engine.generate_performance_dashboard()
        with open('performance_dashboard.html', 'w') as f:
            f.write(dashboard_html)
        print(f"📊 Performance dashboard saved to performance_dashboard.html")
    except Exception as e:
        print(f"⚠️ Dashboard generation error: {e}")
    
    return result_df


def error_handling_and_recovery_example():
    """Demonstrate robust error handling and recovery patterns."""
    print("\n🛡️ ERROR HANDLING AND RECOVERY EXAMPLE")
    print("=" * 50)
    
    def safe_row_id_generation(df: pd.DataFrame, max_retries: int = 3) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """Production-ready row ID generation with error handling."""
        
        for attempt in range(max_retries):
            try:
                # Attempt with full validation
                result_df = generate_unique_row_ids(
                    df,
                    enable_quality_checks=True,
                    uniqueness_threshold=0.95,
                    show_warnings=False  # Don't spam logs in production
                )
                return result_df, None
                
            except DataValidationError as e:
                if attempt < max_retries - 1:
                    print(f"⚠️ Attempt {attempt + 1}: Data validation failed, trying fallback...")
                    # Try with lower quality requirements
                    try:
                        result_df = generate_unique_row_ids(
                            df,
                            uniqueness_threshold=0.8,
                            enable_quality_checks=False
                        )
                        return result_df, f"Used fallback configuration: {e.message if hasattr(e, 'message') else str(e)}"
                    except Exception as fallback_error:
                        print(f"⚠️ Fallback also failed: {fallback_error}")
                        continue
                else:
                    return None, f"Data validation failed: {e}"
                    
            except HashGenerationError as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"⚠️ Attempt {attempt + 1}: Hash generation failed, waiting {wait_time}s...")
                    time.sleep(wait_time)  # Exponential backoff
                    continue
                else:
                    return None, f"Hash generation failed: {e}"
                    
            except Exception as e:
                return None, f"Unexpected error: {str(e)}"
        
        return None, "Max retries exceeded"
    
    # Test with problematic data
    problematic_df = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'data': ['A', 'B', 'C', 'D', 'E'],  # Low uniqueness
        'nulls': [None, None, 'value', None, None]  # Mostly nulls
    })
    
    print("Testing with low-quality data...")
    result_df, error_message = safe_row_id_generation(problematic_df)
    
    if result_df is not None:
        print(f"✅ Successfully processed {len(result_df)} rows")
        if error_message:
            print(f"⚠️ Warning: {error_message}")
        print(f"📊 Result columns: {list(result_df.columns)}")
    else:
        print(f"❌ Processing failed: {error_message}")
    
    return result_df


def performance_optimization_example():
    """Demonstrate performance optimization techniques."""
    print("\n⚡ PERFORMANCE OPTIMIZATION EXAMPLE")
    print("=" * 50)
    
    # Create larger sample dataset
    large_df = pd.DataFrame({
        'user_id': [f'USER_{i:06d}' for i in range(10000)],
        'email': [f'user{i}@domain{i%100}.com' for i in range(10000)],
        'department': [f'Dept_{i%20}' for i in range(10000)],
        'salary': [50000 + (i % 100000) for i in range(10000)],
        'join_date': pd.date_range('2020-01-01', periods=10000, freq='1H')
    })
    
    print(f"Created dataset with {len(large_df):,} rows")
    
    # Test different approaches
    approaches = [
        ("Standard approach", lambda df: generate_unique_row_ids(df, show_progress=False)),
        ("Simple approach", lambda df: generate_row_ids_simple(df, columns=['user_id', 'email'])),
        ("Fast approach", lambda df: generate_row_ids_fast(df, columns=['user_id', 'email'])),
        ("No monitoring", lambda df: generate_unique_row_ids(
            df, 
            enable_monitoring=False,
            enable_quality_checks=False,
            show_progress=False
        ))
    ]
    
    results = {}
    
    for name, func in approaches:
        print(f"\n🔄 Testing {name}...")
        start_time = time.time()
        
        try:
            result_df = func(large_df)
            processing_time = time.time() - start_time
            throughput = len(large_df) / processing_time
            
            results[name] = {
                'time': processing_time,
                'throughput': throughput,
                'success': True,
                'rows': len(result_df)
            }
            
            print(f"   ✅ Time: {processing_time:.3f}s")
            print(f"   📈 Throughput: {throughput:,.0f} rows/sec")
            
        except Exception as e:
            results[name] = {
                'time': None,
                'throughput': None,
                'success': False,
                'error': str(e)
            }
            print(f"   ❌ Failed: {e}")
    
    # Performance comparison
    print(f"\n📊 PERFORMANCE COMPARISON:")
    successful_results = {k: v for k, v in results.items() if v['success']}
    
    if successful_results:
        fastest = min(successful_results.items(), key=lambda x: x[1]['time'])
        highest_throughput = max(successful_results.items(), key=lambda x: x[1]['throughput'])
        
        print(f"   🏆 Fastest: {fastest[0]} ({fastest[1]['time']:.3f}s)")
        print(f"   🚀 Highest throughput: {highest_throughput[0]} ({highest_throughput[1]['throughput']:,.0f} rows/sec)")
    
    return results


def batch_processing_example():
    """Demonstrate batch processing patterns for large datasets."""
    print("\n📦 BATCH PROCESSING EXAMPLE")
    print("=" * 50)
    
    def process_data_in_batches(df: pd.DataFrame, batch_size: int = 1000) -> pd.DataFrame:
        """Process large DataFrame in batches."""
        
        print(f"Processing {len(df):,} rows in batches of {batch_size:,}...")
        
        if len(df) <= batch_size:
            # Small enough to process directly
            return generate_unique_row_ids(df, show_progress=False)
        
        # Split into batches
        batches = [df[i:i+batch_size] for i in range(0, len(df), batch_size)]
        print(f"Created {len(batches)} batches")
        
        batch_results = []
        total_start_time = time.time()
        
        for i, batch in enumerate(batches):
            batch_start_time = time.time()
            
            try:
                batch_result = generate_unique_row_ids(
                    batch,
                    show_progress=False,
                    enable_monitoring=False
                )
                batch_results.append(batch_result)
                
                batch_time = time.time() - batch_start_time
                print(f"   ✅ Batch {i+1}/{len(batches)}: {len(batch)} rows in {batch_time:.3f}s")
                
            except Exception as e:
                print(f"   ❌ Batch {i+1}/{len(batches)} failed: {e}")
                # Continue with other batches
                continue
        
        if not batch_results:
            raise RuntimeError("All batches failed")
        
        # Combine results
        combined_result = pd.concat(batch_results, ignore_index=True)
        total_time = time.time() - total_start_time
        
        print(f"✅ Combined {len(batch_results)} batches into {len(combined_result):,} rows")
        print(f"📊 Total time: {total_time:.3f}s ({len(combined_result)/total_time:,.0f} rows/sec)")
        
        return combined_result
    
    # Create medium-sized dataset for batching
    medium_df = pd.DataFrame({
        'transaction_id': [f'TXN_{i:08d}' for i in range(5000)],
        'customer_email': [f'customer{i}@company{i%50}.com' for i in range(5000)],
        'product_sku': [f'SKU_{i%200:04d}' for i in range(5000)],
        'amount': [10.00 + (i % 1000) for i in range(5000)]
    })
    
    # Process in batches
    batch_result = process_data_in_batches(medium_df, batch_size=1000)
    
    print(f"📊 Final result: {len(batch_result)} rows with columns {list(batch_result.columns)}")
    
    return batch_result


def data_quality_analysis_example():
    """Demonstrate comprehensive data quality analysis."""
    print("\n🔍 DATA QUALITY ANALYSIS EXAMPLE")
    print("=" * 50)
    
    # Create dataset with various quality issues
    quality_test_df = pd.DataFrame({
        'perfect_column': [f'UNIQUE_{i}' for i in range(100)],  # Perfect uniqueness
        'good_column': [f'GOOD_{i//2}' for i in range(100)],    # 50% uniqueness
        'poor_column': [f'POOR_{i//10}' for i in range(100)],   # 10% uniqueness
        'terrible_column': ['SAME_VALUE'] * 100,                # No uniqueness
        'nulls_column': [f'VALUE_{i}' if i % 3 == 0 else None for i in range(100)],  # Lots of nulls
        'mixed_types': [str(i) if i % 2 == 0 else i for i in range(100)]  # Mixed types
    })
    
    print(f"Analyzing quality of {len(quality_test_df)} rows...")
    
    # Comprehensive quality analysis
    try:
        quality_metrics = analyze_dataframe_quality(quality_test_df, "Quality test dataset")
        summary = quality_metrics.get_summary_report()
        
        print(f"\n📊 OVERALL QUALITY:")
        print(f"   Score: {summary.get('overall_score', 'N/A')}")
        print(f"   Grade: {summary.get('grade', 'N/A')}")
        
        recommendations = quality_metrics.metrics.get('recommendations', [])
        if recommendations:
            print(f"\n💡 RECOMMENDATIONS:")
            for i, rec in enumerate(recommendations[:5], 1):  # Show top 5
                print(f"   {i}. {rec}")
    
    except Exception as e:
        print(f"⚠️ Quality analysis error: {e}")
    
    # Individual column analysis
    print(f"\n📋 COLUMN QUALITY SCORES:")
    for column in quality_test_df.columns:
        try:
            score_info = get_column_quality_score(quality_test_df, column)
            score = score_info.get('score', 0)
            grade = score_info.get('grade', 'Unknown')
            print(f"   {column:15s}: {score:5.2f} ({grade})")
        except Exception as e:
            print(f"   {column:15s}: Error - {e}")
    
    # Automatic column selection
    print(f"\n🎯 COLUMN SELECTION:")
    try:
        selected_columns = select_columns_for_hashing(
            quality_test_df,
            uniqueness_threshold=0.5  # Lower threshold for this example
        )
        print(f"   Selected: {selected_columns}")
        
        # Generate row IDs with selected columns
        result_df = generate_unique_row_ids(
            quality_test_df,
            columns=selected_columns,
            show_warnings=True
        )
        print(f"   ✅ Generated row IDs using {len(selected_columns)} columns")
        
    except Exception as e:
        print(f"   ❌ Column selection failed: {e}")
    
    return quality_test_df


def integration_patterns_example():
    """Demonstrate integration with external systems."""
    print("\n🔗 INTEGRATION PATTERNS EXAMPLE")
    print("=" * 50)
    
    def create_etl_pipeline(source_data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate an ETL pipeline with row ID generation."""
        
        pipeline_start = time.time()
        results = {
            'success': False,
            'error': None,
            'metrics': {}
        }
        
        try:
            # Extract (already done - source_data)
            extract_time = 0.1  # Simulated
            print(f"📥 Extract: {len(source_data)} rows loaded")
            
            # Transform: Add row IDs
            transform_start = time.time()
            
            if config.get('use_observable', False):
                engine = create_minimal_observable_engine()
                transformed_df, selected_columns, audit_trail = engine.generate_unique_row_ids(
                    df=source_data,
                    **config.get('row_id_params', {})
                )
                
                # Collect metrics from observable engine
                session_metrics = engine.get_session_summary()
                results['metrics'].update({
                    'selected_columns': selected_columns,
                    'audit_trail': audit_trail,
                    'session_metrics': session_metrics
                })
            else:
                transformed_df = generate_unique_row_ids(
                    source_data,
                    **config.get('row_id_params', {})
                )
            
            transform_time = time.time() - transform_start
            print(f"🔄 Transform: Row IDs added in {transform_time:.3f}s")
            
            # Load (simulated)
            load_start = time.time()
            load_time = 0.2  # Simulated
            time.sleep(load_time)
            print(f"📤 Load: {len(transformed_df)} rows written to target")
            
            total_time = time.time() - pipeline_start
            
            results.update({
                'success': True,
                'processed_rows': len(transformed_df),
                'final_columns': list(transformed_df.columns),
                'timing': {
                    'extract': extract_time,
                    'transform': transform_time,
                    'load': load_time,
                    'total': total_time
                }
            })
            
            print(f"✅ Pipeline completed in {total_time:.3f}s")
            
        except Exception as e:
            results['error'] = str(e)
            print(f"❌ Pipeline failed: {e}")
        
        return results
    
    # Sample configuration
    etl_config = {
        'use_observable': True,
        'row_id_params': {
            'uniqueness_threshold': 0.9,
            'show_progress': False
        }
    }
    
    # Sample source data
    source_df = pd.DataFrame({
        'order_id': [f'ORD_{i:05d}' for i in range(500)],
        'customer_email': [f'cust{i}@example{i%10}.com' for i in range(500)],
        'product_name': [f'Product_{i%50}' for i in range(500)],
        'order_date': pd.date_range('2024-01-01', periods=500, freq='1H'),
        'amount': [20.00 + (i % 200) for i in range(500)]
    })
    
    # Run ETL pipeline
    pipeline_result = create_etl_pipeline(source_df, etl_config)
    
    if pipeline_result['success']:
        print(f"\n📊 PIPELINE METRICS:")
        timing = pipeline_result['timing']
        for stage, duration in timing.items():
            print(f"   {stage.title()}: {duration:.3f}s")
        
        if 'metrics' in pipeline_result and 'session_metrics' in pipeline_result['metrics']:
            session = pipeline_result['metrics']['session_metrics']
            print(f"   Success Rate: {session.get('success_rate', 0):.1%}")
    
    return pipeline_result


def configuration_management_example():
    """Demonstrate advanced configuration management."""
    print("\n⚙️ CONFIGURATION MANAGEMENT EXAMPLE")
    print("=" * 50)
    
    # Define different environment configurations
    configurations = {
        'development': {
            'enable_monitoring': True,
            'enable_quality_checks': True,
            'show_progress': True,
            'show_warnings': True,
            'uniqueness_threshold': 0.95,
            'separator': '|'
        },
        'testing': {
            'enable_monitoring': False,
            'enable_quality_checks': True,
            'show_progress': False,
            'show_warnings': False,
            'uniqueness_threshold': 0.9,
            'separator': '::'
        },
        'production': {
            'enable_monitoring': False,
            'enable_quality_checks': False,
            'show_progress': False,
            'show_warnings': False,
            'uniqueness_threshold': 0.85,
            'separator': '|'
        }
    }
    
    # Sample data
    config_test_df = pd.DataFrame({
        'id': ['ID001', 'ID002', 'ID003'],
        'name': ['Alice', 'Bob', 'Charlie'],
        'email': ['alice@test.com', 'bob@test.com', 'charlie@test.com']
    })
    
    # Test each configuration
    for env_name, config in configurations.items():
        print(f"\n🔧 Testing {env_name.upper()} configuration:")
        
        start_time = time.time()
        try:
            result_df = generate_unique_row_ids(config_test_df, **config)
            processing_time = time.time() - start_time
            
            print(f"   ✅ Success: {len(result_df)} rows in {processing_time:.3f}s")
            print(f"   📊 Config: monitoring={config['enable_monitoring']}, "
                  f"quality={config['enable_quality_checks']}, "
                  f"threshold={config['uniqueness_threshold']}")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
    
    # Environment-specific best practices
    print(f"\n💡 ENVIRONMENT BEST PRACTICES:")
    print(f"   🔧 Development: Full monitoring, verbose output, high quality thresholds")
    print(f"   🧪 Testing: Quality checks enabled, minimal output, moderate thresholds")
    print(f"   🚀 Production: Optimized for performance, minimal overhead, robust thresholds")


def main():
    """Run all advanced examples."""
    print("🚀 ROW ID GENERATOR - ADVANCED USAGE EXAMPLES")
    print("=" * 60)
    
    examples = [
        production_monitoring_example,
        error_handling_and_recovery_example,
        performance_optimization_example,
        batch_processing_example,
        data_quality_analysis_example,
        integration_patterns_example,
        configuration_management_example
    ]
    
    for example_func in examples:
        try:
            result = example_func()
            print(f"✅ {example_func.__name__} completed successfully")
        except Exception as e:
            print(f"❌ {example_func.__name__} failed: {e}")
        finally:
            print()  # Add spacing between examples
    
    print("🎉 All advanced examples completed!")


if __name__ == "__main__":
    # Configure logging for examples
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    main() 